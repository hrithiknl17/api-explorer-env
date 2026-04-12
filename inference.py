"""
Inference script for API Explorer OpenEnv.

Environment variables (mandatory):
    API_BASE_URL      LLM endpoint (default: HF router)
    MODEL_NAME        Model identifier (default: Llama-3.3-70B)
    HF_TOKEN          HuggingFace / API key (no default)
    LOCAL_IMAGE_NAME  Local Docker image name if using from_docker_image() (optional)

Stdout format (mandatory — must match exactly):
    [START] task=<task> env=<env> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, List, Optional

from openai import OpenAI

from api_explorer_env.openenv_env import APIExplorerOpenEnv
from api_explorer_env.openenv_models import APIAction, APIObservation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/together/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "api-explorer"
SUCCESS_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are an API explorer agent. You are given access to an undocumented REST API.
Your job is to complete tasks by calling API endpoints and reasoning about responses.

At each step you must return exactly one JSON object with these fields:
- method: one of GET, POST, PUT, PATCH, DELETE
- path: the API path to call (e.g. "/merchants" or "/disputes/dsp_001/resolve")
- body: a JSON object of query params or request body (use {} if none)
- headers: a JSON object of headers (use {} if none)
- submit_answer: null, OR your final string answer for the current task

Rules:
- Explore systematically. Try /merchants, /transactions, /disputes, /summary to discover routes.
- When you are confident you know the answer to the current task, set submit_answer to your answer.
- Keep api_calls_used low — efficiency is rewarded.
- Do NOT include markdown fences or extra text. Return only the JSON object.

Example response:
{"method": "GET", "path": "/merchants", "body": {}, "headers": {}, "submit_answer": null}
"""


# ── Structured stdout logging (mandatory format) ────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM interaction ─────────────────────────────────────────────────────────────

def build_prompt(obs: APIObservation) -> str:
    return json.dumps({
        "last_api_response": {
            "status_code": obs.status_code,
            "body": obs.response_body,
        },
        "current_task": {
            "name": obs.current_task.name,
            "description": obs.current_task.description,
            "difficulty": obs.current_task.difficulty,
        },
        "api_calls_used": obs.api_calls_used,
        "api_calls_limit": obs.api_calls_limit,
        "tasks_remaining": obs.tasks_remaining,
        "message": obs.message,
    }, indent=2)


def parse_action(text: str) -> APIAction:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model response: {text!r}")
    data = json.loads(text[start:end + 1])
    return APIAction(**data)


def query_model(
    client: OpenAI,
    obs: APIObservation,
    history: List[dict],
) -> tuple[APIAction, str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-8:])  # keep last 4 turns
    messages.append({"role": "user", "content": build_prompt(obs)})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=messages,
    )
    content = completion.choices[0].message.content or ""
    return parse_action(content), content


# ── Episode runner ──────────────────────────────────────────────────────────────

def run_episode(scenario: str, seed: int, max_steps: int) -> dict[str, Any]:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = APIExplorerOpenEnv(scenario=scenario, seed=seed)

    task_name = f"{scenario}-tasks"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    obs: Optional[APIObservation] = None

    try:
        obs = env.reset(seed=seed, scenario=scenario)
        history: List[dict] = []

        for step_num in range(1, max_steps + 1):
            error: Optional[str] = None

            try:
                action, raw = query_model(client, obs, history)
            except Exception as exc:
                error = str(exc)
                log_step(step=step_num, action="parse_error", reward=0.0, done=True, error=error)
                rewards.append(0.0)
                steps_taken = step_num
                break

            try:
                obs = env.step(action)
            except Exception as exc:
                error = str(exc)
                log_step(step=step_num, action=str(action), reward=0.0, done=True, error=error)
                rewards.append(0.0)
                steps_taken = step_num
                break

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            rewards.append(reward)
            steps_taken = step_num

            # Compact action string for the log line
            if action.submit_answer is not None:
                action_str = f"submit:{action.submit_answer}"
            else:
                action_str = f"{action.method}:{action.path}"

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            history.append({"role": "user", "content": build_prompt(obs)})
            history.append({"role": "assistant", "content": raw})

            if done:
                break

        # Aggregate score = mean of final task scores, clamped to [0, 1]
        if obs is not None and obs.task_scores:
            score = sum(obs.task_scores.values()) / len(obs.task_scores)
            score = min(max(score, 0.0), 1.0)

        success = score >= SUCCESS_THRESHOLD

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "steps": steps_taken,
        "score": score,
        "success": success,
        "final_task_scores": obs.task_scores if obs is not None else {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference on the API Explorer OpenEnv.")
    parser.add_argument("--scenario", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_episode(scenario=args.scenario, seed=args.seed, max_steps=args.max_steps)
