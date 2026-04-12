from __future__ import annotations

import argparse
import json
import os
from typing import Any

from openai import OpenAI

from api_explorer_env.openenv_env import APIExplorerOpenEnv
from api_explorer_env.openenv_models import APIAction, APIObservation


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/together/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

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
        raise ValueError(f"No JSON in response: {text}")
    data = json.loads(text[start:end + 1])
    return APIAction(**data)


def query_model(client: OpenAI, obs: APIObservation, history: list[dict]) -> APIAction:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])  # keep last 3 turns in context
    messages.append({"role": "user", "content": build_prompt(obs)})

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=messages,
    )
    content = completion.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": content})
    return parse_action(content)


def run_episode(scenario: str, seed: int, max_steps: int) -> dict[str, Any]:
    api_key = HF_TOKEN or "local"
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = APIExplorerOpenEnv(scenario=scenario, seed=seed)

    obs = env.reset(seed=seed, scenario=scenario)
    history: list[dict] = []
    trajectory: list[dict] = []

    print(json.dumps({"event": "[START]", "scenario": scenario, "seed": seed, "max_steps": max_steps}))

    for step_num in range(max_steps):
        action = query_model(client, obs, history)
        obs = env.step(action)

        step_log = {
            "event": "[STEP]",
            "step": step_num + 1,
            "action": action.model_dump(),
            "status_code": obs.status_code,
            "reward": obs.reward,
            "done": obs.done,
            "task_scores": obs.task_scores,
            "api_calls_used": obs.api_calls_used,
        }
        print(json.dumps(step_log))
        trajectory.append(step_log)

        history.append({"role": "user", "content": build_prompt(obs)})

        if obs.done:
            break

    summary = {
        "steps": len(trajectory),
        "final_task_scores": obs.task_scores,
        "final_state": env.state.model_dump(),
        "trajectory": trajectory,
    }
    env.close()
    print(json.dumps({"event": "[END]", "steps": len(trajectory), "final_task_scores": obs.task_scores}))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference on the API Explorer OpenEnv.")
    parser.add_argument("--scenario", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_episode(scenario=args.scenario, seed=args.seed, max_steps=args.max_steps)
    print(json.dumps(result, indent=2))
