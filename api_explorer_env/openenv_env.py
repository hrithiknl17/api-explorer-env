from __future__ import annotations

import uuid
from typing import Optional

from openenv.core.env_server import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from .mock_api import MockAPI
from .openenv_models import APIAction, APIObservation, APIState, TaskStatus
from .tasks import get_tasks


SCENARIO_CALL_LIMITS = {
    "easy":   15,
    "medium": 25,
    "hard":   40,
}


class APIExplorerOpenEnv(Environment[APIAction, APIObservation, APIState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, scenario: str = "medium", seed: int = 42):
        super().__init__()
        self._scenario = scenario
        self._seed = seed
        self._api = MockAPI(seed=seed)
        self._tasks = get_tasks(scenario, self._api.db)
        self._task_idx = 0
        self._call_limit = SCENARIO_CALL_LIMITS.get(scenario, 25)
        self._state = APIState(
            scenario=scenario,
            tasks_total=len(self._tasks),
        )

    @property
    def state(self) -> APIState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="APIExplorerOpenEnv",
            description=(
                "An LLM agent explores an undocumented REST API and completes tasks "
                "by calling endpoints, inferring structure, and reasoning about responses."
            ),
            version="1.0.0",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        scenario: Optional[str] = None,
        **kwargs,
    ) -> APIObservation:
        self._reset_rubric()
        effective_scenario = scenario or self._scenario
        effective_seed = seed if seed is not None else self._seed

        self._scenario = effective_scenario
        self._api = MockAPI(seed=effective_seed)
        self._tasks = get_tasks(effective_scenario, self._api.db)
        self._task_idx = 0
        self._call_limit = SCENARIO_CALL_LIMITS.get(effective_scenario, 25)

        self._state = APIState(
            episode_id=episode_id or str(uuid.uuid4()),
            scenario=effective_scenario,
            step_count=0,
            api_calls_used=0,
            tasks_completed=0,
            tasks_total=len(self._tasks),
        )

        return self._build_observation(
            status_code=200,
            response_body={
                "message": "Welcome. This is an undocumented API. Explore to complete your tasks.",
                "hint": "Try calling common REST paths like /merchants, /transactions, /disputes",
            },
            reward=None,
            done=False,
            message=f"Episode started. Scenario: {effective_scenario}. "
                    f"Complete {len(self._tasks)} tasks using at most {self._call_limit} API calls.",
        )

    def step(self, action: APIAction, **kwargs) -> APIObservation:
        self._state.step_count += 1
        reward = 0.0
        done = False

        current_task = self._tasks[self._task_idx] if self._task_idx < len(self._tasks) else None

        # ── Handle answer submission ───────────────────────────────────────────
        if action.submit_answer is not None and current_task:
            score = current_task.verify(
                action.submit_answer, self._api.db, self._api.total_calls
            )
            reward = score

            if current_task.completed:
                self._state.tasks_completed += 1
                self._task_idx += 1
                if self._task_idx >= len(self._tasks):
                    done = True
                    message = f"All tasks complete! Final scores: {self._compute_task_scores()}"
                else:
                    next_task = self._tasks[self._task_idx]
                    message = (
                        f"Task '{current_task.name}' score: {score:.2f}. "
                        f"Next task: {next_task.description}"
                    )
            else:
                message = f"Wrong answer for '{current_task.name}'. Score: {score:.2f}. Keep exploring."

            self._state.api_calls_used = self._api.total_calls
            return self._build_observation(
                status_code=200,
                response_body={"answer_received": action.submit_answer, "score": score},
                reward=reward,
                done=done,
                message=message,
            )

        # ── Handle API call ───────────────────────────────────────────────────
        if self._api.total_calls >= self._call_limit:
            done = True
            return self._build_observation(
                status_code=429,
                response_body={"error": "API call limit reached", "limit": self._call_limit},
                reward=-0.1,
                done=True,
                message=f"Call limit {self._call_limit} reached. Episode over.",
            )

        status_code, response_body = self._api.call(
            method=action.method,
            path=action.path,
            body=action.body or None,
            headers=action.headers or None,
        )
        self._state.api_calls_used = self._api.total_calls

        # Small penalty for 404s (exploring wrong paths)
        if status_code == 404:
            reward -= 0.02

        message = (
            f"API call {self._api.total_calls}/{self._call_limit}: "
            f"{action.method} /{action.path.strip('/')} → {status_code}. "
            f"Current task: {current_task.description if current_task else 'None'}"
        )

        return self._build_observation(
            status_code=status_code,
            response_body=response_body,
            reward=reward,
            done=done,
            message=message,
        )

    def close(self) -> None:
        pass

    def _build_observation(
        self,
        status_code: int,
        response_body,
        reward,
        done: bool,
        message: str,
    ) -> APIObservation:
        current_task = self._tasks[self._task_idx] if self._task_idx < len(self._tasks) else None
        tasks_remaining = len(self._tasks) - self._task_idx

        task_status = TaskStatus(
            name=current_task.name if current_task else "done",
            description=current_task.description if current_task else "All tasks completed.",
            difficulty=current_task.difficulty if current_task else "-",
            score=current_task.score if current_task else 1.0,
            completed=current_task.completed if current_task else True,
        )

        obs = APIObservation(
            done=done,
            reward=reward,
            status_code=status_code,
            response_body=response_body,
            current_task=task_status,
            tasks_remaining=tasks_remaining,
            api_calls_used=self._api.total_calls,
            api_calls_limit=self._call_limit,
            message=message,
            task_scores=self._compute_task_scores(),
        )
        return self._apply_transform(obs)

    def _compute_task_scores(self) -> dict[str, float]:
        scores = {}
        for task in self._tasks:
            scores[task.name] = round(task.score, 4)
        return scores
