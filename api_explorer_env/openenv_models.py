from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


class APIAction(Action):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    path: str = Field(description="API path to call, e.g. /merchants or /disputes/dsp_001/resolve")
    body: Dict[str, Any] = Field(default_factory=dict, description="Request body / query params")
    headers: Dict[str, str] = Field(default_factory=dict)
    submit_answer: Optional[str] = Field(
        default=None,
        description="Submit your final answer for the current task. Set this when you know the answer.",
    )


class TaskStatus(BaseModel):
    name: str
    description: str
    difficulty: str
    score: float
    completed: bool


class APIObservation(Observation):
    status_code: int
    response_body: Any
    current_task: TaskStatus
    tasks_remaining: int
    api_calls_used: int
    api_calls_limit: int
    message: str
    task_scores: Dict[str, float] = Field(default_factory=dict)


class APIState(State):
    episode_id: str = "init"
    scenario: str = "medium"
    step_count: int = 0
    api_calls_used: int = 0
    tasks_completed: int = 0
    tasks_total: int = 0
