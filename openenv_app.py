from __future__ import annotations

from openenv.core.env_server import create_app

from api_explorer_env.openenv_env import APIExplorerOpenEnv
from api_explorer_env.openenv_models import APIAction, APIObservation


app = create_app(
    env=lambda: APIExplorerOpenEnv(),
    action_cls=APIAction,
    observation_cls=APIObservation,
    env_name="api-explorer",
)


@app.get("/")
def root():
    return {"status": "ok", "env": "api-explorer", "version": "0.1.0"}
