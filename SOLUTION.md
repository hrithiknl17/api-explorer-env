# API Explorer — Solution Summary

## Problem Statement

Most RL environments test *optimization* — find the best policy through trial and error. This environment tests something different: **reasoning under uncertainty**.

An LLM agent is dropped into an undocumented REST API with no schema, no docs, and no prior knowledge of the routes. It must:
1. Discover what endpoints exist by calling them
2. Understand the data model from raw JSON responses
3. Complete real-world tasks (aggregate data, filter records, mutate state)
4. Do all of this using the *minimum* number of API calls

This is a task where LLMs have a structural advantage over traditional RL agents — they can read a JSON response and reason about what to call next. No training required.

## Environment Design

### The Simulated API

The mock API models a fictional payment platform (merchants, transactions, disputes). It has 9 discoverable routes:

```
GET  /merchants                      — paginated merchant list
GET  /merchants/{id}                 — single merchant
GET  /merchants/{id}/transactions    — transactions for a merchant
GET  /transactions                   — paginated, filterable by status
GET  /transactions/{id}              — single transaction
GET  /disputes                       — list disputes, filterable by status
GET  /disputes/{id}                  — single dispute
POST /disputes/{id}/resolve          — mutate: resolve a dispute
GET  /summary                        — aggregate stats (total volume, failed count)
```

The agent does not know these routes exist. It discovers them by calling paths and reading 404 or 200 responses.

### Action Space

Each step the agent submits an `APIAction`:
- `method`: HTTP verb (GET/POST/PUT/PATCH/DELETE)
- `path`: route to call
- `body`: query params or request body
- `headers`: optional headers
- `submit_answer`: when set, submits the agent's answer for the current task

### Observation Space

Each step returns an `APIObservation`:
- `status_code` + `response_body`: raw API response
- `current_task`: name, description, difficulty of the active task
- `api_calls_used` / `api_calls_limit`: budget tracking
- `task_scores`: per-task scores in `[0.0, 1.0]`
- `message`: natural language feedback

### Reward & Task Scoring

Tasks are scored 0.0–1.0 with an efficiency bonus:
- Correct answer = base score of 1.0
- Fewer API calls than the expected threshold → score stays at 1.0
- Extra calls reduce score toward 0.5 minimum (never penalized to 0 for being slow)
- Wrong answer = 0.0 (keep exploring)
- 404 response = −0.02 per bad call (mild exploration penalty)

### Scenarios

| Scenario | Call Limit | Tasks |
|---|---|---|
| easy | 15 | Count merchants, merchant status, total volume |
| medium | 25 | Count merchants, count failed transactions, merchant volume |
| hard | 40 | Count failed transactions, richest merchant, resolve oldest dispute |

The hard scenario requires pagination across multiple pages and a state-mutating POST call.

## Why This Is a Strong Benchmark

1. **Tests reasoning, not optimization.** A gradient-based agent cannot train on this — there is no meaningful signal until the agent can read JSON. An LLM reads the response and reasons about the next call immediately.

2. **Real-world task structure.** The tasks mirror actual payment platform operations: data aggregation, status filtering, dispute resolution.

3. **Efficiency is measurable.** The call budget creates a clean signal for how well the model understands the API vs. brute-force exploring it.

4. **Difficulty scales naturally.** Easy tasks need 2-3 calls. Hard tasks require pagination, cross-referencing records, and state mutation — skills that chain on each other.

5. **LLM-native.** The environment is designed so that a capable LLM with tool-use abilities should achieve near-perfect scores on easy/medium scenarios without any fine-tuning.

## Repository Layout

```
api_explorer_env/
  __init__.py
  mock_api.py        — deterministic mock REST API (seed-controlled)
  tasks.py           — task definitions and graders (easy/medium/hard)
  openenv_models.py  — Pydantic action/observation/state models
  openenv_env.py     — OpenEnv Environment class
openenv_app.py       — FastAPI server (OpenEnv adapter)
inference.py         — LLM agent runner (OpenAI client)
openenv.yaml         — OpenEnv manifest
Dockerfile           — HuggingFace Spaces container
requirements.txt
```

## Submission

- **GitHub**: https://github.com/hrithiknl17/api-explorer-env
- **HuggingFace Space**: https://huggingface.co/spaces/Hr1th1k17/api-explorer-env
