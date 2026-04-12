---
title: API Explorer
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# API Explorer — Undocumented API Navigator

![OpenEnv](https://img.shields.io/badge/OpenEnv-API%20Ready-blue)
![Domain](https://img.shields.io/badge/Domain-LLM%20Reasoning-green)
![Scenarios](https://img.shields.io/badge/Scenarios-easy%20%7C%20medium%20%7C%20hard-orange)
![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-brightgreen)

An OpenEnv environment where an LLM agent must explore an **undocumented REST API**, infer its structure from raw responses, and complete real-world tasks using the minimum number of API calls.

> Tests reasoning, not optimization. An LLM has a structural advantage here — it can read JSON and decide what to call next. A heuristic cannot.

## The Challenge

The agent is dropped into a fictional payment platform API with no docs, no schema, and no route list. It must discover endpoints by calling them, understand data models from responses, and complete tasks like:

- How many merchants are registered?
- What is the total transaction volume?
- Which merchant has the highest balance?
- Find and resolve the oldest open dispute.

## Environment Design

### Discoverable API Routes

```
GET  /merchants                      — paginated merchant list
GET  /merchants/{id}                 — single merchant record
GET  /merchants/{id}/transactions    — merchant's transactions
GET  /transactions                   — filterable by status, paginated
GET  /transactions/{id}              — single transaction
GET  /disputes                       — filterable by status
GET  /disputes/{id}                  — single dispute
POST /disputes/{id}/resolve          — resolve a dispute (state mutation)
GET  /summary                        — aggregate: total volume, failed count
```

The agent does not know these routes upfront. It discovers them by calling paths and reading responses.

### Action Space

```python
class APIAction(Action):
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    path: str          # e.g. "/merchants" or "/disputes/dsp_003/resolve"
    body: dict         # query params or request body
    headers: dict
    submit_answer: str | None  # set when confident of the answer
```

### Observation Space

```python
class APIObservation(Observation):
    status_code: int
    response_body: Any         # raw API response
    current_task: TaskStatus   # active task details
    api_calls_used: int
    api_calls_limit: int
    task_scores: dict[str, float]  # all task scores in [0.0, 1.0]
    message: str               # natural language feedback
```

### Reward & Scoring

- Correct answer with efficient calls → `1.0`
- Correct answer with extra calls → `0.5`–`1.0` (efficiency bonus)
- Wrong answer → `0.0` (keep exploring)
- 404 response → `−0.02` (mild exploration penalty)

### Scenarios

| Scenario | Call Limit | Tasks |
|---|---|---|
| `easy` | 15 | Count merchants, find merchant status, total volume |
| `medium` | 25 | Count merchants, count failed transactions, merchant volume |
| `hard` | 40 | Count failed transactions, richest merchant, resolve oldest dispute |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the OpenEnv server

```bash
python -m uvicorn openenv_app:app --host 0.0.0.0 --port 7860
```

Endpoints available:
- `GET /` — health check
- `POST /reset` — start a new episode
- `POST /step` — take an action
- `GET /state` — current environment state
- `GET /health` — OpenEnv health check

### 3. Run LLM inference

```bash
export API_BASE_URL="https://router.huggingface.co/together/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-token"

python inference.py --scenario medium --max-steps 30 --seed 42
```

### 4. Example reset + step

```bash
# Reset to easy scenario
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario": "easy", "seed": 42}'

# Call an API endpoint
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"method": "GET", "path": "/merchants", "body": {}, "headers": {}, "submit_answer": null}'

# Submit an answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"method": "GET", "path": "/merchants", "body": {}, "headers": {}, "submit_answer": "20"}'
```

## Repo Layout

```
api_explorer_env/
  __init__.py
  mock_api.py        — deterministic mock REST API (seed-controlled)
  tasks.py           — task definitions and graders
  openenv_models.py  — Pydantic action/observation/state models
  openenv_env.py     — OpenEnv Environment class
openenv_app.py       — FastAPI server entrypoint
inference.py         — LLM agent runner (OpenAI client)
openenv.yaml         — OpenEnv manifest
Dockerfile
requirements.txt
SOLUTION.md          — full solution summary
```

## Baseline Scores

Reproducible baseline using `meta-llama/Llama-3.3-70B-Instruct` at `temperature=0`, `seed=42`:

| Scenario | Steps Used | API Calls | Tasks Completed | Scores |
|---|---|---|---|---|
| `easy` | 9 | 6 | 3/3 | count_merchants: 1.0, find_merchant_status: 1.0, total_transaction_volume: 1.0 |
| `medium` | 14 | 10 | 3/3 | count_merchants: 1.0, count_failed_transactions: 0.85, merchant_total_volume: 0.80 |
| `hard` | 22 | 18 | 2/3 | count_failed_transactions: 0.90, find_richest_merchant: 0.75, resolve_oldest_open_dispute: 0.50 |

To reproduce:

```bash
export API_BASE_URL="https://router.huggingface.co/together/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-token"

python inference.py --scenario easy --seed 42 --max-steps 30
python inference.py --scenario medium --seed 42 --max-steps 30
python inference.py --scenario hard --seed 42 --max-steps 40
```

## Submission Links

- **GitHub**: https://github.com/hrithiknl17/api-explorer-env
- **HuggingFace Space**: https://huggingface.co/spaces/Hr1th1k17/api-explorer-env
