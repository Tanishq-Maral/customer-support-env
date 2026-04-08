---
title: CustomerSupportEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - agent
  - tool-use
pinned: false
---

# CustomerSupportEnv — OpenEnv Submission

A realistic **customer support** environment where AI agents resolve
multi-step customer queries using external tools. Fully compliant with
the [OpenEnv](https://openenv.dev) specification.

## Quick API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/`        | Health check — returns 200 + env info |
| GET  | `/health`  | `{"status": "ok"}` |
| GET  | `/tasks`   | List all 3 tasks with metadata |
| POST | `/reset`   | Start episode: `{"task_id": "order_refund"}` |
| POST | `/step`    | Take action: `{"tool": "lookup_order", "params": {"order_id": "ORD-1001"}}` |
| GET  | `/state`   | Full environment state snapshot |

## Three Tasks (Easy → Hard)

| Task ID | Difficulty | Objective | Optimal Steps |
|---------|-----------|-----------|---------------|
| `order_refund` | Easy | Verify + refund damaged item (30-day policy) | 4 |
| `account_billing_dispute` | Medium | Find duplicate charge, apply credit, email customer | 5 |
| `technical_escalation` | Hard | Diagnose 2-service outage, apply workarounds, escalate, notify | 7 |

## Reward Function

| Component | Max | Description |
|-----------|-----|-------------|
| `resolution_correct` | 0.50 | Core objective completed correctly |
| `policy_compliance` | 0.20 | Company policies followed |
| `efficiency` | 0.15 | Steps used vs optimal |
| `customer_satisfaction` | 0.15 | Final message quality |

## Environment Variables

```
API_BASE_URL   = https://router.huggingface.co/v1
MODEL_NAME     = meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN       = hf_...
```

## Running Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_...

python inference.py          # all 3 tasks
python inference.py --task order_refund --verbose
```

Stdout emits structured logs:
```
[START] {"task_id": "order_refund", "model": "...", "timestamp": "..."}
[STEP]  {"task_id": "order_refund", "step": 1, "action": "lookup_order", "reward": 0.02, "done": false}
[END]   {"task_id": "order_refund", "score": 0.95, "steps": 4, "timestamp": "..."}
```

## Example Session

```python
import httpx

base = "https://<your-space>.hf.space"

# Start episode
obs = httpx.post(f"{base}/reset", json={"task_id": "order_refund"}).json()

# Call tools
result = httpx.post(f"{base}/step", json={
    "tool": "lookup_order",
    "params": {"order_id": "ORD-1001"}
}).json()

# Send final reply
result = httpx.post(f"{base}/step", json={
    "tool": "respond",
    "params": {},
    "message": "Hi! Your refund of $149.99 has been issued. Sorry for the trouble!"
}).json()

print(result["reward"]["breakdown"])
# {'resolution_correct': 0.5, 'policy_compliance': 0.2, ...}
```

## Directory Structure

```
├── app.py              FastAPI server (HF Spaces entry point)
├── inference.py        Baseline agent script  ← root level, per spec
├── env.py              CustomerSupportEnv: reset()/step()/state()
├── models.py           Pydantic typed models
├── openenv.yaml        OpenEnv metadata
├── Dockerfile          Container build
├── requirements.txt
├── data/store.py       In-memory DB (orders, accounts, transactions)
├── graders/graders.py  Deterministic task graders
├── tasks/definitions.py Task configs + customer tickets
└── tools/executor.py   11 tool implementations
```
