"""
app.py — FastAPI HTTP server for CustomerSupportEnv on Hugging Face Spaces.

Endpoints:
  GET  /               → health check / 200 OK
  POST /reset          → reset(task_id) → Observation JSON
  POST /step           → step(action)   → StepResult JSON
  GET  /state          → state()        → EnvState JSON
  GET  /tasks          → list all task IDs and metadata
  GET  /health         → {"status": "ok"}

Designed to run on HF Spaces with:
  uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env import CustomerSupportEnv
from models import Action, TaskId, ToolName
from tasks.definitions import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CustomerSupportEnv",
    description=(
        "OpenEnv-compliant customer support environment. "
        "AI agents resolve multi-step tickets using tools: "
        "order lookup, refunds, billing disputes, escalations."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global env instance (single-session; HF Spaces are single-process)
# ---------------------------------------------------------------------------

_env: Optional[CustomerSupportEnv] = None


def _get_env() -> CustomerSupportEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "order_refund"


class StepRequest(BaseModel):
    tool:    str
    params:  Dict[str, Any] = {}
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check — must return 200."""
    return {
        "status":      "ok",
        "environment": "CustomerSupportEnv",
        "version":     "1.0.0",
        "tasks":       [t.value for t in TaskId],
        "endpoints":   ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/tasks")
async def list_tasks():
    """Return all task IDs with metadata."""
    result = []
    for task_id, cfg in TASK_REGISTRY.items():
        result.append({
            "task_id":       task_id.value,
            "difficulty":    cfg.difficulty,
            "description":   cfg.description,
            "max_steps":     cfg.max_steps,
            "optimal_steps": cfg.optimal_steps,
            "sla_seconds":   cfg.sla_seconds,
            "available_tools": cfg.available_tools,
            "ticket_subject": cfg.ticket.subject,
        })
    return result


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = Body(default=None)):
    """
    Initialise (or reinitialise) the environment for a given task.
    Returns the initial Observation.
    """
    global _env

    # ✅ Handle empty body
    if request is None or request.task_id is None:
        task_id = TaskId("order_refund")  # default task
    else:
        try:
            task_id = TaskId(request.task_id)
        except ValueError:
            valid = [t.value for t in TaskId]
            raise HTTPException(
                status_code=422,
                detail=f"Unknown task_id '{request.task_id}'. Valid: {valid}",
            )

    _env = CustomerSupportEnv(task_id=task_id)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute one action in the environment.
    Returns StepResult (observation, reward, done, info).
    """
    env = _get_env()

    try:
        tool = ToolName(request.tool)
    except ValueError:
        valid = [t.value for t in ToolName]
        raise HTTPException(
            status_code=422,
            detail=f"Unknown tool '{request.tool}'. Valid: {valid}",
        )

    action = Action(
        tool    = tool,
        params  = request.params,
        message = request.message,
    )

    try:
        result = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return result.model_dump()


@app.get("/state")
async def state():
    """Return a full snapshot of the current environment state."""
    env = _get_env()
    return env.state().model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
