"""
validate.py — Pre-submission validation script for CustomerSupportEnv.

Checks every item on the pre-submission checklist:
  1.  openenv.yaml is valid and has required fields
  2.  Typed Pydantic models exist (Observation, Action, Reward)
  3.  reset() returns a valid Observation
  4.  step() returns a valid StepResult
  5.  state() returns a valid EnvState
  6.  All 3 tasks exist and initialise
  7.  Grader scores are in [0.0, 1.0] for all tasks
  8.  Reward breakdown components sum to <= 1.0
  9.  Baseline inference log format is correct ([START]/[STEP]/[END])
  10. inference.py is at the project root
  11. Dockerfile exists
  12. API_BASE_URL / MODEL_NAME / HF_TOKEN env var handling in inference.py
  13. Runtime stays well under 20-minute limit (sampled)

Usage:
  python validate.py
  python validate.py --verbose

Exit code 0 = all checks pass (safe to submit).
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from typing import Callable, List, Tuple

import yaml  # pip install pyyaml


# ---------------------------------------------------------------------------
# Mini test harness
# ---------------------------------------------------------------------------

PASSED: List[str] = []
FAILED: List[Tuple[str, str]] = []
VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def check(name: str, fn: Callable) -> bool:
    try:
        fn()
        PASSED.append(name)
        print(f"  \033[32m✓\033[0m  {name}")
        return True
    except Exception as exc:
        FAILED.append((name, traceback.format_exc()))
        print(f"  \033[31m✗\033[0m  {name}")
        print(f"     \033[33m{exc}\033[0m")
        if VERBOSE:
            traceback.print_exc()
        return False


def assert_eq(a, b, msg=""):
    assert a == b, f"Expected {b!r}, got {a!r}. {msg}"


def assert_in(key, collection, msg=""):
    assert key in collection, f"'{key}' not found in {list(collection)[:10]}. {msg}"


def assert_range(val, lo, hi, msg=""):
    assert lo <= val <= hi, f"{val} not in [{lo}, {hi}]. {msg}"


def assert_strict_range(val, lo, hi, msg=""):
    """Check that value is strictly between lo and hi (exclusive)."""
    assert lo < val < hi, f"{val} not strictly in ({lo}, {hi}). {msg}"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_inference_at_root():
    assert os.path.isfile("inference.py"), "inference.py not found at project root"


def check_dockerfile_exists():
    assert os.path.isfile("Dockerfile"), "Dockerfile not found"


def check_openenv_yaml():
    assert os.path.isfile("openenv.yaml"), "openenv.yaml not found"
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    required = ["name", "version", "description", "tasks", "entry_point"]
    for field in required:
        assert_in(field, data, f"openenv.yaml missing field '{field}'")
    assert len(data["tasks"]) >= 3, "openenv.yaml must define >= 3 tasks"
    for task in data["tasks"]:
        assert_in("id", task)
        assert_in("difficulty", task)


def check_pydantic_models():
    from models import Observation, Action, Reward, StepResult, EnvState, RewardBreakdown
    # Verify they are Pydantic models
    from pydantic import BaseModel
    for cls in (Observation, Action, Reward, StepResult, EnvState, RewardBreakdown):
        assert issubclass(cls, BaseModel), f"{cls.__name__} is not a Pydantic BaseModel"


def check_reset():
    from env import CustomerSupportEnv
    from models import Observation
    env = CustomerSupportEnv("order_refund")
    obs = env.reset()
    assert isinstance(obs, Observation), "reset() must return Observation"
    assert obs.step_number == 0
    assert obs.done == False
    assert obs.ticket is not None
    assert len(obs.available_tools) > 0


def check_step():
    from env import CustomerSupportEnv
    from models import Action, StepResult, ToolName
    env = CustomerSupportEnv("order_refund")
    env.reset()
    result = env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    assert isinstance(result, StepResult), "step() must return StepResult"
    assert result.observation.step_number == 1
    assert_strict_range(result.reward.value, 0.0, 1.0, "reward.value must be strictly in (0,1)")


def check_state():
    from env import CustomerSupportEnv
    from models import EnvState, Action, ToolName
    env = CustomerSupportEnv("order_refund")
    env.reset()
    env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    state = env.state()
    assert isinstance(state, EnvState), "state() must return EnvState"
    assert state.step_number == 1
    assert len(state.history) == 1


def check_all_tasks_init():
    from env import CustomerSupportEnv
    from models import TaskId
    for tid in TaskId:
        env = CustomerSupportEnv(tid)
        obs = env.reset()
        assert obs.ticket is not None, f"Task {tid} has no ticket"
        assert len(obs.available_tools) > 0, f"Task {tid} has no tools"


def check_grader_scores_in_range():
    from env import CustomerSupportEnv
    from models import Action, TaskId, ToolName

    # Minimal trajectory per task that produces a terminal reward
    trajectories = {
        "order_refund": [
            Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}),
            Action(tool=ToolName.PROCESS_REFUND, params={"order_id": "ORD-1001", "reason": "test"}),
            Action(tool=ToolName.RESPOND, message="Refund issued. Sorry for the inconvenience!"),
        ],
        "account_billing_dispute": [
            Action(tool=ToolName.LOOKUP_ACCOUNT, params={"customer_id": "CUST-003"}),
            Action(tool=ToolName.LIST_TRANSACTIONS, params={"customer_id": "CUST-003"}),
            Action(tool=ToolName.APPLY_CREDIT, params={"customer_id": "CUST-003", "amount": 89.99, "reason": "Duplicate"}),
            Action(tool=ToolName.RESPOND, message="Credit of $89.99 applied. Sorry for the trouble!"),
        ],
        "technical_escalation": [
            Action(tool=ToolName.GET_SERVICE_STATUS, params={}),
            Action(tool=ToolName.APPLY_WORKAROUND, params={"service": "authentication", "workaround_code": "AUTH-WA-001"}),
            Action(tool=ToolName.APPLY_WORKAROUND, params={"service": "streaming", "workaround_code": "STREAM-WA-001"}),
            Action(tool=ToolName.CREATE_TICKET, params={"customer_id": "CUST-001", "title": "Outage", "description": "Down", "priority": "urgent", "category": "technical"}),
            Action(tool=ToolName.NOTIFY_CUSTOMERS, params={"service": "streaming", "message": "Working on it."}),
            Action(tool=ToolName.RESPOND, message="Ticket TKT-0001 raised, ETA 2-4 hours. Please don't hesitate to reach out!"),
        ],
    }

    for task_id, actions in trajectories.items():
        env = CustomerSupportEnv(task_id)
        env.reset()
        result = None
        for action in actions:
            result = env.step(action)
        assert result is not None
        bd = result.reward.breakdown
        total = bd.total
        assert_strict_range(total, 0.0, 1.0, f"Task {task_id} score must be strictly in (0,1): {total}")
        assert_range(bd.resolution_correct, 0.0, 0.5)
        assert_range(bd.policy_compliance, 0.0, 0.2)
        assert_range(bd.efficiency, 0.0, 0.15)
        assert_range(bd.customer_satisfaction, 0.0, 0.15)


def check_reward_breakdown_sum():
    """Breakdown components must sum to <= 1.0 always."""
    from models import RewardBreakdown
    bd = RewardBreakdown(
        resolution_correct=0.50,
        policy_compliance=0.20,
        efficiency=0.15,
        customer_satisfaction=0.15,
    )
    assert_range(bd.total, 0.0, 1.0, "Max breakdown sum exceeds 1.0")


def check_log_format():
    """[START]/[STEP]/[END] lines must be exact format."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    from inference import log_start, log_step, log_end
    log_start("order_refund")
    log_step("order_refund", 1, "lookup_order", 0.02, False)
    log_step("order_refund", 2, "respond", 0.95, True)
    log_end("order_refund", 0.95, 2)

    sys.stdout = old_stdout
    lines = [l for l in captured.getvalue().strip().splitlines() if l]

    assert len(lines) == 4, f"Expected 4 log lines, got {len(lines)}"

    # [START]
    assert lines[0].startswith("[START] "), "First line must start with '[START] '"
    start = json.loads(lines[0][len("[START] "):])
    for f in ("task_id", "model", "timestamp"):
        assert_in(f, start, f"[START] missing field '{f}'")

    # [STEP]
    assert lines[1].startswith("[STEP] "), "Step line must start with '[STEP] '"
    step = json.loads(lines[1][len("[STEP] "):])
    for f in ("task_id", "step", "action", "reward", "done"):
        assert_in(f, step, f"[STEP] missing field '{f}'")
    assert_range(step["reward"], 0.0, 1.0)
    assert isinstance(step["done"], bool)

    # [END]
    assert lines[3].startswith("[END] "), "Last line must start with '[END] '"
    end = json.loads(lines[3][len("[END] "):])
    for f in ("task_id", "score", "steps", "timestamp"):
        assert_in(f, end, f"[END] missing field '{f}'")
    assert_range(end["score"], 0.0, 1.0)
    assert end["steps"] > 0


def check_env_vars_in_inference():
    """inference.py must read API_BASE_URL, MODEL_NAME, HF_TOKEN."""
    with open("inference.py") as f:
        src = f.read()
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        assert var in src, f"inference.py must reference env var '{var}'"
    assert "OpenAI" in src, "inference.py must use OpenAI client"
    # Must NOT hardcode an OpenAI key
    assert "OPENAI_API_KEY" not in src or "HF_TOKEN" in src, \
        "inference.py must prefer HF_TOKEN over OPENAI_API_KEY"


def check_runtime_budget():
    """A full reset+step cycle must complete in < 1s (proxy for 20min budget)."""
    from env import CustomerSupportEnv
    from models import Action, ToolName
    t0 = time.perf_counter()
    for _ in range(50):   # 50 full step cycles
        env = CustomerSupportEnv("order_refund")
        env.reset()
        env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
        env.step(Action(tool=ToolName.PROCESS_REFUND, params={"order_id": "ORD-1001", "reason": "test"}))
        env.step(Action(tool=ToolName.RESPOND, message="Done. Please don't hesitate to reach out!"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"50 episodes took {elapsed:.2f}s — env is too slow"


def check_app_routes():
    """FastAPI app must register /, /health, /reset, /step, /state."""
    from app import app
    routes = {r.path for r in app.routes}
    for path in ("/", "/health", "/reset", "/step", "/state", "/tasks"):
        assert path in routes, f"app.py missing route '{path}'"


def check_three_plus_tasks():
    """At least 3 tasks must be registered."""
    from tasks.definitions import TASK_REGISTRY
    assert len(TASK_REGISTRY) >= 3, f"Need >= 3 tasks, found {len(TASK_REGISTRY)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKS = [
    ("inference.py at project root",                check_inference_at_root),
    ("Dockerfile exists",                           check_dockerfile_exists),
    ("openenv.yaml valid + 3 tasks",                check_openenv_yaml),
    ("Pydantic typed models exist",                 check_pydantic_models),
    ("reset() returns Observation",                 check_reset),
    ("step() returns StepResult",                   check_step),
    ("state() returns EnvState",                    check_state),
    ("All 3 tasks initialise correctly",            check_all_tasks_init),
    ("3+ tasks registered",                         check_three_plus_tasks),
    ("Grader scores in [0.0, 1.0]",                 check_grader_scores_in_range),
    ("Reward breakdown sums to <= 1.0",             check_reward_breakdown_sum),
    ("[START]/[STEP]/[END] log format exact",        check_log_format),
    ("API_BASE_URL/MODEL_NAME/HF_TOKEN in inference", check_env_vars_in_inference),
    ("Runtime budget well under 20 min",            check_runtime_budget),
    ("FastAPI routes registered",                   check_app_routes),
]


def main():
    print()
    print("=" * 58)
    print("  CustomerSupportEnv — Pre-Submission Validator")
    print("=" * 58)
    print()

    for name, fn in CHECKS:
        check(name, fn)

    print()
    print("=" * 58)
    total  = len(CHECKS)
    passed = len(PASSED)
    failed = len(FAILED)

    if failed == 0:
        print(f"\033[32m  ALL {total} CHECKS PASSED — safe to submit!\033[0m")
    else:
        print(f"\033[31m  {failed}/{total} CHECKS FAILED\033[0m")
        print()
        for name, tb in FAILED:
            print(f"  \033[31m✗ {name}\033[0m")
            if VERBOSE:
                print("  " + tb.replace("\n", "\n  "))
    print("=" * 58)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
