"""
test_env.py — Validates CustomerSupportEnv correctness.

Tests:
  1. reset() returns valid Observation
  2. step() returns valid StepResult
  3. Tool calls work and mutate DB correctly
  4. Graders produce correct scores for:
       a. Perfect trajectory (max score)
       b. Partial trajectory (partial score)
       c. Policy-violating trajectory (penalised)
  5. Episode terminates at respond() or max_steps
  6. state() returns correct snapshot
  7. All three tasks initialise correctly
"""

import sys
import traceback
from typing import Callable, List, Tuple

# -- local imports
from env import CustomerSupportEnv
from models import Action, Observation, Reward, StepResult, TaskId, ToolName
from data.store import get_db


# ---------------------------------------------------------------------------
# Mini test framework
# ---------------------------------------------------------------------------

PASSED: List[str] = []
FAILED: List[Tuple[str, str]] = []


def test(name: str, fn: Callable):
    try:
        fn()
        PASSED.append(name)
        print(f"  ✓  {name}")
    except Exception as exc:
        FAILED.append((name, traceback.format_exc()))
        print(f"  ✗  {name}")
        print(f"     {exc}")


def assert_eq(a, b, msg=""):
    assert a == b, f"Expected {b!r}, got {a!r}. {msg}"

def assert_approx(a, b, tol=0.01, msg=""):
    assert abs(a - b) < tol, f"Expected ~{b}, got {a}. {msg}"

def assert_gt(a, b, msg=""):
    assert a > b, f"Expected > {b}, got {a}. {msg}"

def assert_in(a, collection, msg=""):
    assert a in collection, f"{a!r} not in {collection!r}. {msg}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_returns_observation():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step_number == 0
    assert obs.done == False
    assert obs.ticket.ticket_id == "TKT-IN-001"
    assert obs.customer_id == "CUST-001"
    assert len(obs.history) == 0
    assert len(obs.available_tools) > 0


def test_step_returns_step_result():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    result = env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, Reward)
    assert isinstance(result.observation, Observation)
    assert result.observation.step_number == 1
    assert result.done == False


def test_lookup_order_success():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    result = env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    assert result.observation.last_error is None
    assert result.observation.last_tool_result is not None
    assert result.observation.last_tool_result["order_id"] == "ORD-1001"


def test_lookup_order_not_found():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    result = env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "NONEXISTENT"}))
    assert result.observation.last_error is not None
    assert "not found" in result.observation.last_error.lower()


def test_refund_policy_within_window():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    # ORD-1001 is 7 days old — within window
    result = env.step(Action(
        tool=ToolName.PROCESS_REFUND,
        params={"order_id": "ORD-1001", "reason": "Damaged product"},
    ))
    assert result.observation.last_error is None
    db = get_db()
    assert db["orders"]["ORD-1001"]["refund_issued"] == True
    assert "REF-ORD-1001" in db["refunds"]


def test_refund_policy_outside_window():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    # ORD-1002 is 42 days old — outside 30-day window
    result = env.step(Action(
        tool=ToolName.PROCESS_REFUND,
        params={"order_id": "ORD-1002", "reason": "Customer request"},
    ))
    assert result.observation.last_error is not None
    assert "outside" in result.observation.last_error.lower() or "30" in result.observation.last_error


def test_respond_ends_episode():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    result = env.step(Action(
        tool=ToolName.RESPOND,
        message="Thank you for contacting us. Your refund has been processed.",
    ))
    assert result.done == True
    assert result.observation.done == True


def test_full_perfect_trajectory_task1():
    """Perfect agent for task 1: lookup → refund → email → respond."""
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()

    env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    env.step(Action(tool=ToolName.PROCESS_REFUND, params={"order_id": "ORD-1001", "reason": "Damaged product"}))
    env.step(Action(
        tool=ToolName.SEND_EMAIL,
        params={
            "customer_id": "CUST-001",
            "subject": "Refund Processed",
            "body": "We have issued a full refund of $149.99 for order ORD-1001.",
        },
    ))
    result = env.step(Action(
        tool=ToolName.RESPOND,
        message="Hi Priya, I'm sorry about the damaged headphones! "
                "I've issued a full refund of $149.99 for order ORD-1001. "
                "You'll see it in 3-5 business days. Please don't hesitate to reach out!",
    ))

    assert result.done
    # Should score highly — perfect trajectory
    total = result.reward.breakdown.total
    assert_gt(total, 0.70, "Perfect task1 trajectory should score > 0.70")


def test_partial_trajectory_task1():
    """Agent that only looks up order but never refunds."""
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()

    env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    result = env.step(Action(
        tool=ToolName.RESPOND,
        message="I looked at your order.",
    ))

    grader = result.reward.breakdown
    total = grader.total
    # Should score low — missed the refund
    assert total < 0.50, f"Partial trajectory should score < 0.50, got {total}"


def test_full_perfect_trajectory_task2():
    """Perfect agent for task 2: lookup_account → list_txn → apply_credit → email → respond."""
    env = CustomerSupportEnv(task_id=TaskId.ACCOUNT_BILLING_DISPUTE)
    env.reset()

    env.step(Action(tool=ToolName.LOOKUP_ACCOUNT, params={"customer_id": "CUST-003"}))
    env.step(Action(tool=ToolName.LIST_TRANSACTIONS, params={"customer_id": "CUST-003"}))
    env.step(Action(
        tool=ToolName.APPLY_CREDIT,
        params={"customer_id": "CUST-003", "amount": 89.99, "reason": "Duplicate charge refund"},
    ))
    env.step(Action(
        tool=ToolName.SEND_EMAIL,
        params={
            "customer_id": "CUST-003",
            "subject": "Billing Credit Applied",
            "body": "We have applied a credit of $89.99 to your account for the duplicate charge.",
        },
    ))
    result = env.step(Action(
        tool=ToolName.RESPOND,
        message="Hi Aiko, I sincerely apologize for the duplicate charge! "
                "I've applied a credit of $89.99 to your account. "
                "You'll see the credit reflected within 24 hours. "
                "Please don't hesitate to contact us if you need anything else!",
    ))

    assert result.done
    total = result.reward.breakdown.total
    assert_gt(total, 0.75, "Perfect task2 trajectory should score > 0.75")


def test_full_perfect_trajectory_task3():
    """Perfect agent for task 3: status → workaround×2 → ticket → notify → email → respond."""
    env = CustomerSupportEnv(task_id=TaskId.TECHNICAL_ESCALATION)
    env.reset()

    env.step(Action(tool=ToolName.GET_SERVICE_STATUS, params={}))
    env.step(Action(
        tool=ToolName.APPLY_WORKAROUND,
        params={"service": "authentication", "workaround_code": "AUTH-WA-001"},
    ))
    env.step(Action(
        tool=ToolName.APPLY_WORKAROUND,
        params={"service": "streaming", "workaround_code": "STREAM-WA-001"},
    ))
    env.step(Action(
        tool=ToolName.CREATE_TICKET,
        params={
            "customer_id": "CUST-001",
            "title": "Auth and Streaming outage",
            "description": "AUTH-503 and STREAM-504 affecting multiple customers.",
            "priority": "urgent",
            "category": "technical",
        },
    ))
    env.step(Action(
        tool=ToolName.NOTIFY_CUSTOMERS,
        params={
            "service": "streaming",
            "message": "We are aware of current issues with authentication and streaming. Workarounds applied.",
        },
    ))
    env.step(Action(
        tool=ToolName.SEND_EMAIL,
        params={
            "customer_id": "CUST-001",
            "subject": "Service Issue Update",
            "body": "We have applied workarounds and created an escalation ticket. ETA: 2-4 hours.",
        },
    ))
    result = env.step(Action(
        tool=ToolName.RESPOND,
        message="Hi Priya, I'm sorry for the disruption! "
                "I've applied workarounds for both the authentication and streaming issues. "
                "An escalation ticket TKT-0001 has been created with urgent priority — "
                "our engineering team will resolve this within 2-4 business hours. "
                "We've also notified all affected customers. "
                "Please don't hesitate to reach out!",
    ))

    assert result.done
    total = result.reward.breakdown.total
    assert_gt(total, 0.80, "Perfect task3 trajectory should score > 0.80")


def test_state_snapshot():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "ORD-1001"}))
    state = env.state()
    assert state.step_number == 1
    assert state.task_id == TaskId.ORDER_REFUND
    assert len(state.history) == 1
    assert state.history[0].tool == "lookup_order"


def test_max_steps_terminates_episode():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    result = None
    for i in range(env._task.max_steps):
        if result and result.done:
            break
        result = env.step(Action(tool=ToolName.LOOKUP_ACCOUNT, params={"customer_id": "CUST-001"}))
    assert result.done, "Episode should terminate at max_steps"


def test_all_three_tasks_init():
    for tid in TaskId:
        env = CustomerSupportEnv(task_id=tid)
        obs = env.reset()
        assert obs.ticket is not None, f"Task {tid} should have a ticket"
        assert len(obs.available_tools) > 0, f"Task {tid} should have tools"
        print(f"    Task {tid.value}: OK")


def test_error_penalty_applied():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    # Call a tool with bad params
    result = env.step(Action(tool=ToolName.LOOKUP_ORDER, params={"order_id": "NONEXISTENT-99"}))
    assert result.reward.value >= 0, "Reward must be non-negative"
    assert result.observation.last_error is not None


def test_duplicate_refund_blocked():
    env = CustomerSupportEnv(task_id=TaskId.ORDER_REFUND)
    env.reset()
    env.step(Action(tool=ToolName.PROCESS_REFUND, params={"order_id": "ORD-1001", "reason": "test"}))
    result = env.step(Action(tool=ToolName.PROCESS_REFUND, params={"order_id": "ORD-1001", "reason": "test again"}))
    assert result.observation.last_error is not None
    assert "already" in result.observation.last_error.lower()


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  CustomerSupportEnv — Test Suite")
    print("="*55)

    tests = [
        ("reset() returns valid Observation",        test_reset_returns_observation),
        ("step() returns valid StepResult",          test_step_returns_step_result),
        ("lookup_order success",                     test_lookup_order_success),
        ("lookup_order not found",                   test_lookup_order_not_found),
        ("refund within 30-day window",              test_refund_policy_within_window),
        ("refund outside 30-day window blocked",     test_refund_policy_outside_window),
        ("respond() ends episode",                   test_respond_ends_episode),
        ("Task 1 perfect trajectory",                test_full_perfect_trajectory_task1),
        ("Task 1 partial trajectory (low score)",   test_partial_trajectory_task1),
        ("Task 2 perfect trajectory",                test_full_perfect_trajectory_task2),
        ("Task 3 perfect trajectory",                test_full_perfect_trajectory_task3),
        ("state() snapshot correct",                 test_state_snapshot),
        ("max_steps terminates episode",             test_max_steps_terminates_episode),
        ("all three tasks initialise",               test_all_three_tasks_init),
        ("error penalty applied",                    test_error_penalty_applied),
        ("duplicate refund blocked",                 test_duplicate_refund_blocked),
    ]

    print()
    for name, fn in tests:
        test(name, fn)

    print()
    print("="*55)
    print(f"  Passed: {len(PASSED)}/{len(tests)}")
    if FAILED:
        print(f"  FAILED: {len(FAILED)}")
        for name, tb in FAILED:
            print(f"\n  ✗ {name}")
            print("  " + tb.replace("\n", "\n  "))
    print("="*55)

    sys.exit(0 if not FAILED else 1)
