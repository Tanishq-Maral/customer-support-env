"""
graders/graders.py — Deterministic graders for all three tasks.

Each grader inspects the live DB state and action history to compute
a score from 0.0 to 1.0 with a detailed breakdown.

Score components:
  resolution_correct    (0–0.50): Core task objective met?
  policy_compliance     (0–0.20): Company policies followed?
  efficiency            (0–0.15): Steps used vs optimal?
  customer_satisfaction (0–0.15): Final message quality?
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from data.store import get_db
from models import RewardBreakdown, ToolCall


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _tool_was_called(history: List[ToolCall], tool_name: str) -> bool:
    return any(h.tool == tool_name and h.error is None for h in history)


def _tool_result(history: List[ToolCall], tool_name: str) -> Any:
    """Return result of the first successful call to tool_name, or None."""
    for h in history:
        if h.tool == tool_name and h.error is None and h.result is not None:
            return h.result
    return None


def _last_message(history: List[ToolCall]) -> str:
    """Return the customer-facing message from the final respond action."""
    for h in reversed(history):
        if h.tool == "respond" and h.result:
            return str(h.result.get("message", ""))
    return ""


def _message_contains(history: List[ToolCall], *keywords: str) -> bool:
    msg = _last_message(history).lower()
    return any(kw.lower() in msg for kw in keywords)


def _score_efficiency(steps_used: int, optimal: int, max_steps: int) -> float:
    """
    1.0 if steps_used <= optimal.
    Linearly decays to 0.0 at max_steps.
    """
    if steps_used <= optimal:
        return 1.0
    if steps_used >= max_steps:
        return 0.0
    return 1.0 - (steps_used - optimal) / (max_steps - optimal)


def _score_message_quality(message: str) -> float:
    """
    Heuristic message quality score (0–1).
    Checks for empathy, action taken, and resolution confirmation.
    """
    if not message or len(message) < 20:
        return 0.0
    score = 0.0
    empathy_words   = ["sorry", "apologize", "understand", "frustrating", "thank"]
    action_words    = ["refund", "credit", "ticket", "resolved", "applied", "issued", "notified"]
    confirm_words   = ["please", "feel free", "let us know", "don't hesitate", "contact"]

    if any(w in message.lower() for w in empathy_words):
        score += 0.35
    if any(w in message.lower() for w in action_words):
        score += 0.40
    if any(w in message.lower() for w in confirm_words):
        score += 0.25
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Grader 1 — Order Refund (Easy)
# ---------------------------------------------------------------------------

def grade_order_refund(
    history: List[ToolCall],
    steps_used: int,
    optimal_steps: int,
    max_steps: int,
) -> RewardBreakdown:
    db = get_db()
    breakdown = RewardBreakdown()

    # --- resolution_correct (0–0.50) -----------------------------------------
    # Full marks if refund was successfully issued for ORD-1001
    refund_issued = db["orders"].get("ORD-1001", {}).get("refund_issued", False)
    refund_exists = "REF-ORD-1001" in db["refunds"]

    if refund_issued and refund_exists:
        breakdown.resolution_correct = 0.50
    elif _tool_was_called(history, "process_refund"):
        # Called process_refund but failed (wrong order? already refunded?)
        breakdown.resolution_correct = 0.15
    elif _tool_was_called(history, "lookup_order"):
        # At least looked up the order
        breakdown.resolution_correct = 0.10
    else:
        breakdown.resolution_correct = 0.0

    # --- policy_compliance (0–0.20) ------------------------------------------
    # Policy: must look up order first, must check it's within 30 days
    pc = 0.0
    if _tool_was_called(history, "lookup_order"):
        pc += 0.10   # Good: looked up order before acting
    # Verify agent did NOT try to refund an out-of-window order (ORD-1002 is out of window)
    bad_refund_calls = [
        h for h in history
        if h.tool == "process_refund"
        and isinstance(h.params, dict)
        and h.params.get("order_id") != "ORD-1001"
    ]
    if not bad_refund_calls:
        pc += 0.10   # Good: only refunded the correct order
    breakdown.policy_compliance = pc

    # --- efficiency (0–0.15) -------------------------------------------------
    eff = _score_efficiency(steps_used, optimal_steps, max_steps)
    breakdown.efficiency = round(eff * 0.15, 4)

    # --- customer_satisfaction (0–0.15) --------------------------------------
    message = _last_message(history)
    mq = _score_message_quality(message)
    # Bonus if agent mentioned refund amount
    if re.search(r"\$\d+|\brefund\b", message.lower()):
        mq = min(mq + 0.1, 1.0)
    breakdown.customer_satisfaction = round(mq * 0.15, 4)

    return breakdown


# ---------------------------------------------------------------------------
# Grader 2 — Account Billing Dispute (Medium)
# ---------------------------------------------------------------------------

def grade_account_billing_dispute(
    history: List[ToolCall],
    steps_used: int,
    optimal_steps: int,
    max_steps: int,
) -> RewardBreakdown:
    db = get_db()
    breakdown = RewardBreakdown()

    # --- resolution_correct (0–0.50) -----------------------------------------
    # Full marks: credit applied for CUST-003 + email sent
    cust_credits = [
        c for c in db["credits"].values()
        if c["customer_id"] == "CUST-003"
    ]
    email_sent = any(
        e["customer_id"] == "CUST-003"
        for e in db["emails_sent"]
    )
    transactions_checked = _tool_was_called(history, "list_transactions")
    account_checked      = _tool_was_called(history, "lookup_account")

    if cust_credits and email_sent:
        # Check credit amount is correct (≈89.99)
        correct_amount = any(abs(c["amount"] - 89.99) < 0.01 for c in cust_credits)
        breakdown.resolution_correct = 0.50 if correct_amount else 0.35
    elif cust_credits:
        breakdown.resolution_correct = 0.30
    elif transactions_checked:
        breakdown.resolution_correct = 0.15
    elif account_checked:
        breakdown.resolution_correct = 0.08
    else:
        breakdown.resolution_correct = 0.0

    # --- policy_compliance (0–0.20) ------------------------------------------
    pc = 0.0
    if account_checked:
        pc += 0.07   # Verified account before financial action
    if transactions_checked:
        pc += 0.07   # Identified duplicate via transaction list
    # Agent should NOT apply a credit without looking at transactions first
    credit_before_txn = False
    seen_txn = False
    for h in history:
        if h.tool == "list_transactions":
            seen_txn = True
        if h.tool == "apply_credit" and not seen_txn:
            credit_before_txn = True
    if not credit_before_txn:
        pc += 0.06
    breakdown.policy_compliance = pc

    # --- efficiency (0–0.15) -------------------------------------------------
    eff = _score_efficiency(steps_used, optimal_steps, max_steps)
    breakdown.efficiency = round(eff * 0.15, 4)

    # --- customer_satisfaction (0–0.15) --------------------------------------
    message = _last_message(history)
    mq = _score_message_quality(message)
    if email_sent:
        mq = min(mq + 0.15, 1.0)   # Proactively emailed = good service
    breakdown.customer_satisfaction = round(mq * 0.15, 4)

    return breakdown


# ---------------------------------------------------------------------------
# Grader 3 — Technical Escalation (Hard)
# ---------------------------------------------------------------------------

def grade_technical_escalation(
    history: List[ToolCall],
    steps_used: int,
    optimal_steps: int,
    max_steps: int,
) -> RewardBreakdown:
    db = get_db()
    breakdown = RewardBreakdown()

    # --- resolution_correct (0–0.50) -----------------------------------------
    # Required: workaround for auth + streaming, escalation ticket, notify customers
    status_checked     = _tool_was_called(history, "get_service_status")
    auth_workaround    = any(
        h.tool == "apply_workaround"
        and isinstance(h.params, dict)
        and h.params.get("service") == "authentication"
        and h.error is None
        for h in history
    )
    stream_workaround  = any(
        h.tool == "apply_workaround"
        and isinstance(h.params, dict)
        and h.params.get("service") == "streaming"
        and h.error is None
        for h in history
    )
    ticket_created     = len(db["support_tickets"]) > 0
    customers_notified = len(db["notifications_sent"]) > 0
    responded          = any(h.tool == "respond" for h in history)

    # Partial credit breakdown:
    score = 0.0
    if status_checked:      score += 0.05
    if auth_workaround:     score += 0.10
    if stream_workaround:   score += 0.10
    if ticket_created:      score += 0.10
    if customers_notified:  score += 0.10
    if responded:           score += 0.05
    breakdown.resolution_correct = min(score, 0.50)

    # --- policy_compliance (0–0.20) ------------------------------------------
    pc = 0.0
    # Must check status before applying workaround
    status_before_workaround = True
    seen_status = False
    for h in history:
        if h.tool == "get_service_status":
            seen_status = True
        if h.tool == "apply_workaround" and not seen_status:
            status_before_workaround = False
    if status_before_workaround and status_checked:
        pc += 0.08

    # Ticket must have priority=urgent or high for this type of issue
    urgent_ticket = any(
        t.get("priority") in ("urgent", "high")
        for t in db["support_tickets"].values()
    )
    if urgent_ticket:
        pc += 0.07

    # Agent should mention ticket number in final response
    ticket_ids = list(db["support_tickets"].keys())
    message = _last_message(history)
    if any(tid in message for tid in ticket_ids):
        pc += 0.05

    breakdown.policy_compliance = min(pc, 0.20)

    # --- efficiency (0–0.15) -------------------------------------------------
    eff = _score_efficiency(steps_used, optimal_steps, max_steps)
    breakdown.efficiency = round(eff * 0.15, 4)

    # --- customer_satisfaction (0–0.15) --------------------------------------
    message = _last_message(history)
    mq = _score_message_quality(message)
    # Bonus: message mentions ETA or ticket number
    if re.search(r"TKT-\d{4}|ticket|hours|ETA", message, re.IGNORECASE):
        mq = min(mq + 0.2, 1.0)
    breakdown.customer_satisfaction = round(mq * 0.15, 4)

    return breakdown


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "order_refund":            grade_order_refund,
    "account_billing_dispute": grade_account_billing_dispute,
    "technical_escalation":    grade_technical_escalation,
}


def grade(
    task_id: str,
    history: List[ToolCall],
    steps_used: int,
    optimal_steps: int,
    max_steps: int,
) -> RewardBreakdown:
    fn = GRADERS.get(task_id)
    if fn is None:
        raise ValueError(f"No grader for task_id='{task_id}'")
    return fn(history, steps_used, optimal_steps, max_steps)
