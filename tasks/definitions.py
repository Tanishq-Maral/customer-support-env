"""
tasks/definitions.py — Task scenario definitions.

Each TaskConfig contains:
  - The incoming customer ticket
  - Task-specific metadata (customer_id, available tools, SLA, optimal_steps)
  - The TaskId to link it to the grader
"""

from __future__ import annotations

from typing import List
from models import CustomerTicket, TaskId


class TaskConfig:
    def __init__(
        self,
        task_id: TaskId,
        ticket: CustomerTicket,
        available_tools: List[str],
        sla_seconds: int,
        optimal_steps: int,
        max_steps: int,
        difficulty: str,
        description: str,
    ):
        self.task_id         = task_id
        self.ticket          = ticket
        self.available_tools = available_tools
        self.sla_seconds     = sla_seconds
        self.optimal_steps   = optimal_steps
        self.max_steps       = max_steps
        self.difficulty      = difficulty
        self.description     = description


# ---------------------------------------------------------------------------
# Task 1 — Easy: Order Refund
# ---------------------------------------------------------------------------
TASK_ORDER_REFUND = TaskConfig(
    task_id = TaskId.ORDER_REFUND,
    difficulty = "easy",
    description = (
        "Customer received damaged headphones (Order ORD-1001). "
        "Agent must verify the order, confirm it is within the refund window, "
        "issue the refund, and communicate the outcome to the customer."
    ),
    ticket = CustomerTicket(
        ticket_id   = "TKT-IN-001",
        customer_id = "CUST-001",
        subject     = "Damaged product — need refund",
        body = (
            "Hi, I received my NoiseBlock Pro Headphones (order ORD-1001) last week "
            "and they arrived with a cracked headband. I'd like a full refund please. "
            "My customer ID is CUST-001. Thank you."
        ),
        created_at  = "2024-06-15T10:30:00Z",
        priority    = "medium",
    ),
    available_tools = [
        "lookup_order",
        "process_refund",
        "lookup_account",
        "search_kb",
        "send_email",
        "respond",
    ],
    sla_seconds    = 300,
    optimal_steps  = 4,   # lookup_order → process_refund → send_email → respond
    max_steps      = 10,
)


# ---------------------------------------------------------------------------
# Task 2 — Medium: Double-Charge Billing Dispute
# ---------------------------------------------------------------------------
TASK_ACCOUNT_BILLING_DISPUTE = TaskConfig(
    task_id = TaskId.ACCOUNT_BILLING_DISPUTE,
    difficulty = "medium",
    description = (
        "Customer Aiko Tanaka (CUST-003) was charged twice for her monthly subscription. "
        "Agent must verify the account, find duplicate transactions, apply a credit "
        "for the duplicate charge, send a confirmation email, and close the ticket."
    ),
    ticket = CustomerTicket(
        ticket_id   = "TKT-IN-002",
        customer_id = "CUST-003",
        subject     = "Charged twice this month!!",
        body = (
            "I was just reviewing my bank statement and I see two charges of $89.99 "
            "from you within the same day. My customer ID is CUST-003. "
            "This is unacceptable — please refund the duplicate charge immediately. "
            "I've been a premium customer for over 6 months!"
        ),
        created_at  = "2024-06-15T11:00:00Z",
        priority    = "high",
    ),
    available_tools = [
        "lookup_account",
        "list_transactions",
        "apply_credit",
        "send_email",
        "search_kb",
        "create_ticket",
        "respond",
    ],
    sla_seconds    = 240,
    optimal_steps  = 5,   # lookup_account → list_transactions → apply_credit → send_email → respond
    max_steps      = 15,
)


# ---------------------------------------------------------------------------
# Task 3 — Hard: Technical Outage Escalation
# ---------------------------------------------------------------------------
TASK_TECHNICAL_ESCALATION = TaskConfig(
    task_id = TaskId.TECHNICAL_ESCALATION,
    difficulty = "hard",
    description = (
        "Customer Priya Sharma (CUST-001) cannot log in and her video streaming is broken. "
        "Agent must: check service status for authentication and streaming, "
        "apply workarounds for both, file an urgent escalation ticket, "
        "notify all affected customers, and respond to Priya with the ticket number and ETA."
    ),
    ticket = CustomerTicket(
        ticket_id   = "TKT-IN-003",
        customer_id = "CUST-001",
        subject     = "Cannot login AND streaming broken — URGENT",
        body = (
            "Hi! I've been trying to log in for the past hour and keep getting error AUTH-503. "
            "Even when I somehow get in, streaming shows error STREAM-504. "
            "I have a presentation in 2 hours that depends on your platform. "
            "My customer ID is CUST-001. PLEASE HELP ASAP."
        ),
        created_at  = "2024-06-15T12:00:00Z",
        priority    = "urgent",
    ),
    available_tools = [
        "lookup_account",
        "get_service_status",
        "apply_workaround",
        "create_ticket",
        "notify_customers",
        "search_kb",
        "send_email",
        "respond",
    ],
    sla_seconds    = 180,   # Tight SLA for urgent
    optimal_steps  = 7,     # get_status → workaround×2 → create_ticket → notify → send_email → respond
    max_steps      = 20,
)


TASK_REGISTRY = {
    TaskId.ORDER_REFUND:            TASK_ORDER_REFUND,
    TaskId.ACCOUNT_BILLING_DISPUTE: TASK_ACCOUNT_BILLING_DISPUTE,
    TaskId.TECHNICAL_ESCALATION:    TASK_TECHNICAL_ESCALATION,
}
