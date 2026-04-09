"""
models.py — Typed Pydantic models for the Customer Support OpenEnv environment.
Defines Observation, Action, Reward, and supporting data structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolName(str, Enum):
    LOOKUP_ORDER       = "lookup_order"
    PROCESS_REFUND     = "process_refund"
    LOOKUP_ACCOUNT     = "lookup_account"
    LIST_TRANSACTIONS  = "list_transactions"
    APPLY_CREDIT       = "apply_credit"
    SEND_EMAIL         = "send_email"
    SEARCH_KB          = "search_kb"
    CREATE_TICKET      = "create_ticket"
    GET_SERVICE_STATUS = "get_service_status"
    APPLY_WORKAROUND   = "apply_workaround"
    NOTIFY_CUSTOMERS   = "notify_customers"
    RESPOND            = "respond"   # Final response to the customer


class TaskId(str, Enum):
    ORDER_REFUND            = "order_refund"
    ACCOUNT_BILLING_DISPUTE = "account_billing_dispute"
    TECHNICAL_ESCALATION    = "technical_escalation"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """Records a single tool invocation in the agent's history."""
    step:   int            = Field(..., description="Step number when called")
    tool:   str            = Field(..., description="Tool name")
    params: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any]  = Field(None, description="Tool response payload")
    error:  Optional[str]  = Field(None, description="Error message if call failed")


class CustomerTicket(BaseModel):
    ticket_id:   str
    customer_id: str
    subject:     str
    body:        str
    created_at:  str   # ISO-8601 string
    priority:    str   # low | medium | high | urgent


# ---------------------------------------------------------------------------
# Core OpenEnv models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket:          CustomerTicket
    customer_id:     str
    history:         List[ToolCall]          = Field(default_factory=list)
    available_tools: List[str]               = Field(default_factory=list)
    step_number:     int                     = 0
    sla_remaining:   int                     = Field(300, description="Seconds before SLA breach")
    last_tool_result: Optional[Any]          = None
    last_error:      Optional[str]           = None
    task_id:         TaskId                  = TaskId.ORDER_REFUND
    done:            bool                    = False


class Action(BaseModel):
    """What the agent wants to do."""
    tool:    ToolName                        = Field(..., description="Tool to invoke")
    params:  Dict[str, Any]                 = Field(default_factory=dict)
    message: Optional[str]                  = Field(None, description="Customer-facing message when tool=respond")


class RewardBreakdown(BaseModel):
    resolution_correct:    float = Field(0.1, ge=0.0, le=0.5)
    policy_compliance:     float = Field(0.1, ge=0.0, le=0.2)
    efficiency:            float = Field(0.1, ge=0.0, le=0.15)
    customer_satisfaction: float = Field(0.1, ge=0.0, le=0.15)
    # total is a real field (not a @property) so it appears in model_dump() / JSON
    # Must be strictly between 0 and 1 (exclusive) per OpenEnv spec
    total:                 float = Field(0.04, gt=0.0, lt=1.0)

    @model_validator(mode='after')
    def _clamp_total(self) -> 'RewardBreakdown':
        """Ensure total is always strictly between 0 and 1."""
        if self.total <= 0.0 or self.total >= 1.0:
            object.__setattr__(self, 'total',
                max(0.1, min(0.9, self.total)))
        return self

    def compute_total(self) -> "RewardBreakdown":
        """Recompute and set total from components. Call after setting components."""
        raw = (self.resolution_correct + self.policy_compliance +
               self.efficiency + self.customer_satisfaction)
        clamped = max(0.1, min(0.9, raw))
        object.__setattr__(self, "total", clamped)
        return self


class Reward(BaseModel):
    """Reward returned by step()."""
    value:     float           = Field(..., gt=0.0, lt=1.0)
    breakdown: RewardBreakdown
    done:      bool
    info:      Dict[str, Any]  = Field(default_factory=dict)


class StepResult(BaseModel):
    """Full result returned by env.step()."""
    observation: Observation
    reward:      Reward
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


class EnvState(BaseModel):
    """Internal state snapshot returned by env.state()."""
    task_id:        TaskId
    step_number:    int
    done:           bool
    cumulative_reward: float
    history:        List[ToolCall]
    db_snapshot:    Dict[str, Any]   = Field(default_factory=dict)