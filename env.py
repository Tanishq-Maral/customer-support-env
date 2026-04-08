"""
env.py — CustomerSupportEnv: Full OpenEnv-spec environment.

Implements:
  reset()  → Observation
  step()   → StepResult (observation, reward, done, info)
  state()  → EnvState
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from data.store import reset_db, get_db
from graders.graders import grade
from models import (
    Action, EnvState, Observation, Reward, RewardBreakdown,
    StepResult, TaskId, ToolCall,
)
from tasks.definitions import TASK_REGISTRY, TaskConfig
from tools.executor import execute_tool, ToolError


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomerSupportEnv:
    """
    Simulates a customer support desk where an AI agent must:
      1. Read an incoming customer ticket
      2. Use available tools to diagnose and resolve the issue
      3. Send a final response to the customer

    Conforms to the OpenEnv step()/reset()/state() interface.
    """

    # Penalty applied per step beyond the optimal step count
    STEP_PENALTY = 0.01
    # Penalty for calling an unknown / erroring tool
    ERROR_PENALTY = 0.02
    # Bonus for completing the episode in optimal steps
    EFFICIENCY_BONUS = 0.05

    def __init__(self, task_id: Union[TaskId, str] = TaskId.ORDER_REFUND):
        if isinstance(task_id, str):
            task_id = TaskId(task_id)
        self.task_id: TaskId = task_id
        self._task: TaskConfig = TASK_REGISTRY[task_id]

        # Episode state
        self._step_number:       int              = 0
        self._history:           List[ToolCall]   = []
        self._done:              bool             = False
        self._cumulative_reward: float            = 0.0
        self._episode_start:     float            = time.time()
        self._last_observation:  Optional[Observation] = None

        # Initialise DB
        reset_db()

    # -----------------------------------------------------------------------
    # OpenEnv interface
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a new episode. Returns the initial Observation."""
        reset_db()
        self._step_number       = 0
        self._history           = []
        self._done              = False
        self._cumulative_reward = 0.0
        self._episode_start     = time.time()

        obs = self._build_observation()
        self._last_observation = obs
        return obs

    def step(self, action: Union[Action, Dict[str, Any]]) -> StepResult:
        """
        Execute one action.

        action may be an Action instance or a plain dict with keys:
          tool, params, message (optional)

        Returns StepResult(observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Parse action
        if isinstance(action, dict):
            action = Action(**action)

        self._step_number += 1

        # ---- Execute tool or respond ----------------------------------------
        tool_result: Any         = None
        tool_error:  Optional[str] = None

        if action.tool.value == "respond":
            # Final customer reply — end the episode
            tool_result = {"message": action.message or ""}
            self._done  = True
        else:
            try:
                tool_result = execute_tool(action.tool.value, action.params)
            except ToolError as exc:
                tool_error  = str(exc)
                tool_result = None
            except TypeError as exc:
                tool_error  = f"Parameter error: {exc}"
                tool_result = None

        # Record in history
        call = ToolCall(
            step   = self._step_number,
            tool   = action.tool.value,
            params = action.params,
            result = tool_result,
            error  = tool_error,
        )
        self._history.append(call)

        # ---- Check max steps ------------------------------------------------
        if self._step_number >= self._task.max_steps and not self._done:
            self._done = True   # Force episode end

        # ---- Compute reward -------------------------------------------------
        reward = self._compute_reward(call)
        self._cumulative_reward += reward.value

        # ---- Build next observation -----------------------------------------
        obs = self._build_observation(
            last_tool_result = tool_result,
            last_error       = tool_error,
        )
        self._last_observation = obs

        step_info = {
            "step":              self._step_number,
            "cumulative_reward": self._cumulative_reward,
            "tool_error":        tool_error,
            "task_id":           self.task_id.value,
        }
        # Merge reward-level info (grader_breakdown, efficiency_bonus, etc.)
        step_info.update(reward.info)

        return StepResult(
            observation = obs,
            reward      = reward,
            done        = self._done,
            info        = step_info,
        )

    def state(self) -> EnvState:
        """Return a full snapshot of the current environment state."""
        db = get_db()
        return EnvState(
            task_id            = self.task_id,
            step_number        = self._step_number,
            done               = self._done,
            cumulative_reward  = self._cumulative_reward,
            history            = list(self._history),
            db_snapshot        = {
                "refunds":        dict(db.get("refunds", {})),
                "credits":        dict(db.get("credits", {})),
                "emails_sent":    list(db.get("emails_sent", [])),
                "support_tickets": dict(db.get("support_tickets", {})),
                "notifications_sent": list(db.get("notifications_sent", [])),
                "workarounds_applied": dict(db.get("workarounds_applied", {})),
            },
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        last_tool_result: Any = None,
        last_error: Optional[str] = None,
    ) -> Observation:
        elapsed = int(time.time() - self._episode_start)
        sla_remaining = max(0, self._task.sla_seconds - elapsed)
        return Observation(
            ticket           = self._task.ticket,
            customer_id      = self._task.ticket.customer_id,
            history          = list(self._history),
            available_tools  = self._task.available_tools,
            step_number      = self._step_number,
            sla_remaining    = sla_remaining,
            last_tool_result = last_tool_result,
            last_error       = last_error,
            task_id          = self.task_id,
            done             = self._done,
        )

    def _compute_reward(self, last_call: ToolCall) -> Reward:
        """
        Compute the per-step reward.

        Intermediate steps get a small shaped signal.
        Terminal step (respond or max_steps) gets the full grader evaluation.
        """
        step_reward = 0.0
        info: Dict[str, Any] = {}

        # Penalty for tool errors
        if last_call.error:
            step_reward -= self.ERROR_PENALTY
            info["error_penalty"] = self.ERROR_PENALTY

        # Small positive signal for making a useful tool call (not respond)
        USEFUL_TOOLS = {
            "lookup_order", "lookup_account", "list_transactions",
            "get_service_status", "search_kb",
        }
        if last_call.tool in USEFUL_TOOLS and not last_call.error:
            step_reward += 0.02
            info["useful_tool_bonus"] = 0.02

        # Terminal evaluation via grader
        breakdown = RewardBreakdown()
        if self._done:
            breakdown = grade(
                task_id      = self.task_id.value,
                history      = self._history,
                steps_used   = self._step_number,
                optimal_steps = self._task.optimal_steps,
                max_steps    = self._task.max_steps,
            )
            step_reward += breakdown.total
            info["grader_breakdown"] = breakdown.model_dump()
            info["terminal"] = True

            # Efficiency bonus for finishing within optimal steps
            if self._step_number <= self._task.optimal_steps:
                step_reward += self.EFFICIENCY_BONUS
                info["efficiency_bonus"] = self.EFFICIENCY_BONUS

        # Clip to [0, 1]
        step_reward = max(0.0, min(1.0, step_reward))

        return Reward(
            value     = round(step_reward, 4),
            breakdown = breakdown,
            done      = self._done,
            info      = info,
        )

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def render(self) -> str:
        """Human-readable summary of current state."""
        lines = [
            f"=== CustomerSupportEnv | Task: {self.task_id.value} ===",
            f"Step: {self._step_number}/{self._task.max_steps}  |  "
            f"Done: {self._done}  |  CumReward: {self._cumulative_reward:.4f}",
            "",
            f"Ticket [{self._task.ticket.ticket_id}] — {self._task.ticket.subject}",
            f"Customer: {self._task.ticket.customer_id}",
            "",
            "History:",
        ]
        for call in self._history:
            status = f"✓ {call.result}" if call.error is None else f"✗ {call.error}"
            lines.append(f"  Step {call.step}: {call.tool}({call.params}) → {status}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory helpers used by openenv.yaml entry_point
# ---------------------------------------------------------------------------

def make_env(task_id: str = "order_refund") -> CustomerSupportEnv:
    return CustomerSupportEnv(task_id=TaskId(task_id))
