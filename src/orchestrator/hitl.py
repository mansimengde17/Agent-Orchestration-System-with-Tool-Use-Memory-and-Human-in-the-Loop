"""Human in the loop: escalation triggers and the approval queue.

Approval levels, in increasing depth of human involvement:
- notify: proceed, but tell the human
- approve_action: pause until the human confirms the next step
- approve_plan: the full plan needs sign off before any work starts
- take_over: the human provides the output, agents stand down
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

PLAN_CONFIDENCE_FLOOR = 0.60
REVIEW_SCORE_FLOOR = 0.55


@dataclass
class Escalation:
    escalation_id: str
    task_id: str
    level: str
    reason: str
    context: dict
    status: str = "waiting"    # waiting | approved | rejected | modified
    resolution: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class ApprovalQueue:
    def __init__(self):
        self.items: dict[str, Escalation] = {}
        self._counter = 0

    def escalate(self, task_id: str, level: str, reason: str,
                 context: dict) -> Escalation:
        self._counter += 1
        escalation = Escalation(f"esc-{self._counter:03d}", task_id,
                                level, reason, context)
        self.items[escalation.escalation_id] = escalation
        return escalation

    def waiting(self) -> list[Escalation]:
        return [e for e in self.items.values() if e.status == "waiting"]

    def resolve(self, escalation_id: str, decision: str,
                operator: str, note: str = "",
                replacement_output: str = "") -> Escalation:
        escalation = self.items[escalation_id]
        escalation.status = decision
        escalation.resolution = {"operator": operator, "note": note,
                                 "replacement_output": replacement_output,
                                 "at": time.time()}
        return escalation


def escalation_needed(kind: str, **facts) -> tuple[bool, str, str]:
    """Central trigger table. Returns (needed, level, reason)."""
    if kind == "plan" and facts["confidence"] < PLAN_CONFIDENCE_FLOOR:
        return True, "approve_plan", \
            f"plan confidence {facts['confidence']} below" \
            f" {PLAN_CONFIDENCE_FLOOR}"
    if kind == "specialist_failed_twice":
        return True, "take_over", \
            f"specialist {facts['specialist']} failed twice on" \
            f" {facts['subtask']}"
    if kind == "sensitive_action":
        return True, "approve_action", \
            f"sensitive operation: {facts['description']}"
    if kind == "review_score" and facts["score"] < REVIEW_SCORE_FLOOR:
        return True, "approve_action", \
            f"reviewer score {facts['score']} below floor"
    if kind == "user_requested":
        return True, "approve_plan", "user explicitly requested review"
    return False, "", ""
