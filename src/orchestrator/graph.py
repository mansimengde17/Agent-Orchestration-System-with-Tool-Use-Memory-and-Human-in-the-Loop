"""The orchestration state machine.

intake -> memory recall -> planning -> (plan approval?) ->
subtask execution in dependency order -> review -> (retry / escalate) ->
synthesis -> memory write -> delivery

Conditional edges:
- specialist failure: retry once, second failure escalates take_over
- reviewer rejection: send back to the specialist with feedback, once
- sensitive subtask: pause for approve_action before executing
- low plan confidence: pause for approve_plan before any work
"""

from __future__ import annotations

import uuid

from .agents import Reviewer, Specialist, Subtask, SubtaskResult, Supervisor
from .hitl import ApprovalQueue, escalation_needed
from .memory import LongTermMemory, WorkingMemory
from .tools import ToolRegistry
from .tracing import TaskTrace


class Orchestrator:
    def __init__(self):
        self.registry = ToolRegistry()
        self.supervisor = Supervisor()
        self.specialists = {name: Specialist(name, self.registry)
                            for name in ("research", "analysis",
                                         "writing", "code")}
        self.reviewer = Reviewer()
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self.approvals = ApprovalQueue()
        self.traces: dict[str, TaskTrace] = {}
        self.tasks: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def run(self, task: str, user: str = "default",
            auto_approve: bool = True) -> dict:
        """Execute a task end to end. With auto_approve=False the run
        pauses at every escalation and must be resumed through
        resolve_and_continue, which is how the API drives it."""
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        trace = TaskTrace(task_id, task)
        self.traces[task_id] = trace
        state = {"task_id": task_id, "task": task, "user": user,
                 "status": "running", "results": {}, "pending": None,
                 "auto_approve": auto_approve, "deliverable": ""}
        self.tasks[task_id] = state

        memories = self.long_term.recall(task, user)
        trace.add(trace.root, "supervisor", "memory_recall",
                  f"{len(memories)} memories retrieved", tokens=150)
        state["memories"] = memories

        plan = self.supervisor.plan(task, memories)
        confidence = self.supervisor.plan_confidence(task, memories)
        plan_node = trace.add(
            trace.root, "supervisor", "planning",
            f"{len(plan)} subtasks, confidence {confidence}", tokens=600)
        state["plan"] = plan
        self.working.put(task_id, "plan",
                         [s.subtask_id for s in plan])

        needed, level, reason = escalation_needed(
            "plan", confidence=confidence)
        if needed:
            plan_node.status = "escalated"
            return self._pause(state, trace, level, reason,
                               {"plan": [s.description for s in plan]})
        return self._execute(state, trace)

    # ------------------------------------------------------------------
    def _pause(self, state: dict, trace: TaskTrace, level: str,
               reason: str, context: dict) -> dict:
        escalation = self.approvals.escalate(state["task_id"], level,
                                             reason, context)
        trace.add(trace.root, "system", "escalation",
                  f"{level}: {reason}", status="escalated")
        state["status"] = "waiting_for_human"
        state["pending"] = escalation.escalation_id
        if state["auto_approve"]:
            # Demo mode: a simulated operator approves after 20 seconds
            # of review time, then execution continues.
            trace.human_seconds += 20
            self.approvals.resolve(escalation.escalation_id, "approved",
                                   "operator-sim", "looks safe")
            state["status"] = "running"
            state["pending"] = None
            return self._execute(state, trace)
        return self._snapshot(state)

    def resolve_and_continue(self, escalation_id: str, decision: str,
                             operator: str, note: str = "",
                             replacement_output: str = "") -> dict:
        escalation = self.approvals.resolve(escalation_id, decision,
                                            operator, note,
                                            replacement_output)
        state = self.tasks[escalation.task_id]
        trace = self.traces[escalation.task_id]
        trace.human_seconds += 30
        state["pending"] = None
        if decision == "rejected":
            state["status"] = "rejected_by_human"
            return self._snapshot(state)
        if decision == "modified" and replacement_output:
            subtask_id = escalation.context.get("subtask_id")
            if subtask_id:
                state["results"][subtask_id] = replacement_output
        state["status"] = "running"
        return self._execute(state, trace)

    # ------------------------------------------------------------------
    def _execute(self, state: dict, trace: TaskTrace) -> dict:
        task_id = state["task_id"]
        for subtask in state["plan"]:
            if subtask.subtask_id in state["results"]:
                continue
            missing = [d for d in subtask.depends_on
                       if d not in state["results"]]
            if missing:
                state["status"] = "failed"
                trace.add(trace.root, "supervisor", "dependency_error",
                          f"{subtask.subtask_id} needs {missing}",
                          status="failure")
                return self._snapshot(state)

            if subtask.sensitive and not state.get(
                    f"approved:{subtask.subtask_id}"):
                needed, level, reason = escalation_needed(
                    "sensitive_action", description=subtask.description)
                state[f"approved:{subtask.subtask_id}"] = True
                result = self._pause(state, trace, level, reason,
                                     {"subtask_id": subtask.subtask_id,
                                      "action": subtask.description})
                if state["status"] != "running" and \
                        state["status"] != "completed":
                    return result
                if state["status"] == "completed":
                    return result
                continue

            outcome = self._run_subtask(state, trace, subtask)
            if outcome is not None:
                return outcome

        deliverable = self._synthesize(state, trace)
        state["deliverable"] = deliverable
        state["status"] = "completed"
        self._write_memories(state, trace)
        self.working.clear(task_id)
        trace.add(trace.root, "supervisor", "delivery",
                  f"{len(deliverable)} char deliverable", tokens=200)
        return self._snapshot(state)

    def _run_subtask(self, state: dict, trace: TaskTrace,
                     subtask: Subtask):
        task_id = state["task_id"]
        specialist = self.specialists[subtask.specialist]
        context = {dep: state["results"][dep]
                   for dep in subtask.depends_on}
        result: SubtaskResult | None = None
        for attempt in (1, 2):
            try:
                result = specialist.execute(task_id, subtask, context)
                result.attempts = attempt
                break
            except RuntimeError as error:
                trace.add(trace.root, subtask.specialist, "execution",
                          str(error), status="failure", tokens=300)
                if attempt == 2:
                    needed, level, reason = escalation_needed(
                        "specialist_failed_twice",
                        specialist=subtask.specialist,
                        subtask=subtask.subtask_id)
                    paused = self._pause(
                        state, trace, level, reason,
                        {"subtask_id": subtask.subtask_id,
                         "error": str(error)})
                    if state["status"] != "running" and \
                            state["status"] != "completed":
                        return paused
                    if subtask.subtask_id not in state["results"]:
                        state["results"][subtask.subtask_id] = \
                            "[human provided output]"
                    return None

        node = trace.add(trace.root, subtask.specialist, "execution",
                         f"{subtask.subtask_id}: {subtask.description[:40]}",
                         tokens=800,
                         latency_ms=sum(c["latency_ms"]
                                        for c in result.tool_calls))
        for call in result.tool_calls:
            trace.add(node, subtask.specialist, "tool_call",
                      f"{call['tool']} ok={call['ok']}",
                      status="success" if call["ok"] else "warning")

        review = self.reviewer.review(subtask, result)
        trace.add(node, "reviewer", "review",
                  f"score {review['score']}: {review['feedback']}",
                  status="success" if review["approved"] else "warning",
                  tokens=250)
        if not review["approved"]:
            # one feedback round: the specialist re-runs with feedback
            subtask.description += f" [reviewer feedback:" \
                                   f" {review['feedback']}]"
            retry = specialist.execute(task_id, subtask, context)
            second = self.reviewer.review(subtask, retry)
            trace.add(node, "reviewer", "re-review",
                      f"score {second['score']}",
                      status="success" if second["approved"]
                      else "failure", tokens=250)
            if second["approved"]:
                result = retry
            else:
                needed, level, reason = escalation_needed(
                    "review_score", score=second["score"])
                paused = self._pause(state, trace, level, reason,
                                     {"subtask_id": subtask.subtask_id,
                                      "output": retry.output[:200]})
                if state["status"] not in ("running", "completed"):
                    return paused
        state["results"][subtask.subtask_id] = result.output
        self.working.put(task_id, subtask.subtask_id, result.output)
        return None

    def _synthesize(self, state: dict, trace: TaskTrace) -> str:
        trace.add(trace.root, "supervisor", "synthesis",
                  f"merging {len(state['results'])} subtask outputs",
                  tokens=500)
        parts = [f"# Deliverable for: {state['task']}"]
        for subtask_id, output in state["results"].items():
            parts.append(f"\n## {subtask_id}\n{output}")
        return "\n".join(parts)

    def _write_memories(self, state: dict, trace: TaskTrace) -> None:
        self.long_term.remember(
            "approach",
            f"task '{state['task'][:60]}' solved with plan"
            f" {self.working.get(state['task_id'], 'plan')}",
            state["user"])
        trace.add(trace.root, "supervisor", "memory_write",
                  "stored approach for future planning", tokens=100)

    def _snapshot(self, state: dict) -> dict:
        trace = self.traces[state["task_id"]]
        return {"task_id": state["task_id"], "status": state["status"],
                "pending_escalation": state["pending"],
                "results": dict(state["results"]),
                "deliverable": state["deliverable"],
                "trace_totals": trace.totals()}
