"""The agent hierarchy: supervisor, specialists, and reviewer.

Each agent is a node with a defined input/output contract. The reasoning
inside each specialist is a deterministic simulation of what the LLM
call produces, so the orchestration logic, tool permissions, review
loop, and escalation paths are all exercised for real.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .tools import ToolRegistry


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


@dataclass
class Subtask:
    subtask_id: str
    description: str
    specialist: str
    depends_on: list[str] = field(default_factory=list)
    expected_output: str = "text"
    complexity: str = "moderate"
    sensitive: bool = False


@dataclass
class SubtaskResult:
    subtask_id: str
    specialist: str
    output: str
    tool_calls: list[dict]
    confidence: float
    attempts: int = 1


class Supervisor:
    """Decomposes tasks into ordered subtasks with dependencies."""

    def plan(self, task: str, memories: list[dict]) -> list[Subtask]:
        lowered = task.lower()
        plan: list[Subtask] = []
        hints = " ".join(m["text"] for m in memories)
        if "research" in lowered or "report" in lowered or \
                "market" in lowered:
            plan.append(Subtask("s1", f"gather sources on: {task}",
                                "research", expected_output="snippets"))
            plan.append(Subtask("s2", "analyze gathered data and compute"
                                      " the headline figures", "analysis",
                                depends_on=["s1"],
                                expected_output="table"))
            plan.append(Subtask("s3", "write the summary with citations",
                                "writing", depends_on=["s2"],
                                complexity="hard"))
        elif "sql" in lowered or "revenue" in lowered:
            plan.append(Subtask("s1", "query revenue by region",
                                "analysis", expected_output="table"))
            plan.append(Subtask("s2", "write commentary on the numbers",
                                "writing", depends_on=["s1"]))
        else:
            plan.append(Subtask("s1", f"execute request: {task}",
                                "writing"))
        if "email" in lowered or "send" in lowered or \
                "publish" in lowered:
            plan.append(Subtask(f"s{len(plan) + 1}",
                                "send the deliverable to stakeholders",
                                "writing", depends_on=[plan[-1].subtask_id],
                                sensitive=True))
        if "prefer" in hints and any("bullet" in m["text"]
                                     for m in memories):
            for subtask in plan:
                if subtask.specialist == "writing":
                    subtask.description += " (user prefers bullet points)"
        return plan

    def plan_confidence(self, task: str, memories: list[dict]) -> float:
        """Low confidence on vague or unfamiliar tasks triggers plan
        approval by a human before any work starts."""
        words = len(task.split())
        confidence = 0.55 + min(0.3, words / 40)
        if memories:
            confidence += 0.1
        if words < 4 or "somehow" in task.lower() or "?" in task:
            confidence -= 0.25
        return round(max(0.1, min(0.99, confidence)), 2)


class Specialist:
    def __init__(self, name: str, registry: ToolRegistry):
        self.name = name
        self.registry = registry
        self.fail_next = 0  # test hook: force failures

    def execute(self, task_id: str, subtask: Subtask,
                context: dict) -> SubtaskResult:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError(f"{self.name} failed on"
                               f" {subtask.subtask_id} (injected)")
        calls = []
        if self.name == "research":
            call = self.registry.invoke(task_id, self.name, "web_search",
                                        query=subtask.description)
            calls.append(call)
            output = "sources:\n" + "\n".join(
                f"- {s}" for s in call.output)
        elif self.name == "analysis":
            call = self.registry.invoke(task_id, self.name,
                                        "query_database",
                                        sql="SELECT region, revenue ...")
            calls.append(call)
            rows = call.output
            total = sum(r["revenue"] for r in rows)
            upstream = context.get("s1", "")
            output = (f"analysis over {len(rows)} regions,"
                      f" total {total}:\n"
                      + "\n".join(f"- {r['region']}: {r['revenue']}"
                                  for r in rows))
            if "sources" in str(upstream):
                output += "\ncross checked against gathered sources"
        elif self.name == "code":
            call = self.registry.invoke(task_id, self.name, "run_code",
                                        code="print(sum(range(10)))")
            calls.append(call)
            output = f"code result: {call.output}"
        else:  # writing
            upstream = "\n".join(str(v) for v in context.values())
            style = "bullet points" if "bullet" in subtask.description \
                else "prose"
            output = (f"deliverable ({style}) for"
                      f" '{subtask.description[:50]}'\n"
                      f"grounded in {len(context)} upstream outputs,"
                      f" {len(upstream)} chars of evidence")
            call = self.registry.invoke(task_id, self.name, "write_file",
                                        path=f"{task_id}/{subtask.subtask_id}.md",
                                        content=output)
            calls.append(call)
        seed = _seed(subtask.description + self.name)
        confidence = round(0.72 + (seed % 25) / 100, 2)
        return SubtaskResult(
            subtask.subtask_id, self.name, output,
            [{"tool": c.tool, "ok": c.ok, "latency_ms": c.latency_ms}
             for c in calls],
            confidence)


class Reviewer:
    """Validates specialist output before the supervisor accepts it."""

    QUALITY_FLOOR = 0.55

    def review(self, subtask: Subtask, result: SubtaskResult) -> dict:
        score = 0.9
        problems = []
        if len(result.output) < 40:
            score -= 0.3
            problems.append("output too thin for the request")
        if subtask.expected_output == "table" and \
                "-" not in result.output:
            score -= 0.3
            problems.append("expected tabular breakdown, none present")
        if "[missing file" in result.output or \
                "unknown tool" in result.output:
            score -= 0.4
            problems.append("a tool call failed inside the output")
        if "injected-bad" in result.output:
            score -= 0.5
            problems.append("fabricated content detected")
        score = round(max(0.0, score), 2)
        return {"approved": score >= self.QUALITY_FLOOR, "score": score,
                "problems": problems,
                "feedback": "; ".join(problems) or "clean"}
