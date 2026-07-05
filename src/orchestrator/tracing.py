"""Execution tracing, cost tracking, and replay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

TOKEN_PRICE_PER_1K = {"supervisor": 0.01, "research": 0.002,
                      "analysis": 0.002, "writing": 0.004,
                      "code": 0.002, "reviewer": 0.003}


@dataclass
class TraceNode:
    node_id: str
    agent: str
    action: str
    detail: str
    status: str = "success"     # success | warning | failure | escalated
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    children: list["TraceNode"] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"node_id": self.node_id, "agent": self.agent,
                "action": self.action, "detail": self.detail,
                "status": self.status, "tokens": self.tokens,
                "cost_usd": self.cost_usd, "latency_ms": self.latency_ms,
                "children": [c.to_dict() for c in self.children]}


class TaskTrace:
    def __init__(self, task_id: str, task: str):
        self.task_id = task_id
        self.task = task
        self.root = TraceNode("root", "system", "task_intake", task)
        self._counter = 0
        self.human_seconds = 0.0

    def add(self, parent: TraceNode, agent: str, action: str,
            detail: str, status: str = "success",
            tokens: int = 0, latency_ms: float = 0.0) -> TraceNode:
        self._counter += 1
        cost = round(tokens / 1000
                     * TOKEN_PRICE_PER_1K.get(agent, 0.002), 6)
        node = TraceNode(f"n{self._counter:03d}", agent, action, detail,
                         status, tokens, cost, latency_ms)
        parent.children.append(node)
        return node

    def totals(self) -> dict:
        tokens = cost = calls = 0
        failures = escalations = 0

        def walk(node: TraceNode):
            nonlocal tokens, cost, calls, failures, escalations
            tokens += node.tokens
            cost += node.cost_usd
            calls += 1
            if node.status == "failure":
                failures += 1
            if node.status == "escalated":
                escalations += 1
            for child in node.children:
                walk(child)
        walk(self.root)
        return {"tokens": tokens, "cost_usd": round(cost, 6),
                "nodes": calls, "failures": failures,
                "escalations": escalations,
                "human_seconds": self.human_seconds}


class Replayer:
    """Re-runs a stored task while letting the caller patch any input.
    Comparing the replay against the original shows exactly where an
    execution diverges, which is how failures get diagnosed."""

    def __init__(self, orchestrator_factory):
        self.factory = orchestrator_factory

    def replay(self, original_task: str, patch: str | None = None) -> dict:
        orchestrator = self.factory()
        task = patch if patch is not None else original_task
        result = orchestrator.run(task, user="replay")
        return {"task": task,
                "status": result["status"],
                "totals": result["trace_totals"]}
