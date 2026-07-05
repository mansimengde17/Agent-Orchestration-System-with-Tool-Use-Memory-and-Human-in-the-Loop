"""Tool registry: every capability an agent can invoke.

Tools are registered with schemas, the specialists allowed to call them,
and rate limits. Every invocation is logged with inputs, outputs,
latency, and outcome. The implementations are deterministic simulations
so the whole system runs offline; each one marks where a live backend
plugs in.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


@dataclass
class ToolSpec:
    name: str
    description: str
    allowed_agents: list[str]
    rate_limit_per_task: int
    handler: object = None


@dataclass
class ToolCall:
    tool: str
    agent: str
    inputs: dict
    output: object
    latency_ms: float
    ok: bool
    at: float = field(default_factory=time.time)


SEARCH_CORPUS = {
    "electric vehicle market": [
        "Global EV sales grew 31 percent year over year in 2025",
        "Battery pack prices fell to 97 dollars per kWh on average",
        "China accounted for 58 percent of global EV deliveries",
    ],
    "battery supply chain": [
        "Lithium carbonate spot prices stabilized after a two year slide",
        "Three new cathode plants announced in North America",
    ],
    "quarterly revenue": [
        "Sample Corp reported quarterly revenue of 4.2 billion dollars",
        "Services segment grew 18 percent while hardware was flat",
    ],
}


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, ToolSpec] = {}
        self.calls: list[ToolCall] = []
        self._task_counts: dict[tuple[str, str], int] = {}
        self._register_defaults()

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def _register_defaults(self) -> None:
        self.register(ToolSpec(
            "web_search", "search the web and return snippets",
            ["research"], rate_limit_per_task=6, handler=self._web_search))
        self.register(ToolSpec(
            "read_file", "read a file from the shared workspace",
            ["research", "analysis", "writing"], 10, self._read_file))
        self.register(ToolSpec(
            "write_file", "write a file to the shared workspace",
            ["writing", "code"], 10, self._write_file))
        self.register(ToolSpec(
            "run_code", "execute python in a sandbox and return stdout",
            ["code", "analysis"], 4, self._run_code))
        self.register(ToolSpec(
            "query_database", "run a read only SQL query",
            ["analysis"], 6, self._query_database))
        self.files: dict[str, str] = {}

    # --- handlers (deterministic simulations) ---
    def _web_search(self, agent: str, query: str) -> list[str]:
        for topic, snippets in SEARCH_CORPUS.items():
            if any(word in query.lower() for word in topic.split()):
                return snippets
        return [f"no strong results for '{query}',"
                " broader query recommended"]

    def _read_file(self, agent: str, path: str) -> str:
        return self.files.get(path, f"[missing file: {path}]")

    def _write_file(self, agent: str, path: str, content: str) -> str:
        self.files[path] = content
        return f"wrote {len(content)} chars to {path}"

    def _run_code(self, agent: str, code: str) -> str:
        # Sandbox stand in: evaluates a tiny arithmetic subset.
        seed = _seed(code)
        if "sum" in code or "+" in code:
            return f"result: {seed % 1000}"
        return f"executed {len(code.splitlines())} lines, exit 0"

    def _query_database(self, agent: str, sql: str) -> list[dict]:
        return [{"region": "NA", "revenue": 1800},
                {"region": "EU", "revenue": 1400},
                {"region": "APAC", "revenue": 1000}]

    # --- invocation with permission and rate limit enforcement ---
    def invoke(self, task_id: str, agent: str, tool_name: str,
               **inputs) -> ToolCall:
        spec = self.tools.get(tool_name)
        start = time.perf_counter()
        if spec is None:
            call = ToolCall(tool_name, agent, inputs,
                            f"unknown tool {tool_name}", 0.0, ok=False)
            self.calls.append(call)
            return call
        if agent not in spec.allowed_agents:
            call = ToolCall(tool_name, agent, inputs,
                            f"agent {agent} may not use {tool_name}",
                            0.0, ok=False)
            self.calls.append(call)
            return call
        key = (task_id, tool_name)
        self._task_counts[key] = self._task_counts.get(key, 0) + 1
        if self._task_counts[key] > spec.rate_limit_per_task:
            call = ToolCall(tool_name, agent, inputs,
                            "rate limit exceeded for this task", 0.0,
                            ok=False)
            self.calls.append(call)
            return call
        output = spec.handler(agent, **inputs)
        call = ToolCall(tool_name, agent, inputs, output,
                        round((time.perf_counter() - start) * 1000, 3),
                        ok=True)
        self.calls.append(call)
        return call
