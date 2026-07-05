"""Showcase scenario for the agent orchestration system.

A research task flows through the full lifecycle:
1. The supervisor recalls relevant memories and builds a plan.
2. Research, analysis, and writing specialists execute with tool calls.
3. The reviewer validates each output before it is accepted.
4. A sensitive final step (sending the report) escalates to a human.
5. Lessons learned are written to long term memory, and a repeat task
   plans better because of it.
6. A forced double failure shows the take_over escalation path.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from orchestrator.graph import Orchestrator


def section(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def print_trace(node: dict, depth: int = 0) -> None:
    marker = {"success": " ", "warning": "!", "failure": "X",
              "escalated": "H"}[node["status"]]
    print(f"  {'  ' * depth}[{marker}] {node['agent']}:"
          f" {node['action']} - {node['detail'][:58]}")
    for child in node["children"]:
        print_trace(child, depth + 1)


def main() -> None:
    orchestrator = Orchestrator()
    orchestrator.long_term.remember(
        "preference", "user prefers bullet points in written summaries",
        "analyst-1")

    section("Task 1: research report with a sensitive send step")
    result = orchestrator.run(
        "Research the electric vehicle market and email the report"
        " to stakeholders", user="analyst-1")
    print(f"  status: {result['status']}")
    print(f"  subtasks completed: {len(result['results'])}")
    totals = result["trace_totals"]
    print(f"  tokens {totals['tokens']}, cost ${totals['cost_usd']},"
          f" escalations {totals['escalations']},"
          f" human seconds {totals['human_seconds']}")

    section("Execution trace")
    trace = orchestrator.traces[result["task_id"]]
    print_trace(trace.root.to_dict())

    section("Task 2: memory makes the repeat task smarter")
    repeat = orchestrator.run(
        "Research the battery supply chain market and summarize",
        user="analyst-1")
    memories = orchestrator.long_term.recall(
        "research market report", "analyst-1")
    print(f"  status: {repeat['status']}")
    print(f"  memories now available to planning: {len(memories)}")
    for memory in memories:
        print(f"    [{memory['kind']}] {memory['text'][:60]}")

    section("Task 3: double failure escalates to take_over")
    orchestrator.specialists["research"].fail_next = 2
    failed = orchestrator.run(
        "Research the quarterly revenue numbers and report",
        user="analyst-1")
    print(f"  status: {failed['status']}")
    escalations = [e for e in orchestrator.approvals.items.values()
                   if e.task_id == failed["task_id"]]
    for escalation in escalations:
        print(f"  escalation {escalation.escalation_id}:"
              f" {escalation.level} - {escalation.reason}"
              f" -> {escalation.status}")

    section("Memory dashboard for analyst-1")
    for memory in orchestrator.long_term.dashboard("analyst-1"):
        print(f"  {memory['memory_id']} [{memory['kind']}]"
              f" importance {memory['importance']}"
              f" accesses {memory['accesses']}: {memory['text'][:52]}")

    print("\nDemo complete. Start the API with:"
          " uvicorn orchestrator.api:app --app-dir src")


if __name__ == "__main__":
    main()
