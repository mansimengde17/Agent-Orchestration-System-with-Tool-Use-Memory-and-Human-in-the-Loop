import sys
import unittest

sys.path.insert(0, "src")

from orchestrator.graph import Orchestrator
from orchestrator.memory import LongTermMemory
from orchestrator.tools import ToolRegistry


class ToolTests(unittest.TestCase):
    def test_permission_enforced(self):
        registry = ToolRegistry()
        call = registry.invoke("t1", "writing", "web_search", query="x")
        self.assertFalse(call.ok)

    def test_rate_limit_enforced(self):
        registry = ToolRegistry()
        for _ in range(6):
            registry.invoke("t1", "research", "web_search", query="x")
        call = registry.invoke("t1", "research", "web_search", query="x")
        self.assertFalse(call.ok)
        self.assertIn("rate limit", str(call.output))


class MemoryTests(unittest.TestCase):
    def test_recall_scopes_to_user(self):
        memory = LongTermMemory()
        memory.remember("fact", "revenue data lives in warehouse", "a")
        memory.remember("fact", "revenue data lives in warehouse", "b")
        results = memory.recall("where is revenue data", "a")
        self.assertEqual(len(results), 1)

    def test_consolidation_merges_duplicates(self):
        memory = LongTermMemory()
        memory.remember("fact", "user prefers short bullet summaries", "a")
        memory.remember("fact", "user prefers short bullet summaries!", "a")
        merged = memory.consolidate()
        self.assertEqual(merged, 1)
        self.assertEqual(len(memory.memories), 1)

    def test_forget_user(self):
        memory = LongTermMemory()
        memory.remember("fact", "something", "a")
        self.assertEqual(memory.forget_user("a"), 1)


class OrchestrationTests(unittest.TestCase):
    def test_full_task_completes(self):
        orchestrator = Orchestrator()
        result = orchestrator.run(
            "Research the electric vehicle market and write a report")
        self.assertEqual(result["status"], "completed")
        self.assertGreaterEqual(len(result["results"]), 3)
        self.assertIn("Deliverable", result["deliverable"])

    def test_sensitive_step_escalates(self):
        orchestrator = Orchestrator()
        result = orchestrator.run(
            "Research the electric vehicle market and email the report")
        self.assertGreaterEqual(result["trace_totals"]["escalations"], 1)

    def test_manual_approval_pauses_execution(self):
        orchestrator = Orchestrator()
        result = orchestrator.run(
            "Research the electric vehicle market and email stakeholders",
            auto_approve=False)
        self.assertEqual(result["status"], "waiting_for_human")
        escalation_id = result["pending_escalation"]
        resumed = orchestrator.resolve_and_continue(
            escalation_id, "approved", "op-1")
        self.assertEqual(resumed["status"], "completed")

    def test_rejection_stops_task(self):
        orchestrator = Orchestrator()
        result = orchestrator.run(
            "Research the electric vehicle market and email stakeholders",
            auto_approve=False)
        resumed = orchestrator.resolve_and_continue(
            result["pending_escalation"], "rejected", "op-1",
            note="not approved for external send")
        self.assertEqual(resumed["status"], "rejected_by_human")

    def test_double_failure_takes_over(self):
        orchestrator = Orchestrator()
        orchestrator.specialists["research"].fail_next = 2
        result = orchestrator.run(
            "Research the quarterly revenue and report")
        self.assertEqual(result["status"], "completed")
        levels = [e.level for e in
                  orchestrator.approvals.items.values()]
        self.assertIn("take_over", levels)

    def test_memory_written_after_completion(self):
        orchestrator = Orchestrator()
        orchestrator.run("Research the electric vehicle market report",
                         user="u1")
        self.assertGreaterEqual(
            len(orchestrator.long_term.dashboard("u1")), 1)


if __name__ == "__main__":
    unittest.main()
