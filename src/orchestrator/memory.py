"""Two tier memory: task scoped working memory and long term semantic
memory with importance scoring, consolidation, and decay."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field


def embed(text: str) -> dict[str, float]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    counts = Counter(tokens)
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {t: v / norm for t, v in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(b) < len(a):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


class WorkingMemory:
    """Shared scratch space for one task. Cleared when the task ends.
    In production this maps to a Redis hash keyed by task id."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def put(self, task_id: str, key: str, value) -> None:
        self._store.setdefault(task_id, {})[key] = value

    def get(self, task_id: str, key: str, default=None):
        return self._store.get(task_id, {}).get(key, default)

    def all(self, task_id: str) -> dict:
        return dict(self._store.get(task_id, {}))

    def clear(self, task_id: str) -> None:
        self._store.pop(task_id, None)


@dataclass
class Memory:
    memory_id: str
    kind: str          # approach | fact | preference
    text: str
    user: str
    importance: float = 1.0
    accesses: int = 0
    created: float = field(default_factory=time.time)
    vector: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.vector:
            self.vector = embed(self.text)


class LongTermMemory:
    """Semantic store the supervisor queries before planning.
    ChromaDB fills this role in a live deployment."""

    DECAY_HALF_LIFE_DAYS = 30.0

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        self._counter = 0

    def remember(self, kind: str, text: str, user: str,
                 importance: float = 1.0) -> Memory:
        self._counter += 1
        memory = Memory(f"mem-{self._counter:04d}", kind, text, user,
                        importance)
        self.memories[memory.memory_id] = memory
        return memory

    def recall(self, query: str, user: str, top_k: int = 3,
               now: float | None = None) -> list[dict]:
        now = now or time.time()
        vector = embed(query)
        scored = []
        for memory in self.memories.values():
            if memory.user not in (user, "*"):
                continue
            age_days = (now - memory.created) / 86400
            decay = 0.5 ** (age_days / self.DECAY_HALF_LIFE_DAYS)
            relevance = cosine(vector, memory.vector)
            score = relevance * memory.importance * decay
            if relevance > 0.05:
                scored.append((score, memory))
        scored.sort(key=lambda pair: -pair[0])
        results = []
        for score, memory in scored[:top_k]:
            memory.accesses += 1
            memory.importance = min(3.0, memory.importance + 0.1)
            results.append({"memory_id": memory.memory_id,
                            "kind": memory.kind, "text": memory.text,
                            "score": round(score, 4)})
        return results

    def consolidate(self, similarity: float = 0.75) -> int:
        """Merge near duplicate memories into one with summed importance."""
        merged = 0
        items = list(self.memories.values())
        removed: set[str] = set()
        for i, first in enumerate(items):
            if first.memory_id in removed:
                continue
            for second in items[i + 1:]:
                if second.memory_id in removed:
                    continue
                if first.kind == second.kind and \
                        cosine(first.vector, second.vector) >= similarity:
                    first.importance = min(
                        3.0, first.importance + second.importance * 0.5)
                    removed.add(second.memory_id)
                    merged += 1
        for memory_id in removed:
            del self.memories[memory_id]
        return merged

    def forget_user(self, user: str) -> int:
        """Delete endpoint for user data requests."""
        doomed = [m for m in self.memories.values() if m.user == user]
        for memory in doomed:
            del self.memories[memory.memory_id]
        return len(doomed)

    def dashboard(self, user: str) -> list[dict]:
        return [{"memory_id": m.memory_id, "kind": m.kind, "text": m.text,
                 "importance": round(m.importance, 2),
                 "accesses": m.accesses}
                for m in self.memories.values() if m.user in (user, "*")]
