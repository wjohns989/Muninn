"""
Muninn Session Inhibition
-------------------------
CoALA-style short-term inhibition: within one agent session, memories that
were already returned are demoted on later searches so fresh context can take
their slots instead of the same items being re-injected every turn.

Demotion is positional (a seen result moves down ``rank_penalty`` places in
the final ranked pool) rather than a score multiplier, because cross-encoder
scores are unbounded and can be negative. Strong repeats can therefore still
surface, while weaker repeats give way to unseen results.

State is bounded: at most ``max_sessions`` sessions (least recently used are
evicted), at most ``max_ids_per_session`` IDs per session, and every entry
expires after ``ttl_seconds``.
"""

import os
import threading
import time
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Sequence, Set, TypeVar

T = TypeVar("T")


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


class SessionInhibitor:
    def __init__(
        self,
        rank_penalty: int = 3,
        ttl_seconds: float = 1800.0,
        max_sessions: int = 256,
        max_ids_per_session: int = 200,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rank_penalty = max(0, int(rank_penalty))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_sessions = max(1, int(max_sessions))
        self.max_ids_per_session = max(1, int(max_ids_per_session))
        self._clock = clock
        self._sessions: "OrderedDict[str, OrderedDict[str, float]]" = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> Optional["SessionInhibitor"]:
        """Build from MUNINN_SESSION_INHIBITION_* settings; None when disabled."""
        if os.environ.get("MUNINN_SESSION_INHIBITION", "1").strip().lower() in ("0", "false", "no", "off"):
            return None
        return cls(
            rank_penalty=_env_int("MUNINN_SESSION_INHIBITION_RANK_PENALTY", 3, 0),
            ttl_seconds=_env_int("MUNINN_SESSION_INHIBITION_TTL_SEC", 1800, 1),
            max_sessions=_env_int("MUNINN_SESSION_INHIBITION_MAX_SESSIONS", 256, 1),
            max_ids_per_session=_env_int("MUNINN_SESSION_INHIBITION_MAX_IDS", 200, 1),
        )

    def seen(self, session_id: str) -> Set[str]:
        """Return the unexpired memory IDs already returned in this session."""
        with self._lock:
            entries = self._sessions.get(session_id)
            if entries is None:
                return set()
            self._expire(entries)
            return set(entries)

    def rerank(self, session_id: str, ranked: Sequence[T], key: Callable[[T], str]) -> List[T]:
        """Demote already-seen items by ``rank_penalty`` positions (stable for ties)."""
        seen = self.seen(session_id)
        if not seen or self.rank_penalty == 0:
            return list(ranked)
        # Unseen items win ties so a demoted item lands just after them.
        order = sorted(
            range(len(ranked)),
            key=lambda i: (i + self.rank_penalty, 1) if key(ranked[i]) in seen else (i, 0),
        )
        return [ranked[i] for i in order]

    def record(self, session_id: str, memory_ids: Iterable[str]) -> None:
        """Mark memory IDs as returned in this session."""
        now = self._clock()
        with self._lock:
            entries = self._sessions.get(session_id)
            if entries is None:
                entries = OrderedDict()
                self._sessions[session_id] = entries
            self._sessions.move_to_end(session_id)
            for memory_id in memory_ids:
                entries[memory_id] = now
                entries.move_to_end(memory_id)
            while len(entries) > self.max_ids_per_session:
                entries.popitem(last=False)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _expire(self, entries: "OrderedDict[str, float]") -> None:
        cutoff = self._clock() - self.ttl_seconds
        while entries and next(iter(entries.values())) < cutoff:
            entries.popitem(last=False)
