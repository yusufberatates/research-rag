"""Thread-safe rate limiters shared across parallel download workers.

Each limiter enforces a minimum spacing between calls regardless of how many
worker threads call it, so the whole process stays within a source's
published request rate.
"""
from __future__ import annotations

import threading
import time

from research_rag.config import ARXIV_RATE_PER_SEC, S2_RATE_PER_MIN


class RateLimiter:
    """Token-free min-interval limiter: blocks until ``min_interval`` has
    elapsed since the previous acquisition, across all threads."""

    def __init__(self, rate_per_sec: float):
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


# Process-wide limiters, one per source/host.
arxiv_limiter = RateLimiter(ARXIV_RATE_PER_SEC)            # ~3 requests / second
# Driven by config (default S2_RATE_PER_MIN=55 -> ~0.92 req/s, just under S2's
# 1 req/s keyed ceiling). NOTE: this paces the /paper and /references calls but
# S2 throttles /paper/search far harder server-side, so search 429s in bursts
# even when this limiter is obeyed and a valid key is attached.
s2_limiter = RateLimiter(S2_RATE_PER_MIN / 60.0)
