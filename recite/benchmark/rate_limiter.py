"""Thread-safe rate limiter for API calls."""

import threading
import time
from typing import Optional


class SharedRateLimiter:
    """
    Thread-safe rate limiter: at most 1 call per min_interval seconds across all callers.
    Call acquire() before each API request; it blocks until the interval has elapsed.
    """

    def __init__(self, requests_per_second: float = 5.0):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._last_acquire = 0.0

    def acquire(self) -> None:
        """Block until at least min_interval seconds since last acquire, then return."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_acquire
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_acquire = time.time()
