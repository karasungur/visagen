"""
GUI utilities.

Provides:
- Debouncer: Rate-limiting utility for real-time UI updates.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class Debouncer:
    """
    Debounce utility to limit frequency of function calls.

    Useful for real-time UI updates where an event (like slider change)
    triggers expensive operations.

    Example:
        >>> debouncer = Debouncer(delay=0.5)
        >>> def update(val): print(val)
        >>> debouncer.call(update, 1)  # Scheduled
        >>> debouncer.call(update, 2)  # Replaces previous schedule
        >>> # ... 0.5s later ...
        >>> # prints: 2
    """

    def __init__(self, delay: float = 0.3) -> None:
        """
        Initialize debouncer.

        Args:
            delay: Delay in seconds before executing the call.
        """
        self.delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """
        Schedule a function call.

        If a call is already scheduled, it is cancelled and the new one
        replaces it.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            self._timer = threading.Timer(self.delay, func, args=args, kwargs=kwargs)
            self._timer.start()

    def cancel(self) -> None:
        """Cancel any pending call."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


class Throttler:
    """
    Throttle utility to ensure function is called at most once every N seconds.
    """

    def __init__(self, rate_limit: float = 0.3) -> None:
        self.rate_limit = rate_limit
        self.last_call = 0.0

    def should_call(self) -> bool:
        """Check if enough time has passed since last call."""
        now = time.time()
        if now - self.last_call >= self.rate_limit:
            self.last_call = now
            return True
        return False
