"""
generation/rate_limiter.py

A client-side token-bucket rate limiter for Groq's free-tier
tokens-per-minute (TPM) limit.

WHY THIS EXISTS: Groq's client library silently retries (with backoff)
when it hits a 429 rate-limit response, up to max_retries times
(default 2). This makes rate-limiting invisible - it just looks like
"the API is randomly slow sometimes." Measured against this project's
real prompt sizes (~2,200 tokens/question across generation + judge
calls) against a 6,000 TPM budget, the budget runs out after roughly
2-3 questions, so nearly every subsequent call was silently waiting
out Groq's retry backoff.

This class instead tracks token usage explicitly in a rolling 60-second
window and sleeps proactively, for exactly as long as needed - no more,
no less - before a call that would exceed the budget. It also prints
when and why it's waiting, so pacing is visible instead of mysterious.

This is a simplified token bucket: rather than modeling a continuously
refilling bucket, it tracks a rolling window of (timestamp, tokens)
events and sums tokens still "in window" (< 60s old). Simpler to reason
about and test than a true continuous bucket, and accurate enough for
this use case.
"""

import time
from collections import deque


class TokenRateLimiter:
    def __init__(self, max_tokens_per_minute: int, window_seconds: float = 60.0):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.window_seconds = window_seconds
        # Each entry: (timestamp, token_count). A deque so we can cheaply
        # drop old entries off the left as the window slides forward.
        self._usage_log: deque[tuple[float, int]] = deque()

    def _purge_old_entries(self, now: float) -> None:
        while self._usage_log and (now - self._usage_log[0][0]) >= self.window_seconds:
            self._usage_log.popleft()

    def _tokens_used_in_window(self, now: float) -> int:
        self._purge_old_entries(now)
        return sum(tokens for _, tokens in self._usage_log)

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """
        Call this BEFORE making an API call. Blocks (sleeps) if adding
        estimated_tokens would exceed the budget within the current
        rolling window - sleeps only long enough for the oldest entry
        to age out of the window, then re-checks (in case that alone
        isn't enough).
        """
        while True:
            now = time.time()
            used = self._tokens_used_in_window(now)
            if used + estimated_tokens <= self.max_tokens_per_minute:
                return  # budget available, proceed immediately

            # Not enough budget yet - sleep until the oldest entry in
            # the window expires, freeing up its tokens, then re-check.
            oldest_timestamp = self._usage_log[0][0]
            sleep_for = self.window_seconds - (now - oldest_timestamp) + 0.1  # +0.1s safety margin
            sleep_for = max(sleep_for, 0.1)
            print(
                f"    [rate limiter] {used}/{self.max_tokens_per_minute} tokens used in "
                f"the last {self.window_seconds:.0f}s - waiting {sleep_for:.1f}s for budget to free up"
            )
            time.sleep(sleep_for)

    def record_usage(self, actual_tokens: int) -> None:
        """
        Call this AFTER a successful API call, with the real token
        count from the response (response.usage.total_tokens) - more
        accurate than the pre-call estimate, so future wait_if_needed()
        calls are based on real data, not guesses.
        """
        self._usage_log.append((time.time(), actual_tokens))