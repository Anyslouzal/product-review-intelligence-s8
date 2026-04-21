"""
tools/guardrails.py
-------------------
Week 4 deliverable: input validation and sanitisation that runs
*before* any review reaches the CrewAI pipeline.

The goal is to make ``analyze_review`` robust by rejecting obviously
broken inputs cheaply, instead of letting them burn LLM quota and
potentially crash the crew.

Rules:
    - Reject empty / whitespace-only strings.
    - Reject reviews shorter than ``MIN_WORDS`` (currently 3 words).
    - Reject reviews longer than ``MAX_WORDS`` (currently 1000 words).
    - Strip leading / trailing whitespace.
    - Collapse any run of whitespace into a single space.

Public API:
    ``validate_review(text: str) -> tuple[bool, str]``
        - ``(True, cleaned_text)`` when the input is acceptable.
        - ``(False, reason)``       when the input must be refused.
"""

from __future__ import annotations

import re

# Tunables — kept at module level so tests and callers can inspect /
# monkey-patch them if needed.
MIN_WORDS: int = 3
MAX_WORDS: int = 1000

# Pre-compiled regex for collapsing internal whitespace (spaces,
# tabs, newlines, etc.) into a single space. Compiling once keeps
# repeated calls cheap when the pipeline processes a batch.
_WHITESPACE_RUN = re.compile(r"\s+")


def validate_review(text: str) -> tuple[bool, str]:
    """Validate + sanitise a raw review string.

    Parameters
    ----------
    text : str
        The user-supplied review text.

    Returns
    -------
    tuple[bool, str]
        ``(True, cleaned_text)`` if the input is acceptable.
        ``(False, reason)`` otherwise — ``reason`` is a short human-
        readable message suitable for logging or returning to the
        caller.
    """
    # Defensive type check — the rest of the function assumes `str`.
    if not isinstance(text, str):
        return (False, f"Review must be a string (got {type(text).__name__}).")

    # 1. Strip leading/trailing whitespace.
    cleaned = text.strip()
    if not cleaned:
        return (False, "Review is empty.")

    # 2. Normalise internal whitespace (multiple spaces, tabs, newlines).
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)

    # 3. Word-count bounds. `split()` without arguments already handles
    #    any whitespace and skips empty tokens, so after the previous
    #    normalisation step the count is accurate.
    word_count = len(cleaned.split())
    if word_count < MIN_WORDS:
        return (
            False,
            f"Review too short: {word_count} word(s); "
            f"minimum is {MIN_WORDS}.",
        )
    if word_count > MAX_WORDS:
        return (
            False,
            f"Review too long: {word_count} words; "
            f"maximum is {MAX_WORDS}.",
        )

    return (True, cleaned)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = [
        "",
        "   ",
        "Good",
        "This is great!",
        "  Absolutely   love\tthis  product!!  ",
        "word " * 1200,
    ]
    for c in cases:
        ok, msg = validate_review(c)
        preview = c if len(c) < 60 else c[:57] + "..."
        print(f"[{'OK' if ok else 'REJECT'}] {preview!r:<65} -> {msg!r}")
