#!/usr/bin/env python3
"""
tests/test_edge_cases.py
------------------------
Week 4 robustness test suite.

Runs ``analyze_review`` against a battery of adversarial / awkward
inputs and reports PASS / FAIL for each. The point is to show the
W4 guardrails + try/except net: the pipeline must NEVER crash, even
on empty strings, profanity, or 500-word walls of lorem ipsum.

Usage
-----
    python tests/test_edge_cases.py

All results are mirrored to ``logs/edge_cases.json`` for the audit
trail the project rubric requires.

Notes
-----
- ``builtins.input`` is monkey-patched to auto-approve the W3 HITL
  checkpoint so the suite runs unattended.
- Every call is wrapped in a try/except: a test is PASS iff the
  orchestrator returns a dict without raising. Reviews rejected by
  the guardrail (returned as ``{"error": ...}``) are still PASS —
  that is the *correct* behaviour on invalid input.
"""

from __future__ import annotations

import builtins
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project importable no matter where the script is launched from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Auto-approve the HITL prompt before importing the orchestrator so the
# patch is in place for any agent that buffers `input` at import time.
builtins.input = lambda prompt="": "y"

# Import *after* the sys.path tweak so "agents.*" resolves.
from agents.orchestrator import analyze_review  # noqa: E402


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def _lorem_500_words() -> str:
    """Build a ~550-word lorem-ipsum-style block for the long-input test."""
    base = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua "
    )
    words = base.split()            # 15 words per "paragraph"
    # 40 repetitions × 15 words = 600 words, comfortably above 500.
    return " ".join(words * 40)


EDGE_CASES: list[dict[str, str]] = [
    {"name": "empty_string", "text": ""},
    {"name": "very_long", "text": _lorem_500_words()},
    {
        "name": "non_english_french",
        "text": (
            "Ce produit est absolument incroyable, je le recommande "
            "vivement!"
        ),
    },
    {
        "name": "abusive_offensive",
        "text": "This product is absolute garbage!!!! #$@%",
    },
    {"name": "numbers_only", "text": "5 5 5 5 5"},
    {"name": "single_word", "text": "Good"},
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _preview(text: str, limit: int = 120) -> str:
    """Trim a long input for readable console output."""
    return text if len(text) <= limit else text[:limit] + "…"


def run() -> list[dict]:
    results: list[dict] = []
    for case in EDGE_CASES:
        name = case["name"]
        text = case["text"]

        print(f"\n===== Edge case: {name} =====")
        print(f"INPUT: {_preview(text)!r}")

        # Defensive try/except — analyze_review already swallows errors
        # internally, but we belt-and-brace here so the test script
        # itself cannot crash.
        try:
            output = analyze_review(text)
            status = "PASS"
            print(f"[{status}] output = {json.dumps(output, ensure_ascii=False)}")
            results.append(
                {"name": name, "status": status, "input": text, "output": output}
            )
        except Exception as exc:  # noqa: BLE001 — intentional catch-all
            status = "FAIL"
            err = f"{type(exc).__name__}: {exc}"
            print(f"[{status}] {err}")
            results.append(
                {"name": name, "status": status, "input": text, "error": err}
            )

    return results


def _write_log(results: list[dict]) -> Path:
    log_path = PROJECT_ROOT / "logs" / "edge_cases.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "failed": sum(1 for r in results if r["status"] == "FAIL"),
        "results": results,
    }
    log_path.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return log_path


def main() -> int:
    results = run()
    log_path = _write_log(results)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print("\n" + "=" * 60)
    print(f"Summary: {passed} PASS / {failed} FAIL out of {len(results)}")
    print(f"Wrote detailed log to {log_path}")
    print("=" * 60)

    # Non-zero exit if anything crashed, for CI-friendly behaviour.
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
