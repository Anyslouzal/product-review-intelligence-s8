"""
main.py
-------
Demo entry point for the product-review-intelligence pipeline.

Stages:
    1. Analyse three canonical reviews (positive / negative / mixed).
    2. (W4) Run three edge cases through the guardrails so a reviewer
       can see the robustness layer in action without reading the
       test harness.

Every run is appended to logs/run_log.json with a UTC timestamp. The
file is machine-readable, provides an auditable trail of every agent
action, and is easy to diff between weekly deliverables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator import analyze_review
from config import RUN_LOG_PATH

# Three canonical test reviews — one per sentiment class — chosen so the
# agent's behaviour is easy to verify by eye during a demo.
EXAMPLE_REVIEWS: list[dict[str, str]] = [
    {
        "label": "positive",
        "text": (
            "Absolutely love this moisturizer! My skin feels hydrated all "
            "day, the scent is subtle, and a little goes a long way. "
            "Best beauty purchase I've made this year."
        ),
    },
    {
        "label": "negative",
        "text": (
            "Terrible experience. The bottle arrived half-empty, the "
            "product smells chemical, and it broke me out within two "
            "days. Completely wasted my money — would not recommend."
        ),
    },
    {
        "label": "mixed",
        "text": (
            "The packaging is gorgeous and the shade range is impressive, "
            "but the formula is way too drying on my skin and it creases "
            "around the eyes after a few hours. Torn about keeping it."
        ),
    },
]

# W4: three deliberately awkward inputs to demonstrate the guardrails.
# The first and third should be rejected by `validate_review` BEFORE
# the crew is spun up; the French review should pass the guardrail
# and exercise the full pipeline on non-English text.
EDGE_CASE_REVIEWS: list[dict[str, str]] = [
    {"label": "edge_empty", "text": ""},
    {
        "label": "edge_french",
        "text": (
            "Ce produit est absolument incroyable, je le recommande "
            "vivement!"
        ),
    },
    {"label": "edge_single_word", "text": "Good"},
]


def _append_log_entry(entry: dict) -> None:
    """Append a single run record to logs/run_log.json.

    The file stores a JSON array so downstream tooling (pandas, jq) can
    read it in one shot. We load-modify-write rather than streaming
    because the log is small and we want it to remain valid JSON at all
    times.
    """
    log_path: Path = RUN_LOG_PATH
    if log_path.exists() and log_path.stat().st_size > 0:
        with log_path.open("r", encoding="utf-8") as fp:
            try:
                existing = json.load(fp)
                if not isinstance(existing, list):
                    existing = [existing]
            except json.JSONDecodeError:
                # Corrupted log — start fresh rather than crashing the run.
                existing = []
    else:
        existing = []

    existing.append(entry)

    with log_path.open("w", encoding="utf-8") as fp:
        json.dump(existing, fp, ensure_ascii=False, indent=2)


def _run_batch(batch: list[dict[str, str]], banner: str) -> None:
    """Analyse every review in ``batch`` and mirror the result to the log."""
    print(f"\n{'#' * 72}\n# {banner}\n{'#' * 72}")
    for example in batch:
        label = example["label"]
        review = example["text"]

        print(f"\n=== Analysing {label.upper()} review ===")
        print(review if review else "<empty>")

        # Core call into the CrewAI agent. analyze_review never raises
        # thanks to the W4 guardrail + try/except net, so we do not
        # need a try/except here.
        result = analyze_review(review)

        # Pretty-print to the console so the demo output is legible.
        print("--- Agent output ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Build an auditable log record: timestamp + input + output.
        log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "agent": "Product Review Orchestrator",
            "input_label": label,
            "input_review": review,
            "output": result,
        }
        _append_log_entry(log_entry)


def main() -> None:
    """Analyse the canonical examples then the W4 edge cases."""
    _run_batch(EXAMPLE_REVIEWS, "Canonical reviews (positive / negative / mixed)")
    _run_batch(EDGE_CASE_REVIEWS, "W4 edge cases (guardrails in action)")
    print(f"\nDone. Run appended to {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
