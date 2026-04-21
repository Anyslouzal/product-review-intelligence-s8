"""
main.py
-------
Week 1 entry point. Runs the orchestrator agent against three hardcoded
example reviews (positive / negative / mixed), prints the structured
result to the console, and appends each run to logs/run_log.json with a
UTC timestamp.

Why a JSON log?
    - It is machine-readable (easy to diff between weekly runs).
    - It gives the oral-defence panel a clear audit trail of every
      agent action, which is an explicit project constraint.
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


def main() -> None:
    """Analyse each example review and log the outcome."""
    for example in EXAMPLE_REVIEWS:
        label = example["label"]
        review = example["text"]

        print(f"\n=== Analysing {label.upper()} review ===")
        print(review)

        # Core call into the CrewAI agent.
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

    print(f"\nDone. Run appended to {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
