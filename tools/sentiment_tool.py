"""
tools/sentiment_tool.py
-----------------------
CrewAI-compatible tool wrapping the fine-tuned BERT sentiment classifier
produced by `notebooks/bert_finetuning.ipynb`.

Why a dedicated tool?
    In Week 1 the orchestrator agent did sentiment estimation itself via
    the LLM — fast to prototype but expensive and non-deterministic.
    Wrapping a local fine-tuned BERT as a CrewAI tool gives the agent a
    fast, deterministic, and offline-capable sentiment classifier.

Version note:
    crewai==0.5.0 does not ship `crewai.tools.BaseTool` yet. We try the
    CrewAI import first and fall back to LangChain's `BaseTool`, which
    is what CrewAI 0.5.x consumes internally. Either import lands on a
    pydantic-compatible base class that implements the `Tool` protocol
    expected by `Agent(tools=[...])`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

# --- Tool base class --------------------------------------------------------
# Try CrewAI's native BaseTool first; fall back to LangChain for 0.5.x.
try:  # pragma: no cover — depends on installed crewai version
    from crewai.tools import BaseTool  # type: ignore
except ImportError:  # crewai==0.5.0 path
    from langchain.tools import BaseTool  # type: ignore

from transformers import BertForSequenceClassification, BertTokenizerFast

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Path to the model folder saved by the W2 fine-tuning notebook. We
# resolve it relative to the repo root so the tool works regardless of
# the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = _REPO_ROOT / "models" / "bert_sentiment"

# Must match the order used at training time
# (see notebooks/bert_finetuning.ipynb, Section 2).
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LENGTH = 128

# Pick the best device at import time. BERT inference on CPU is fine for
# one-off reviews; GPU kicks in automatically on machines that have one.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level cache so repeated calls inside the same process do not
# reload ~440 MB of weights every time. We keep it outside the class to
# avoid clashing with pydantic's field machinery on BaseTool.
_MODEL_CACHE: dict[str, Any] = {"model": None, "tokenizer": None}


def _lazy_load() -> tuple[Any, Any]:
    """Load the fine-tuned BERT + tokenizer once per process."""
    if _MODEL_CACHE["model"] is None:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found: {MODEL_DIR}. "
                "Run notebooks/bert_finetuning.ipynb first to produce it."
            )
        tokenizer = BertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(_DEVICE)
        model.eval()
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
    return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class SentimentAnalysisTool(BaseTool):
    """CrewAI tool: classify a review's sentiment with fine-tuned BERT."""

    # Pydantic fields — BaseTool (LangChain or CrewAI) reads these to
    # advertise the tool to the agent's planner.
    name: str = "sentiment_classifier"
    description: str = (
        "Classifies the sentiment of a product review. "
        "Input: review text (string). "
        "Output: JSON with sentiment label, confidence score, and "
        "per-class probabilities."
    )

    # Declared as ClassVar so pydantic does not try to validate it as a
    # model field.
    return_direct: ClassVar[bool] = False

    def _run(self, review: str) -> str:
        """Synchronous entry point invoked by the agent.

        Parameters
        ----------
        review : str
            Raw review text.

        Returns
        -------
        str
            A JSON string. Shape on success::

                {
                    "label": "positive",
                    "confidence": 0.97,
                    "scores": {
                        "negative": 0.01,
                        "neutral":  0.02,
                        "positive": 0.97
                    }
                }

            Shape on failure::

                {"error": "<message>"}
        """
        try:
            if not isinstance(review, str) or not review.strip():
                return json.dumps({"error": "Empty or non-string review input."})

            model, tokenizer = _lazy_load()

            enc = tokenizer(
                review,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(_DEVICE)

            # No gradient tracking -> faster inference + less memory.
            with torch.no_grad():
                logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            top_id = int(np.argmax(probs))
            result = {
                "label": LABEL_NAMES[top_id],
                "confidence": float(probs[top_id]),
                # Ordered negative / neutral / positive per the contract.
                "scores": {
                    "negative": float(probs[0]),
                    "neutral": float(probs[1]),
                    "positive": float(probs[2]),
                },
            }
            return json.dumps(result)

        except Exception as exc:  # pragma: no cover — defensive catch-all
            # We never want a tool crash to bubble up and kill the Crew.
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _arun(self, review: str) -> str:
        """Async variant — BaseTool requires it, but we just delegate."""
        return self._run(review)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Three hand-crafted reviews covering each class.
    EXAMPLES = [
        "Absolutely love this moisturizer! My skin feels hydrated all day "
        "and the scent is lovely. Best beauty purchase I've made this year.",
        "Terrible experience. The bottle arrived half-empty, the product "
        "smells chemical, and it broke me out within two days.",
        "The packaging is gorgeous and the shade range is impressive, but "
        "the formula is way too drying on my skin and it creases around "
        "the eyes after a few hours.",
    ]

    tool = SentimentAnalysisTool()
    for i, review in enumerate(EXAMPLES, 1):
        print(f"\n===== Example {i} =====")
        print(f"REVIEW: {review}")
        print(f"TOOL OUTPUT: {tool._run(review)}")
