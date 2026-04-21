"""
agents/orchestrator.py
----------------------
Week 1 scope: a single CrewAI agent that analyses one product review and
returns structured JSON (sentiment, key themes, one-sentence summary).

We intentionally keep the design minimal:
    * one Agent  -> "Product Review Orchestrator"
    * one Task   -> describes what the agent must return
    * one Crew   -> runs the task sequentially

In later weeks this orchestrator will dispatch to specialist sub-agents
(sentiment, theme extractor, summarizer) using Tools. For now it does the
whole job itself via Gemini.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------
# crewai==0.5.0 does not ship a `crewai.llm` module, so we plug a
# LangChain ChatOpenAI model directly into the Agent. We target
# gpt-4o-mini — cheap, fast, and reliable for structured JSON output.
load_dotenv()

_llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------
# The triad (role / goal / backstory) shapes the system prompt CrewAI
# builds internally. Keep the goal precise: it is the contract we expect
# the agent to fulfil on every call.
_orchestrator_agent = Agent(
    role="Product Review Orchestrator",
    goal=(
        "Analyze a product review and return a structured JSON with: "
        "sentiment (positive/neutral/negative), key themes (list), and a "
        "one-sentence summary."
    ),
    backstory="You are an expert product analyst.",
    llm=_llm,
    verbose=False,       # flip to True during debugging
    allow_delegation=False,  # single-agent crew — nothing to delegate to yet
)


def _build_task(review_text: str) -> Task:
    """Create a fresh Task bound to the review we want analysed.

    We rebuild the Task per call (instead of reusing one instance) so the
    description always contains the current review verbatim — CrewAI does
    not re-render templates at execution time.
    """
    description = (
        "Read the following product review carefully and produce your "
        "analysis.\n\n"
        f"REVIEW:\n\"\"\"{review_text}\"\"\"\n\n"
        "Return ONLY a valid JSON object (no markdown fences, no prose) "
        "with EXACTLY these keys:\n"
        '  - "sentiment": one of "positive", "neutral", "negative"\n'
        '  - "key_themes": a list of short strings (max 5 items)\n'
        '  - "summary": a single sentence summarising the review\n'
    )

    expected_output = (
        'A JSON object like: {"sentiment": "positive", '
        '"key_themes": ["battery life", "build quality"], '
        '"summary": "The reviewer loved the long battery life and solid feel."}'
    )

    return Task(
        description=description,
        expected_output=expected_output,
        agent=_orchestrator_agent,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _coerce_to_dict(raw_output: str) -> dict[str, Any]:
    """Best-effort parse of the LLM's text output into a Python dict.

    Gemini is generally good at honouring the "JSON only" instruction, but
    occasionally wraps the object in ```json ... ``` fences. We strip any
    such fencing before json.loads, and fall back to a regex scan for the
    first balanced {...} block if the direct parse fails.
    """
    text = raw_output.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: grab the first {...} block we can find.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise  # re-raise the original error for the caller to handle


def analyze_review(review_text: str) -> dict[str, Any]:
    """Run the orchestrator crew on a single review.

    Parameters
    ----------
    review_text : str
        The raw product review to analyse.

    Returns
    -------
    dict
        Parsed JSON with keys ``sentiment``, ``key_themes``, ``summary``.
        If parsing fails, the dict will contain a ``raw`` key with the
        untouched LLM output and an ``error`` key describing the issue —
        this keeps main.py from crashing mid-batch.
    """
    # Guard against empty input: CrewAI would still execute, but burning
    # free-tier quota on an empty string is wasteful.
    if not review_text or not review_text.strip():
        return {
            "sentiment": "neutral",
            "key_themes": [],
            "summary": "Empty review — nothing to analyse.",
        }

    task = _build_task(review_text)

    # A Crew with a single agent and a single task running sequentially is
    # the simplest possible CrewAI topology. We recreate it per call so
    # state never leaks between reviews.
    crew = Crew(
        agents=[_orchestrator_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # kickoff() returns a CrewOutput; .raw is the final string the agent
    # produced. We coerce it into a dict before handing it back.
    crew_output = crew.kickoff()
    raw_text = str(getattr(crew_output, "raw", crew_output))

    try:
        return _coerce_to_dict(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "sentiment": "unknown",
            "key_themes": [],
            "summary": "Failed to parse model output.",
            "raw": raw_text,
            "error": str(exc),
        }
