"""
agents/orchestrator.py
----------------------
Week 3 multi-agent orchestrator.

The single-agent prototype from W1/W2 is now split into three roles
running inside a single sequential Crew:

    1. SentimentAgent    — calls the BERT sentiment_classifier tool.
    2. MarketAgent       — calls the DuckDuckGo web_search tool for
                           competitive context.
    3. ReportAgent       — synthesises the two specialist outputs into
                           the final JSON contract.

A Human-In-The-Loop (HITL) checkpoint runs after the sentiment task:
the reviewer is shown the BERT classification and asked to approve
before the market/report stages fire. Rejection raises a ValueError
so the whole pipeline halts cleanly.

Every task output is appended to ``logs/run_log.json`` with a UTC
timestamp, agent name, task identifier, and the raw agent output.

Public API:
    ``analyze_review(review_text: str) -> dict``
        Unchanged from previous weeks — main.py keeps working.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_openai import ChatOpenAI

from agents.sentiment_agent import build_sentiment_agent
from agents.market_agent import build_market_agent
from tools.guardrails import validate_review

# ---------------------------------------------------------------------------
# crewai==0.5.0 workaround
# ---------------------------------------------------------------------------
# In crewai==0.5.0 a tool-less agent can short-circuit and return an
# `AgentFinish` object that the executor does not know how to unwrap,
# producing:
#     ValueError: Unexpected output type from agent: AgentFinish
# Giving the agent a no-op tool forces the executor through its tool
# loop, which handles the final output correctly. The agent never
# actually calls this tool because the prompt tells it not to.
_dummy_tool = Tool(
    name="no_op",
    func=lambda x: x,
    description="No-op tool. Do not use.",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RUN_LOG_PATH = _PROJECT_ROOT / "logs" / "run_log.json"
_RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    """ISO-8601 UTC timestamp used in every log record."""
    return datetime.now(timezone.utc).isoformat()


def _append_log(entry: dict[str, Any]) -> None:
    """Append an audit record to ``logs/run_log.json``.

    The log is stored as a JSON array so downstream tooling (pandas,
    jq) can read the whole history in one shot. We do a
    load-modify-write on each call rather than streaming newline-JSON,
    which keeps the file valid JSON at all times.
    """
    existing: list[dict[str, Any]] = []
    if _RUN_LOG_PATH.exists() and _RUN_LOG_PATH.stat().st_size > 0:
        try:
            loaded = json.loads(_RUN_LOG_PATH.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            # Corrupted log — start fresh rather than crashing the run.
            existing = []

    existing.append(entry)
    _RUN_LOG_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _log_task(agent_name: str, task_name: str, output: Any) -> None:
    """Standardised per-task log record."""
    _append_log(
        {
            "timestamp_utc": _utc_now(),
            "agent": agent_name,
            "task": task_name,
            "output": str(output),
        }
    )


# ---------------------------------------------------------------------------
# Report agent (defined here because it only exists to merge outputs)
# ---------------------------------------------------------------------------
def _build_report_agent() -> Agent:
    """Create the ReportAgent that composes the final JSON.

    Kept local to the orchestrator since it has no tools of its own —
    it is purely an LLM-driven synthesiser.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,       # synthesis must be reproducible
    )
    return Agent(
        role="Report Synthesizer",
        goal=(
            "Combine the sentiment classification and the market "
            "research summary into a single structured JSON report."
        ),
        backstory=(
            "You are the final editor. You trust the Sentiment "
            "Specialist's label (it came from a fine-tuned BERT model) "
            "and you trust the Market Specialist's competitive "
            "summary. Your only job is to merge them into the exact "
            "JSON schema the downstream pipeline expects."
        ),
        llm=llm,
        # Dummy tool attached purely to work around the crewai==0.5.0
        # AgentFinish bug — see note at top of file.
        tools=[_dummy_tool],
        allow_delegation=False,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# JSON parsing helper (unchanged from W2)
# ---------------------------------------------------------------------------
def _coerce_to_dict(raw_output: str) -> dict[str, Any]:
    """Best-effort parse of an LLM's text output into a dict.

    Strips optional ```json fences and falls back to a regex scan for
    the first balanced ``{...}`` block if direct parsing fails.
    """
    text = raw_output.strip()

    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------------------
# HITL checkpoint
# ---------------------------------------------------------------------------
def _sentiment_hitl_checkpoint(raw_output: Any) -> None:
    """Pause the pipeline and ask the human to approve the sentiment.

    Called as a Task callback right after the SentimentAgent finishes.
    Raising ValueError halts the sequential Crew before the market or
    report tasks are dispatched.
    """
    _log_task(
        agent_name="Sentiment Analysis Specialist",
        task_name="sentiment_analysis",
        output=raw_output,
    )

    print("\n--- HITL checkpoint: sentiment result ---")
    print(raw_output)
    approval = input("Approve sentiment? (y/n): ").strip().lower()
    if approval in ("n", "no"):
        raise ValueError("Sentiment rejected by human reviewer")


# ---------------------------------------------------------------------------
# Task factory
# ---------------------------------------------------------------------------
def _build_tasks(
    review_text: str,
    sentiment_agent: Agent,
    market_agent: Agent,
    report_agent: Agent,
) -> tuple[Task, Task, Task]:
    """Build the three sequential tasks for one review.

    Tasks are rebuilt per call so the ``review_text`` is interpolated
    verbatim into the description (CrewAI does not re-render templates
    at execution time).
    """

    # ---- Task 1: sentiment ------------------------------------------------
    sentiment_task = Task(
        description=(
            "Classify the sentiment of the following product review by "
            "calling the `sentiment_classifier` tool with the review "
            "text. Do NOT guess the sentiment yourself.\n\n"
            f"REVIEW:\n\"\"\"{review_text}\"\"\"\n\n"
            "Return ONLY the JSON string produced by the tool — it "
            "already contains the keys `label`, `confidence`, and "
            "`scores`."
        ),
        expected_output=(
            'A JSON string like: {"label": "positive", '
            '"confidence": 0.97, "scores": {"negative": 0.01, '
            '"neutral": 0.02, "positive": 0.97}}'
        ),
        agent=sentiment_agent,
        # After this task, pause for human approval (and log).
        callback=_sentiment_hitl_checkpoint,
    )

    # ---- Task 2: market research -----------------------------------------
    market_task = Task(
        description=(
            "Using the `web_search` tool, gather competitive market "
            "context about the product described in the review below. "
            "Formulate 1–2 focused queries (e.g. product category, "
            "distinguishing features, competitor names) and synthesise "
            "the hits into a short competitive summary (3–5 sentences).\n\n"
            f"REVIEW:\n\"\"\"{review_text}\"\"\"\n\n"
            "You may also take the sentiment from the previous task "
            "(available in your context) into account when framing the "
            "summary."
        ),
        expected_output=(
            "A concise plaintext paragraph (3–5 sentences) summarising "
            "the competitive landscape for this product."
        ),
        agent=market_agent,
        context=[sentiment_task],
        callback=lambda out: _log_task(
            "Market Research Specialist", "market_research", out
        ),
    )

    # ---- Task 3: final synthesis -----------------------------------------
    report_task = Task(
        description=(
            "You are given two upstream outputs in your context:\n"
            "  - A JSON sentiment object with `label`, `confidence`, "
            "and `scores`.\n"
            "  - A plaintext competitive summary.\n\n"
            "Read the original review yourself to extract up to 5 "
            "short key themes and a one-sentence summary.\n\n"
            f"REVIEW:\n\"\"\"{review_text}\"\"\"\n\n"
            "Return ONLY a valid JSON object (no markdown fences, no "
            "prose) with EXACTLY these keys:\n"
            '  - "sentiment": one of "positive", "neutral", "negative" '
            "(copied from the sentiment task's `label`)\n"
            '  - "confidence": float from the sentiment task\n'
            '  - "key_themes": list of up to 5 short strings\n'
            '  - "summary": one sentence summarising the review\n'
            '  - "market_context": the competitive summary from the '
            "market task\n"
        ),
        expected_output=(
            'A JSON object with keys sentiment, confidence, '
            'key_themes, summary, market_context.'
        ),
        agent=report_agent,
        context=[sentiment_task, market_task],
        callback=lambda out: _log_task(
            "Report Synthesizer", "final_report", out
        ),
    )

    return sentiment_task, market_task, report_task


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_review(review_text: str) -> dict[str, Any]:
    """Run the full 3-agent pipeline on a single review.

    Parameters
    ----------
    review_text : str
        The raw product review to analyse.

    Returns
    -------
    dict
        JSON with ``sentiment``, ``confidence``, ``key_themes``,
        ``summary``, and ``market_context``. If parsing fails, the
        dict contains ``raw`` and ``error`` keys so main.py can keep
        iterating through the batch.

    Raises
    ------
    ValueError
        When the human reviewer rejects the sentiment at the HITL
        checkpoint. Callers can catch this to skip the review or abort
        the batch.
    """
    # ---- W4 guardrail -------------------------------------------------
    # Validate and sanitise the review BEFORE spinning up any agents.
    # This prevents empty / trivial / runaway inputs from burning LLM
    # quota or reaching the BERT tool.
    ok, cleaned_or_reason = validate_review(review_text)
    if not ok:
        reason = cleaned_or_reason
        _append_log(
            {
                "timestamp_utc": _utc_now(),
                "agent": "orchestrator",
                "task": "guardrail_rejected",
                "output": {"input": review_text, "reason": reason},
            }
        )
        return {
            "error": reason,
            "sentiment": "unknown",
            "confidence": 0.0,
        }
    # From here on, work with the cleaned text.
    review_text = cleaned_or_reason

    # Fresh agents per call -> no state leaks between reviews.
    sentiment_agent = build_sentiment_agent()
    market_agent = build_market_agent()
    report_agent = _build_report_agent()

    sentiment_task, market_task, report_task = _build_tasks(
        review_text=review_text,
        sentiment_agent=sentiment_agent,
        market_agent=market_agent,
        report_agent=report_agent,
    )

    crew = Crew(
        agents=[sentiment_agent, market_agent, report_agent],
        tasks=[sentiment_task, market_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    # Mark the start of this run in the audit log so grouping by
    # timestamp is easy even if multiple reviews run back-to-back.
    _append_log(
        {
            "timestamp_utc": _utc_now(),
            "agent": "orchestrator",
            "task": "pipeline_start",
            "output": review_text,
        }
    )

    # ---- W4 robustness --------------------------------------------------
    # kickoff() runs the tasks sequentially; per-task callbacks handle
    # logging and the HITL checkpoint. Any uncaught exception (network
    # error, LLM timeout, HITL rejection, parser failure, …) is
    # converted into a graceful error dict so the caller keeps running.
    try:
        crew_output = crew.kickoff()
        raw_text = str(getattr(crew_output, "raw", crew_output))

        try:
            result = _coerce_to_dict(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {
                "sentiment": "unknown",
                "confidence": 0.0,
                "key_themes": [],
                "summary": "Failed to parse final report.",
                "market_context": "",
                "raw": raw_text,
                "error": str(exc),
            }
    except Exception as exc:  # noqa: BLE001 — intentional catch-all
        _append_log(
            {
                "timestamp_utc": _utc_now(),
                "agent": "orchestrator",
                "task": "pipeline_error",
                "output": f"{type(exc).__name__}: {exc}",
            }
        )
        return {
            "error": str(exc),
            "sentiment": "unknown",
            "confidence": 0.0,
        }

    _append_log(
        {
            "timestamp_utc": _utc_now(),
            "agent": "orchestrator",
            "task": "pipeline_end",
            "output": result,
        }
    )
    return result
