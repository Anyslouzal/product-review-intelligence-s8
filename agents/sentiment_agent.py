"""
agents/sentiment_agent.py
-------------------------
Specialist agent for sentiment analysis (W3).

Responsibility:
    Given a raw product review, call the BERT-backed
    ``sentiment_classifier`` tool and return its JSON verbatim. The
    agent is intentionally *not* allowed to guess sentiment on its own
    — the goal of Week 3 is to show how a CrewAI agent can delegate a
    well-defined sub-task to a deterministic tool.

Public API:
    ``build_sentiment_agent() -> Agent``
        Returns a configured CrewAI Agent ready to be wired into a Crew.
"""

from __future__ import annotations

import os

from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import SentimentAnalysisTool

# Load the .env once per process so OPENAI_API_KEY is available to the
# specialist's LLM.
load_dotenv()


def _build_llm() -> ChatOpenAI:
    """Factory for the specialist's LLM — gpt-4o-mini, deterministic."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0,   # sentiment decisions must be reproducible
    )


def build_sentiment_agent() -> Agent:
    """Return a fresh Sentiment Analysis Specialist agent.

    We instantiate the tool here (rather than at module import) so the
    BERT weights are only loaded when the agent is actually used, and
    so each Crew run can start with a clean agent instance.
    """
    return Agent(
        role="Sentiment Analysis Specialist",
        goal=(
            "Analyse product review sentiment using the "
            "sentiment_classifier tool. Always call the tool — never "
            "guess."
        ),
        backstory=(
            "You are a domain expert who trusts the fine-tuned BERT "
            "classifier over any intuition. Your job is to feed the "
            "review to the tool and report its structured output to "
            "the rest of the crew."
        ),
        llm=_build_llm(),
        tools=[SentimentAnalysisTool()],
        allow_delegation=False,
        verbose=False,
    )
