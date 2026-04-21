"""
agents/market_agent.py
----------------------
Specialist agent for market / competitive research (W3).

Responsibility:
    Given a product review (and, via task context, the sentiment
    produced by the SentimentAgent), formulate a DuckDuckGo query,
    call the ``web_search`` tool, and return a short competitive
    summary the orchestrator can merge into the final report.

Public API:
    ``build_market_agent() -> Agent``
"""

from __future__ import annotations

import os

from crewai import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import WebSearchTool

load_dotenv()


def _build_llm() -> ChatOpenAI:
    """gpt-4o-mini with a touch of temperature for more varied queries."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,
    )


def build_market_agent() -> Agent:
    """Return a fresh Market Research Specialist agent."""
    return Agent(
        role="Market Research Specialist",
        goal=(
            "Search for competitive market information about the product "
            "mentioned in the review using the web_search tool. Return "
            "a brief competitive summary."
        ),
        backstory=(
            "You are a market analyst. You read the review, identify "
            "the product category and distinguishing features, then "
            "query the web to understand how competitors are positioned "
            "and what customers are saying about similar products."
        ),
        llm=_build_llm(),
        tools=[WebSearchTool()],
        allow_delegation=False,
        verbose=False,
    )
