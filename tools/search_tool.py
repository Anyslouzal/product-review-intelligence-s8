"""
tools/search_tool.py
--------------------
Week 3 deliverable: a CrewAI-compatible web-search tool that the
MarketResearch specialist agent uses to pull competitive / market
context for the product under review.

We use the `duckduckgo-search` Python package so the tool works with
zero API keys — perfect for the oral-defence demo and for the course
constraint of "no personal API spending".

Version note:
    crewai==0.5.0 does not yet ship `crewai.tools.BaseTool`. We try
    that import first and fall back to LangChain's `BaseTool`, which
    is what 0.5.x consumes internally.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

try:  # pragma: no cover — depends on installed crewai version
    from crewai.tools import BaseTool  # type: ignore
except ImportError:  # crewai==0.5.0 path
    from langchain.tools import BaseTool  # type: ignore


# Number of hits we return per query. Kept small because the downstream
# LLM will re-read each result as prompt context.
MAX_RESULTS = 3


class WebSearchTool(BaseTool):
    """CrewAI tool: DuckDuckGo web search for competitive context."""

    name: str = "web_search"
    description: str = (
        "Searches the web for competitive product information and market "
        "trends. Input: search query (string). Output: JSON list of top 3 "
        "results with title, url, snippet."
    )

    # ClassVar so pydantic does not treat it as a model field.
    return_direct: ClassVar[bool] = False

    def _run(self, query: str) -> str:
        """Run a DuckDuckGo search and return the top-3 results as JSON.

        Parameters
        ----------
        query : str
            Natural-language search query (e.g. a product name or a
            theme we want competitive context for).

        Returns
        -------
        str
            JSON array of ``{"title", "url", "snippet"}`` dicts, or a
            JSON ``{"error": "..."}`` object on failure.
        """
        try:
            if not isinstance(query, str) or not query.strip():
                return json.dumps({"error": "Empty or non-string search query."})

            # Imported inside _run so a missing `duckduckgo-search`
            # package only breaks this tool, not the whole codebase.
            from duckduckgo_search import DDGS

            # `DDGS().text(...)` returns a list of dicts with the keys
            # {title, href, body}. We normalise them into our own
            # schema (href -> url, body -> snippet) so the agent prompt
            # stays stable even if the ddgs library renames fields.
            with DDGS() as ddgs:
                raw_hits: list[dict[str, Any]] = list(
                    ddgs.text(query, max_results=MAX_RESULTS) or []
                )

            results = [
                {
                    "title": hit.get("title", ""),
                    "url": hit.get("href", hit.get("url", "")),
                    "snippet": hit.get("body", hit.get("snippet", "")),
                }
                for hit in raw_hits[:MAX_RESULTS]
            ]
            return json.dumps(results, ensure_ascii=False)

        except Exception as exc:  # pragma: no cover — defensive catch-all
            # Never let a transient network hiccup kill the Crew.
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    async def _arun(self, query: str) -> str:
        """Async variant — BaseTool requires it; delegate to _run."""
        return self._run(query)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Three queries we would plausibly fire in a real market-research
    # step on beauty-category reviews.
    EXAMPLES = [
        "best drugstore moisturizer 2024 reviews",
        "niacinamide serum side effects skincare",
        "cruelty free makeup brands market trends",
    ]

    tool = WebSearchTool()
    for i, query in enumerate(EXAMPLES, 1):
        print(f"\n===== Example {i} =====")
        print(f"QUERY: {query}")
        print(f"TOOL OUTPUT: {tool._run(query)}")
