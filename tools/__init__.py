"""
tools/__init__.py
-----------------
CrewAI tools used by the orchestrator and the specialist agents.

Shipped so far:
    * SentimentAnalysisTool — fine-tuned BERT sentiment classifier (W2).
    * WebSearchTool         — DuckDuckGo web search for market context (W3).

Planned:
    * ThemeExtractorTool    — keyword / topic extraction over reviews.
    * DatasetSearchTool     — retrieve similar reviews from reviews.csv.
"""

from .sentiment_tool import SentimentAnalysisTool
from .search_tool import WebSearchTool

__all__ = ["SentimentAnalysisTool", "WebSearchTool"]
