"""
tools/__init__.py
-----------------
CrewAI tools used by the orchestrator agent.

Shipped in Week 2:
    * SentimentAnalysisTool — fine-tuned BERT sentiment classifier.

Planned for Week 3:
    * ThemeExtractorTool    — keyword / topic extraction over reviews.
    * SummarizerTool        — abstractive summary of a review batch.
    * DatasetSearchTool     — retrieve similar reviews from reviews.csv.
"""

from .sentiment_tool import SentimentAnalysisTool

__all__ = ["SentimentAnalysisTool"]
