# Product Review Intelligence — Architecture (Week 1)

**Author:** Anys Louzal — ESIN IA S8
**Status:** Week 1 / 12 — foundation only
**Scope of this document:** the minimal agentic pipeline delivered in
Week 1, plus the extensions planned for Weeks 2–3 so the reader can see
where the project is heading.

---

## 1. System overview

The goal of the project is to build a **multi-agent AI system** that
takes a raw product review (free-form text) and produces a structured,
machine-readable analysis suitable for downstream analytics (sentiment
dashboards, theme trend tracking, product-quality alerts).

In Week 1 the system is intentionally reduced to a single orchestrator
agent so that every building block — configuration, LLM access, logging,
data ingestion, evaluation — can be validated end-to-end before more
agents are layered on top.

```
+----------------------+      +----------------------+
|  Raw review (text)   | ---> |   Orchestrator agent |
+----------------------+      +----------------------+
                                        |
                                        v
                           +-----------------------------+
                           | Structured JSON result      |
                           |  - sentiment                |
                           |  - key_themes[]             |
                           |  - summary                  |
                           +-----------------------------+
                                        |
                                        v
                           +-----------------------------+
                           | logs/run_log.json (audit)   |
                           +-----------------------------+
```

## 2. Agent roles and responsibilities

| Week | Agent | Role | Responsibility |
|------|-------|------|----------------|
| W1 | **Orchestrator** | Product Review Orchestrator | Reads the review and produces the full JSON analysis on its own. |
| W2 | Sentiment Specialist | Classifier | Fine-grained sentiment (positive / neutral / negative + confidence). |
| W2 | Theme Extractor | Keyphrase miner | Extracts the key topics / attributes mentioned in the review. |
| W3 | Summarizer | Writer | Produces a concise one-sentence summary. |
| W3 | Orchestrator (upgraded) | Router | Delegates to the three specialists and merges their output. |

The Week 1 agent is a *proto-orchestrator* — it does every job in one
LLM call. From Week 2 it will delegate via CrewAI's native task chaining.

## 3. Tool descriptions (planned, W2/W3)

Tools will be implemented inside `tools/` as CrewAI `BaseTool`
subclasses. None are wired up in Week 1.

| Tool | Purpose | Consumer agent |
|------|---------|----------------|
| `SentimentTool` | Deterministic sentiment scoring (e.g. VADER) as a cross-check on the LLM. | Sentiment Specialist |
| `ThemeExtractorTool` | TF-IDF / KeyBERT-based phrase extraction. | Theme Extractor |
| `DatasetSearchTool` | Retrieve the N most similar reviews from `reviews.csv` for context. | Orchestrator |
| `SummarizerTool` | Optional extractive fallback if the LLM output is too long. | Summarizer |

## 4. Data flow (ASCII)

```
            +-------------------------+
            | HuggingFace datasets    |
            | McAuley-Lab/            |
            | Amazon-Reviews-2023     |
            +-----------+-------------+
                        | (streaming, 5 000 rows)
                        v
            +-------------------------+
            | data/download_dataset.py|
            +-----------+-------------+
                        |
                        v
            +-------------------------+
            | data/reviews.csv        | <-- notebooks/eda.ipynb reads here
            +-------------------------+

             user-provided review text
                        |
                        v
            +-------------------------+        +--------------------+
            | main.py                 |------->| agents/            |
            | (examples + logging)    |        | orchestrator.py    |
            +-----------+-------------+        +----------+---------+
                        |                                 |
                        |                                 v
                        |                       +---------------------+
                        |                       | CrewAI LLM          |
                        |                       |  -> Gemini 2.0 Flash|
                        |                       +----------+----------+
                        |                                  |
                        v                                  v
            +-------------------------+        +---------------------+
            | logs/run_log.json       |<-------| Structured JSON     |
            | (timestamped entries)   |        | (sentiment/themes/  |
            +-------------------------+        |  summary)           |
                                               +---------------------+
```

## 5. Tech stack justification

- **CrewAI** — the course explicitly targets multi-agent orchestration.
  CrewAI gives us a minimal, well-documented agent/task/crew abstraction
  without pulling in the full LangGraph runtime, and has native LLM
  integration so we do not need LangChain on the hot path.
- **Gemini 2.0 Flash (free tier)** — no personal API spending is allowed
  for this project. Gemini Flash is fast, has a generous free quota,
  and handles JSON-mode instructions reliably.
- **HuggingFace `datasets`** — canonical way to pull `McAuley-Lab/
  Amazon-Reviews-2023`; streaming mode lets us grab exactly the first
  5 000 rows without downloading the full corpus.
- **pandas + matplotlib** — zero-friction EDA in the notebook; no
  additional environment setup at defence time.
- **python-dotenv** — keeps the API key in `.env` (gitignored) and out
  of source control, which is both a security and a reproducibility win.

## 6. Known limitations (Week 1)

1. **Single point of failure.** The orchestrator does sentiment, theme
   extraction, and summarization in one LLM call. Errors in any sub-task
   contaminate the others. W2 splits them into specialists.
2. **No deterministic evaluation yet.** We have qualitative examples in
   `main.py` but no labelled test set. A small hand-labelled evaluation
   slice is planned for W2.
3. **Gemini free-tier rate limits.** Long batch runs will be throttled;
   anything larger than a handful of reviews should go through the W3
   caching layer (not yet implemented).
4. **JSON parsing is best-effort.** If Gemini ignores the "JSON only"
   instruction, the parser falls back to regex extraction. A stricter
   schema validation step (pydantic) will be added with the specialist
   agents.
5. **No tool use yet.** `tools/` is a stub. The CrewAI Tools surface is
   where most of the engineering work for W2/W3 lives.
6. **English-only assumption.** Stopwords list and prompt are English;
   non-English reviews in the dataset are processed but may yield lower
   quality themes.
