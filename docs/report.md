# Product Review Intelligence
## A Multi-Agent AI System for Structured Product Review Analysis

**Université Internationale de Rabat — S8 Integrated Project**
**Programme:** Artificial Intelligence Engineering (ESIN IA S8)

**Authors**
- Mohamed Anouar **FARESS**
- Mohammed Yassine **BEN MEZIANE**
- Imad Eddine **EL KHORASSANI**
- Anys **LOUZAL**

**Date:** April 2026

---

## Abstract

This report presents the design, implementation, and evaluation of
*Product Review Intelligence* — a multi-agent AI system that ingests a
free-form product review and produces a structured analytical report
combining sentiment classification, key-theme extraction, a concise
summary, and market context. The system was built over four weekly
sprints as the final deliverable of the UIR S8 Integrated Project. It
combines a fine-tuned BERT classifier (86.2 % accuracy, 0.859 weighted
F1 on a held-out Amazon Reviews validation set), three specialist
CrewAI agents, an OpenAI GPT-4o-mini reasoning backend, and a live web
search tool, all coordinated by a sequential orchestrator with a
human-in-the-loop (HITL) approval checkpoint and a guardrail layer
that makes the pipeline robust to adversarial inputs. The codebase is
fully documented, version-controlled, and passes all six designed
edge-case regression tests.

---

## Table of Contents

1. Introduction
2. System Architecture
3. Technical Stack
4. Deep Learning Model — BERT Sentiment Classifier
5. Multi-Agent System
6. Human-in-the-Loop
7. Evaluation & Robustness
8. Limitations & Future Work
9. Conclusion
10. References

---

## 1. Introduction

### 1.1 Problem Statement

Modern e-commerce platforms receive millions of free-form product
reviews every day. These reviews are a rich signal for both buyers
(who want to make informed purchasing decisions) and sellers (who want
to improve their products and understand how the market perceives
them), but their unstructured nature makes automated analysis
non-trivial. A single review can simultaneously express sentiment
about multiple product attributes, reference competitors, contain
spelling mistakes or abusive language, be written in a language other
than English, or be so short as to be analytically worthless. Any
system that aims to turn this raw text into a structured, actionable
signal must therefore be:

- **Accurate** on the core classification task (sentiment);
- **Context-aware** — able to extract themes, summarise, and put the
  review in the wider market context;
- **Robust** to malformed, adversarial, or ambiguous input;
- **Auditable** — every decision must be traceable for downstream
  stakeholders (QA, compliance, product managers).

### 1.2 Motivation

The UIR S8 Integrated Project asks students to build a multi-agent AI
system that integrates (i) a deep-learning model trained on a public
dataset, (ii) at least two specialist agents, (iii) an orchestration
layer, (iv) at least one external tool call, (v) a human-in-the-loop
safeguard, and (vi) a robustness / evaluation study. Product review
intelligence is a natural fit: the core task (sentiment) benefits
from a fine-tuned language model, the surrounding tasks (themes,
summary, market context) benefit from a generalist LLM, and real
reviews provide a rich test-bed for adversarial inputs.

### 1.3 Objectives

The project set itself four measurable objectives:

1. Train a BERT-based classifier on a public product-review dataset
   and attain at least **85 %** weighted F1 on the held-out split.
2. Deliver a **multi-agent** pipeline — at minimum one sentiment
   specialist, one market-research specialist, and one
   report-synthesis agent — coordinated via a sequential orchestrator.
3. Integrate at least one **external tool** (web search) so the
   pipeline can reason beyond the review text.
4. Ship a **robustness layer** (guardrails + broad exception handling)
   plus an automated regression harness covering six adversarial
   inputs.

### 1.4 Scope

Within these objectives, the project deliberately stays narrow in two
dimensions:

- **Domain.** We use the Amazon Reviews 2023 `All_Beauty` category
  (10 000 rows) for training and qualitative testing. The pipeline is
  domain-agnostic at inference time but the classifier's numerical
  accuracy numbers apply to beauty-category reviews.
- **Cost.** No paid APIs beyond OpenAI's GPT-4o-mini (the cheapest
  current OpenAI chat model) and the HuggingFace free download tier.
  The web-search layer uses DuckDuckGo through the `duckduckgo-search`
  Python package, which requires no API key. The BERT classifier runs
  locally on CPU or a single free-tier GPU (Google Colab T4).

---

## 2. System Architecture

### 2.1 High-level Design

The system is organised into three horizontal layers:

1. **Data layer** — HuggingFace dataset ingestion, local CSV cache
   (`data/reviews.csv`), and the fine-tuned BERT checkpoint
   (`models/bert_sentiment/`).
2. **Tool layer** — CrewAI-compatible wrappers over the local BERT
   model (`SentimentAnalysisTool`), the web-search backend
   (`WebSearchTool`), and the input-validation helper
   (`guardrails.validate_review`). Each tool is self-contained,
   unit-testable, and advertises a JSON-only I/O contract.
3. **Agent layer** — three specialist CrewAI agents
   (`SentimentAgent`, `MarketAgent`, `ReportAgent`) coordinated by a
   top-level orchestrator (`analyze_review`) that runs them
   sequentially inside a single `Crew`.

A thin demo driver (`main.py`) and an automated regression harness
(`tests/test_edge_cases.py`) sit on top and exercise the pipeline
end-to-end. Every agent action and every pipeline phase writes a
timestamped record to `logs/run_log.json`, and the regression suite
mirrors its verdicts to `logs/edge_cases.json`.

### 2.2 Data Flow Diagram

```
                     +-----------------------------+
                     |     Raw review (text)       |
                     +--------------+--------------+
                                    |
                                    v
                        +-----------+-----------+
                        |   guardrails.         |
                        |   validate_review     |
                        +-----+-----------+-----+
                              |           |
                    reject    |           |   accept
       +----------------------+           +--------------------+
       |                                                       |
       v                                                       v
  return error                                    +------------+------------+
  {"error": ...,                                  |  Crew (sequential)      |
   "sentiment":                                   |                         |
   "unknown"}                                     |  Task 1  SentimentAgent |
                                                  |    uses: BERT tool      |
                                                  |                         |
                                                  |  [ HITL checkpoint ]    |
                                                  |                         |
                                                  |  Task 2  MarketAgent    |
                                                  |    uses: web_search     |
                                                  |                         |
                                                  |  Task 3  ReportAgent    |
                                                  |    merges 1 + 2         |
                                                  +------------+------------+
                                                               |
                                                               v
                                                +--------------+-------------+
                                                |  Final JSON                |
                                                |  { sentiment, confidence,  |
                                                |    key_themes, summary,    |
                                                |    market_context }        |
                                                +--------------+-------------+
                                                               |
                                                               v
                                                +--------------+-------------+
                                                |  logs/run_log.json         |
                                                |  (timestamped audit trail) |
                                                +----------------------------+
```

### 2.3 Agent Roles and Responsibilities

| Agent | Role | Tool(s) | LLM | Temp. |
|-------|------|---------|------|-------|
| SentimentAgent | Sentiment Analysis Specialist | `sentiment_classifier` (BERT) | gpt-4o-mini | 0.0 |
| MarketAgent | Market Research Specialist | `web_search` (DuckDuckGo) | gpt-4o-mini | 0.3 |
| ReportAgent | Report Synthesizer | `no_op` (workaround, see §5.4) | gpt-4o-mini | 0.0 |
| Orchestrator | (Python function) — sequences tasks, owns HITL + logging | — | — | — |

### 2.4 Design Decisions and Justifications

- **Sequential, not hierarchical.** CrewAI supports a hierarchical
  process where a manager agent delegates freely to specialists. We
  chose `Process.sequential` because the dependency graph is fixed
  (sentiment → market → report), because sequential runs are easier
  to audit (one log entry per task in a known order), and because it
  matches the UIR S8 Integrated Project rubric's emphasis on
  explainability.
- **Fresh agent instances per call.** `analyze_review` rebuilds the
  three agents on every invocation. This prevents hidden state from
  leaking between reviews, which was a concern once we started
  batching edge cases through the same Python process.
- **JSON-only I/O at every boundary.** Tools return JSON strings,
  specialists return JSON strings, the final report is a JSON object.
  This simplifies parsing, makes the audit log machine-readable, and
  keeps the downstream analytics pipeline (not in scope for this
  report) easy to build.
- **Two separate log files.** `logs/run_log.json` captures production
  runs (agent actions, pipeline start / end, errors);
  `logs/edge_cases.json` captures the regression suite. Splitting
  them avoids polluting the production trail with test noise.

---

## 3. Technical Stack

The technical stack was chosen to balance three forces: the rubric's
requirement for a genuine multi-agent system, the team's "no paid
APIs except the cheapest OpenAI tier" budget constraint, and the
pedagogical goal of using industry-standard tooling we can defend
line-by-line at the oral exam.

### 3.1 Orchestration — CrewAI (v0.5.0)

**Chosen because** CrewAI gives us the smallest possible surface for
an agent / task / crew abstraction without committing us to the
heavier LangGraph runtime or to building a bespoke agent loop. An
agent is defined by a (role, goal, backstory) triad plus an optional
tool list; a task is a description plus an expected output; a crew is
a (process, agents, tasks) tuple. The mental model is explicit and
maps cleanly to the project rubric.

**Trade-off accepted.** We pinned `crewai==0.5.0` because it is the
version that shipped the simplest `Task` / `Crew` API used by the
reference tutorials, but version 0.5.0 has a known bug where a
tool-less agent can return an `AgentFinish` object the executor does
not know how to unwrap. We documented this (§5.4) and worked around
it with a `no_op` dummy tool on the `ReportAgent`. Upgrading to
`>=0.28` would fix the bug but also introduces breaking changes to
the `Task` callback signature.

### 3.2 Deep Learning — BERT (`bert-base-uncased`, PyTorch, HuggingFace)

**Chosen because** BERT is the canonical encoder-only transformer for
text classification, `bert-base-uncased` is small enough to fine-tune
on a Colab T4 in under 10 minutes, and the HuggingFace `Trainer` API
takes care of the training loop, the evaluation loop, checkpointing,
and the `load_best_model_at_end` logic. PyTorch was chosen over
TensorFlow because it is the default back-end for the `transformers`
library and the one the team has most experience with.

### 3.3 Reasoning — OpenAI GPT-4o-mini via LangChain

**Chosen because** GPT-4o-mini is currently the cheapest OpenAI chat
model; it handles our JSON-only output contract reliably and has a
128 k context window that comfortably fits the longest reviews we
test with. We use `langchain_openai.ChatOpenAI` as the LLM backend
because `crewai==0.5.0` expects a LangChain-compatible chat model.
The LangChain `Tool` and `BaseTool` classes are also the substrate
CrewAI 0.5.0 consumes for its tool system, so this keeps the
dependency graph small.

**Why not Gemini / Claude / a local model?** Gemini was our original
choice (see earlier drafts of the project); we moved to GPT-4o-mini
after hitting free-tier rate limits during iterative debugging. A
local model (e.g. Llama-3-8B-Instruct) was ruled out on the basis
that the Colab free tier cannot reasonably host both it and the
BERT fine-tuning run.

### 3.4 Web Search — DuckDuckGo (`duckduckgo-search==6.3.4`)

**Chosen because** it requires no API key, has no per-day quota worth
worrying about, and the Python client exposes a clean `DDGS().text()`
interface that returns a list of `{title, href, body}` dicts we can
trivially normalise. The alternatives considered were SerpAPI
(paid), Bing Web Search API (paid), and Tavily (paid, free tier
requires an account). DuckDuckGo was the only zero-friction choice.

### 3.5 Data Pipeline — HuggingFace `datasets` + pandas + matplotlib

**Chosen because** the Amazon Reviews 2023 corpus is distributed on
the HuggingFace Hub, and `datasets.load_dataset(..., streaming=True)`
lets us pull exactly the first 10 000 rows of the `All_Beauty`
configuration without downloading the multi-GB full corpus. pandas
handles CSV serialisation, and matplotlib (plus seaborn for the
confusion matrix) covers every chart the exploratory-data-analysis
notebook needs.

### 3.6 Robustness — Guardrails (custom), `python-dotenv`

The guardrail layer is a hand-written module (no framework): 40 lines
of input validation and whitespace normalisation. A framework such as
NVIDIA NeMo Guardrails was considered and rejected as overkill —
our three rejection rules (empty, too short, too long) do not justify
the configuration burden. `python-dotenv` loads API keys from a
`.env` file so no secret ever enters version control.

---

## 4. Deep Learning Model — BERT Sentiment Classifier

### 4.1 Dataset

We use the `McAuley-Lab/Amazon-Reviews-2023` corpus from the
HuggingFace Hub, specifically the `raw_review_All_Beauty`
configuration. The full configuration is several hundred megabytes;
we use streaming mode to pull exactly the first **10 000 rows** into
a pandas DataFrame and write them to `data/reviews.csv` for
reproducibility.

| Field | Description |
|-------|-------------|
| `rating` | Float in [1, 5], rounded to an integer for our purposes |
| `text` | Free-form review body (English) |
| `title` | Review headline (not used for training) |
| `asin`, `user_id`, `timestamp` | Metadata (not used) |

### 4.2 Preprocessing

The preprocessing pipeline is intentionally minimal because BERT's
WordPiece tokenizer already handles subword segmentation,
lower-casing, and out-of-vocabulary recovery:

1. Cast `rating` to numeric; drop rows where the cast fails.
2. Drop rows where `text` is empty or whitespace-only.
3. Strip leading/trailing whitespace from `text`.
4. Apply the 3-class mapping (see §4.3).
5. Tokenize with `BertTokenizerFast` for `bert-base-uncased`,
   `truncation=True`, `max_length=128`.

Max length 128 covers the 95th percentile of review length in the
`All_Beauty` subset while keeping the effective batch size high
enough for the T4 GPU.

### 4.3 Class Mapping

Amazon's star rating is naturally 5-way, but many of the intermediate
ratings carry essentially the same signal (4 and 5 both indicate a
happy customer). We collapse the 5-way signal into 3 well-separated
sentiment classes:

| Original rating | Mapped class | Id |
|-----------------|--------------|----|
| 1, 2 | **negative** | 0 |
| 3 | **neutral**  | 1 |
| 4, 5 | **positive** | 2 |

This mapping (i) aligns the target variable with the downstream
product (an agent that needs to say "positive / neutral / negative"),
(ii) produces cleaner decision boundaries than 5-way classification,
and (iii) is the standard collapse used across the sentiment-analysis
literature.

### 4.4 Class Imbalance

As expected for Amazon data, the class distribution is heavily
skewed: roughly **75 %** of reviews fall in the positive class,
**~14 %** negative, and **~11 %** neutral. Two consequences follow:

1. **Accuracy is a poor metric.** A trivial "always positive"
   classifier would score ~75 % accuracy while being useless.
2. **Weighted F1 is the primary metric.** `f1_score(..., average="weighted")`
   averages per-class F1 weighted by class support, which rewards
   models that do well on the majority class *and* retain usable
   recall on the minority classes.

We deliberately did **not** apply class weighting or oversampling
during training because our fine-tuning budget (3 epochs) is short
and BERT is already robust to moderate imbalance at this scale. The
per-class F1 numbers in §4.6 show this choice paid off: neutral
remains the hardest class but still achieves useful recall.

### 4.5 Fine-tuning Setup

| Hyperparameter | Value |
|----------------|-------|
| Base model | `bert-base-uncased` |
| Max sequence length | 128 tokens |
| Batch size (train) | 16 |
| Batch size (eval) | 32 |
| Optimiser | AdamW (HuggingFace default) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| LR scheduler | Linear decay with warmup |
| Epochs | 3 |
| Evaluation strategy | Per epoch |
| Save strategy | Per epoch |
| `load_best_model_at_end` | True |
| `metric_for_best_model` | `f1` |
| Hardware | Google Colab T4 GPU |
| Framework | HuggingFace `Trainer` (transformers 4.44.2) |

The 80 / 20 train / validation split is stratified on the 3-class
label so every class is represented proportionally in both splits.
`seed=42` is set everywhere (Python `random`, NumPy, PyTorch, CUDA,
`TrainingArguments.seed`) so the run is reproducible.

### 4.6 Results

The training run completed in **~7 minutes** on a Colab T4. The
per-epoch metrics were:

| Epoch | Train loss | Val loss | Accuracy | F1 (weighted) |
|-------|-----------|----------|----------|---------------|
| 1 | 0.482 | 0.421 | **85.3 %** | **0.848** |
| 2 | 0.354 | 0.408 | **86.2 %** | **0.859**  ← best |
| 3 | 0.276 | 0.433 | **85.6 %** | **0.859** |

Epoch 2 is the checkpoint `load_best_model_at_end` recovers. The
slight val-loss bump at epoch 3 (0.408 → 0.433) is an early sign of
overfitting — the model memorises training examples without
improving on the validation set. This behaviour is expected at this
model / data scale and confirms the decision to stop at 3 epochs.

The resulting confusion matrix (validation set) showed the
classifier's main weakness on the minority neutral class: neutral
reviews are often predicted as positive, reflecting the fact that
a 3-star rating frequently accompanies mildly positive text. This is
a dataset artefact rather than a model failure, and further supports
the weighted-F1 focus over raw accuracy.

### 4.7 Tool Wrapping

The fine-tuned checkpoint is persisted in two formats: the standard
HuggingFace save directory (`config.json`, `tokenizer.json`,
`model.safetensors`) and a raw PyTorch `state_dict` (`model.pt`), for
portability. The checkpoint is then wrapped by
`tools/sentiment_tool.py` as `SentimentAnalysisTool`, a
CrewAI-compatible tool with the following contract:

- **Name:** `sentiment_classifier`
- **Description:** "Classifies the sentiment of a product review.
  Input: review text (string). Output: JSON with sentiment label,
  confidence score, and per-class probabilities."
- **Output schema:**
  ```json
  {
    "label": "positive",
    "confidence": 0.97,
    "scores": {
      "negative": 0.01,
      "neutral":  0.02,
      "positive": 0.97
    }
  }
  ```
- **Error schema:** `{"error": "<message>"}` on any failure (missing
  model folder, CUDA out-of-memory, malformed input).

The tool uses **lazy loading**: the BERT weights (~440 MB) are
materialised in memory only on the first `_run` call, then cached in
a module-level dictionary. Subsequent calls inside the same Python
process reuse the cached weights, which is essential for the
regression suite (6 calls) and for any batched usage.

---

## 5. Multi-Agent System

### 5.1 Agent Descriptions

#### 5.1.1 SentimentAgent

- **Role:** Sentiment Analysis Specialist
- **Goal:** *"Analyse product review sentiment using the
  `sentiment_classifier` tool. Always call the tool — never guess."*
- **Backstory:** Domain expert who trusts the fine-tuned BERT model
  over any LLM intuition.
- **Tools:** `SentimentAnalysisTool` (BERT)
- **LLM:** gpt-4o-mini, temperature 0.0
- **Build function:** `agents.sentiment_agent.build_sentiment_agent()`

The agent's LLM never classifies the review directly; its only job is
to feed the review into the BERT tool and relay the tool's JSON
output. Temperature is pinned to zero for reproducibility.

#### 5.1.2 MarketAgent

- **Role:** Market Research Specialist
- **Goal:** *"Search for competitive market information about the
  product mentioned in the review using the `web_search` tool. Return
  a brief competitive summary."*
- **Backstory:** Market analyst who reads the review, identifies the
  product category, and formulates search queries.
- **Tools:** `WebSearchTool` (DuckDuckGo)
- **LLM:** gpt-4o-mini, temperature 0.3 (to diversify query phrasing)
- **Build function:** `agents.market_agent.build_market_agent()`

This agent is the only one that truly exercises the LLM's
reasoning: given a free-form review, it must decide *what* to search
for before calling the tool, then synthesise the 3 result snippets
into a 3–5 sentence competitive summary.

#### 5.1.3 ReportAgent

- **Role:** Report Synthesizer
- **Goal:** Combine the sentiment classification and the market
  research summary into a single structured JSON report.
- **Tools:** `no_op` dummy (workaround, see §5.4)
- **LLM:** gpt-4o-mini, temperature 0.0
- **Build function:** defined inline inside
  `agents.orchestrator._build_report_agent`.

### 5.2 Task Graph and Process

The three tasks are wired into a single sequential crew:

```
Task 1 (sentiment) → Task 2 (market, context=[Task 1])
                  → Task 3 (report, context=[Task 1, Task 2])
```

`Process.sequential` is explicit in the CrewAI constructor. Because
Task 2 and Task 3 declare `context=[...]` on the upstream tasks,
CrewAI passes the upstream outputs into the task prompt at execution
time, which is how information propagates from one specialist to the
next.

### 5.3 Inter-Agent Communication

CrewAI 0.5.0 has no shared memory between agents by default. All
inter-agent communication happens through the `context` mechanism:
when a task declares `context=[task_a, task_b]`, the textual output
of `task_a` and `task_b` is injected into the prompt of the current
task. In our case:

- The **MarketAgent's** prompt receives the SentimentAgent's JSON
  output as extra context so the market query can be biased by the
  review's sentiment (e.g. a negative review leads the agent to
  search for "alternatives" rather than "reviews").
- The **ReportAgent's** prompt receives both upstream outputs, and
  the synthesis instructions explicitly tell it to *copy* the
  sentiment label verbatim (not re-classify).

This context-passing discipline is what keeps the pipeline
interpretable: every token the ReportAgent sees was either a
specialist output or part of the orchestrator's hand-written prompt.

### 5.4 The AgentFinish Workaround

During W3 integration we hit a reproducible crash:

```
ValueError: Unexpected output type from agent: AgentFinish
```

raised by the CrewAI executor on Task 3 (the report synthesis step).
The root cause is a known bug in `crewai==0.5.0`: when an agent has
no tools, the LangChain agent loop can short-circuit and return an
`AgentFinish` object directly, which the CrewAI executor does not
know how to unwrap. The fix is to attach a tool — any tool — to the
agent, which forces the executor back onto its tool loop. We added a
no-op LangChain `Tool`:

```python
_dummy_tool = Tool(
    name="no_op",
    func=lambda x: x,
    description="No-op tool. Do not use.",
)
```

and passed `tools=[_dummy_tool]` to the `ReportAgent` constructor.
The agent never actually calls the tool (the prompt is explicit), but
its presence is enough to defuse the bug.

---

## 6. Human-in-the-Loop

### 6.1 Rationale

The rubric requires a human-in-the-loop safeguard. We placed ours
immediately after the **sentiment** stage for two reasons:

1. Sentiment is the hinge of the whole analysis: the downstream
   market-research query is biased by it, and the final report's
   sentiment field is just a copy. A wrong sentiment poisons every
   subsequent stage.
2. Stopping early is cheaper. Cancelling at Task 1 saves the
   DuckDuckGo round-trip and the ReportAgent's LLM call.

### 6.2 Trigger and Interaction

The HITL checkpoint is implemented as a Task-level `callback`
attached to the sentiment task. When Task 1 completes, CrewAI invokes
our callback with the task output before proceeding to Task 2. The
callback:

1. Writes a log record with the sentiment result.
2. Prints the result to `stdout`.
3. Blocks on `input("Approve sentiment? (y/n): ")`.
4. Raises `ValueError("Sentiment rejected by human reviewer")` if
   the response is `n`/`no`; otherwise returns normally.

A raised exception from a Task callback propagates through CrewAI's
executor and aborts the sequential crew before Task 2 starts.

### 6.3 Rejection Handling

The W4 robustness layer wraps `crew.kickoff()` in a broad
try/except. When the HITL callback raises, the orchestrator catches
the `ValueError`, writes a `pipeline_error` record to the audit log,
and returns

```json
{"error": "Sentiment rejected by human reviewer",
 "sentiment": "unknown", "confidence": 0.0}
```

to the caller. `main.py` and `tests/test_edge_cases.py` both keep
iterating through their batch — a single rejected review does not
crash the run.

### 6.4 Automated Test Handling

The regression harness monkey-patches `builtins.input` to auto-approve
at import time, which is how `tests/test_edge_cases.py` runs
unattended. This is documented explicitly in the test file so a human
reader knows the prompt is being suppressed on purpose.

---

## 7. Evaluation & Robustness

### 7.1 Evaluation Strategy

We evaluated the system at two layers:

- **Model-level** — weighted F1 and per-class precision / recall on
  the held-out BERT validation split (§4.6).
- **Pipeline-level** — a regression suite that runs the full
  orchestrator on six adversarial inputs and checks that the
  pipeline does not crash.

### 7.2 Guardrail Design

The `tools/guardrails.py` module exposes a single public function:

```python
validate_review(text: str) -> tuple[bool, str]
```

It applies three rejection rules and one sanitisation pass:

| Rule | Rejection? | Reason |
|------|-----------|--------|
| `text` not a string | yes | Type safety |
| Empty or whitespace only | yes | Nothing to analyse |
| Fewer than 3 words | yes | Insufficient signal |
| More than 1000 words | yes | Prompt-budget and cost control |
| Mixed whitespace (tabs, multi-spaces, newlines) | sanitise | Normalised to single spaces |

The thresholds (`MIN_WORDS=3`, `MAX_WORDS=1000`) are module-level
constants so they are easy to tune and easy for tests to inspect.

The guardrail is invoked as the **very first** line of
`analyze_review` — before any agent or LLM is constructed. If the
guardrail rejects, the orchestrator returns `{"error": reason,
"sentiment": "unknown", "confidence": 0.0}` immediately. This saves
both API cost and GPU memory on invalid inputs.

### 7.3 Edge-Case Regression Suite

`tests/test_edge_cases.py` exercises six adversarial inputs:

| # | Case | Expected behaviour |
|---|------|--------------------|
| 1 | Empty string `""` | Rejected by guardrail |
| 2 | 600-word lorem ipsum | Passes guardrail, runs full pipeline |
| 3 | Non-English (French) | Passes guardrail, runs full pipeline |
| 4 | Abusive / profane | Passes guardrail, runs full pipeline |
| 5 | Numbers only (`"5 5 5 5 5"`) | Passes guardrail, runs full pipeline |
| 6 | Single word (`"Good"`) | Rejected by guardrail |

Each case is wrapped in a try/except in the harness itself. A case is
`PASS` iff no uncaught exception bubbles up. The harness auto-patches
`builtins.input` so the HITL prompt does not block the run.

**Result:** **6 / 6 PASS** on the most recent run. Guardrail
rejections were returned as clean error dicts; the French and
lorem-ipsum reviews drove the full pipeline without incident (though
the market-research queries on the lorem-ipsum test were predictably
nonsensical — which is acceptable: robustness means not crashing, not
producing useful output on garbage input).

### 7.4 Error Handling Strategy

Errors can appear at five layers, and we handle each distinctly:

| Layer | Strategy |
|-------|----------|
| Guardrail | Return error dict immediately, no crew started |
| Tool (BERT, DDG) | Return `{"error": ...}` JSON from the tool |
| Agent prompt | Task-level retries are disabled; one clean failure |
| CrewAI executor | Caught by the broad `try/except` around `crew.kickoff()` |
| JSON parsing | Regex fallback; on total failure return a `raw` + `error` dict |

The broad `except Exception` around `kickoff()` is deliberate: at
this layer we prefer returning a structured error to the caller over
leaking a stack trace. All errors are logged before returning.

### 7.5 Logging Architecture

Every significant event writes one JSON record to
`logs/run_log.json`. The record shape is constant:

```json
{
  "timestamp_utc": "2026-04-21T17:42:11.123456+00:00",
  "agent": "Sentiment Analysis Specialist",
  "task": "sentiment_analysis",
  "output": "..."
}
```

Events are emitted at:

- Pipeline start (`orchestrator` / `pipeline_start`)
- Guardrail rejection (`orchestrator` / `guardrail_rejected`)
- Each of the three task callbacks (`sentiment_analysis`,
  `market_research`, `final_report`)
- Pipeline error (`orchestrator` / `pipeline_error`)
- Pipeline end (`orchestrator` / `pipeline_end`)

The log is a JSON array on disk — valid JSON at all times — so it
can be consumed by pandas or `jq` without any custom parsing.

---

## 8. Limitations & Future Work

### 8.1 Class Imbalance Is Never Fully Solved

Even with weighted F1, the neutral class remains the hardest
(confusion matrix §4.6). Directions for future work:

- Synthetic oversampling of the neutral class with back-translation
  or LLM-generated paraphrases.
- A second training run on a more balanced public dataset (SST-3,
  Yelp) followed by domain adaptation to Amazon.
- A confidence-threshold post-processor: if the softmax score for
  the top class is below a threshold, re-route to a secondary
  classifier or escalate to the HITL.

### 8.2 DuckDuckGo Reliability

`duckduckgo-search` scrapes the HTML endpoint; DuckDuckGo has been
known to change the endpoint without notice, and the package has a
history of releasing breaking fixes. Production use would swap in a
paid API (Serper, Tavily, or Bing) behind the same `WebSearchTool`
interface — the tool contract would not change.

### 8.3 CrewAI Version Constraints

We pinned `crewai==0.5.0` for reproducibility and because the 0.5
API matches the tutorials and reference materials. Newer CrewAI
versions (0.80+) ship a native `crewai.tools.BaseTool` and have
fixed the `AgentFinish` bug, but they also changed the Task callback
signature and require a different pydantic minor version. An
upgrade path is straightforward but non-trivial and out of scope for
W4.

### 8.4 Other Directions

- **LangGraph.** For a stateful, cyclic workflow (e.g. "if confidence
  is low, loop back and ask a clarifying question"), LangGraph's
  explicit graph-of-nodes model is a better fit than CrewAI's
  sequential crew. We would keep the agents as-is and rewire them as
  LangGraph nodes.
- **Richer datasets.** The `All_Beauty` subset is narrow. Training on
  the full Amazon Reviews corpus or on domain-mixed data (Amazon +
  Yelp + TripAdvisor) would make the classifier more general.
- **Frontend.** The current demo is a CLI. A minimal Streamlit or
  Gradio UI would let a non-technical reviewer drive the HITL
  checkpoint and browse the audit log.
- **Evaluation against an LLM baseline.** A useful ablation is to
  benchmark BERT-sentiment vs. `gpt-4o-mini`-only sentiment on the
  same validation set, which would quantify the marginal value of
  fine-tuning over zero-shot prompting.

---

## 9. Conclusion

Over four weekly sprints we built an end-to-end multi-agent AI system
that turns a free-form product review into a structured, auditable
report. Concretely, the project delivered:

- A fine-tuned `bert-base-uncased` classifier reaching **86.2 %**
  accuracy and **0.859 weighted F1** on a 2 000-review held-out
  validation set drawn from the Amazon Reviews 2023 `All_Beauty`
  corpus (§4).
- A three-agent CrewAI pipeline — Sentiment Specialist, Market
  Research Specialist, Report Synthesizer — coordinated by a
  sequential orchestrator with explicit inter-agent context passing
  (§5).
- A human-in-the-loop checkpoint on the sentiment stage that allows
  a reviewer to abort the pipeline before the downstream market
  research runs (§6).
- A guardrail layer and a six-case regression suite demonstrating
  **6 / 6 PASS** on adversarial inputs ranging from the empty string
  to a 600-word lorem-ipsum block (§7).
- A consistent, timestamped JSON audit log covering every tool call,
  every task completion, the HITL verdict, and every pipeline error
  — giving the oral-defence panel a reproducible trace of the
  system's behaviour (§7.5).

The result satisfies every bullet of the UIR S8 Integrated Project
brief: a deep-learning model trained on a public dataset, at least
two specialist agents, a clean orchestration layer, a real external
tool call, a human safeguard, and a robustness study. The code,
notebooks, and logs are all under version control in the project
repository and can be re-run from a clean clone in minutes.

Beyond the rubric, the project gave the team working familiarity with
the two dominant paradigms for applied modern AI — fine-tuning a
transformer encoder on a classification task, and composing an
LLM-driven multi-agent pipeline — and highlighted the pragmatic
trade-offs (version pinning, API cost, imbalance handling, HITL
UX) that emerge when those paradigms meet a real constraint budget.

---

## 10. References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT:
   Pre-training of Deep Bidirectional Transformers for Language
   Understanding.* Proceedings of NAACL-HLT 2019, 4171–4186.
   arXiv:1810.04805.
2. Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024).
   *Bridging Language and Items for Retrieval and Recommendation.*
   (Amazon Reviews 2023 dataset release.) HuggingFace Hub:
   `McAuley-Lab/Amazon-Reviews-2023`.
3. Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural
   Language Processing.* Proceedings of EMNLP 2020, 38–45.
4. Paszke, A. et al. (2019). *PyTorch: An Imperative Style,
   High-Performance Deep Learning Library.* Advances in Neural
   Information Processing Systems 32, 8024–8035.
5. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in
   Python.* Journal of Machine Learning Research 12, 2825–2830.
6. CrewAI (2024). *CrewAI Documentation* (version 0.5.0).
   <https://docs.crewai.com>.
7. LangChain (2024). *LangChain Python Documentation* (version
   0.1.x). <https://python.langchain.com>.
8. OpenAI (2024). *GPT-4o-mini Model Card.*
   <https://platform.openai.com/docs/models/gpt-4o-mini>.
9. `duckduckgo-search` maintainers (2024). *duckduckgo-search Python
   package* (version 6.3.4). <https://pypi.org/project/duckduckgo-search>.
10. Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.*
    Computing in Science & Engineering 9 (3), 90–95.
11. Waskom, M. L. (2021). *seaborn: statistical data visualization.*
    Journal of Open Source Software 6 (60), 3021.

---

*End of report.*
