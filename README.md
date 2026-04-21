# Product Review Intelligence

A multi-agent AI system that reads a product review and returns a
structured analysis (sentiment, key themes, one-sentence summary).

This repository contains the **Week 1** scope: a single CrewAI
orchestrator agent backed by Gemini 2.0 Flash, a HuggingFace dataset
ingestion script, and an exploratory data analysis notebook.

> Course context: ESIN IA S8 — multi-agent systems project.

---

## Project layout

```
product-review-intelligence/
├── agents/
│   └── orchestrator.py       # CrewAI orchestrator agent
├── tools/                    # placeholder for W2/W3 CrewAI tools
│   └── __init__.py
├── data/
│   └── download_dataset.py   # downloads 5 000 All_Beauty reviews
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis
├── logs/
│   └── run_log.json          # audit log (created at first run)
├── docs/
│   └── architecture.md       # architecture document
├── config.py                 # loads GEMINI_API_KEY from .env
├── main.py                   # demo: 3 reviews -> agent -> log
├── requirements.txt
└── README.md
```

## 1. Requirements

- Python **3.10+**
- A Gemini API key (free tier) from
  <https://aistudio.google.com/app/apikey>

## 2. Setup

```bash
# 1) Clone the repo
git clone <repo-url> product-review-intelligence
cd product-review-intelligence

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Create a .env file at the repo root
cat > .env <<'EOF'
GEMINI_API_KEY=your_key_here
EOF
```

The `.env` file is gitignored — your key never leaves the machine.

## 3. Run the agent demo

```bash
python main.py
```

This runs the orchestrator on three hardcoded reviews
(positive / negative / mixed), prints the JSON output to stdout, and
appends each run to `logs/run_log.json` with a UTC timestamp.

Expected output shape per review:

```json
{
  "sentiment": "positive",
  "key_themes": ["hydration", "scent", "value"],
  "summary": "A well-liked moisturizer praised for lasting hydration and good value."
}
```

## 4. Download the dataset

```bash
python data/download_dataset.py
```

Writes the first 5 000 rows of
`McAuley-Lab/Amazon-Reviews-2023 / raw_review_All_Beauty` to
`data/reviews.csv`.

## 5. Run the EDA notebook

```bash
jupyter notebook notebooks/eda.ipynb
# or
jupyter lab
```

The notebook produces:
1. Class distribution of ratings.
2. Average review length per rating.
3. Top-20 non-stopword tokens.
4. Sample of 5 reviews per rating class.

## 6. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `RuntimeError: GEMINI_API_KEY is not set` | Create `.env` at the repo root with `GEMINI_API_KEY=...` |
| `datasets` fails to load | Re-run with an active internet connection; streaming mode needs HF to be reachable. |
| Rate-limit errors from Gemini | Free tier has a per-minute cap — wait and retry. |

## 7. Roadmap

- **W2** — split the orchestrator into sentiment / theme / summarizer
  specialist agents, add CrewAI tools, start an evaluation slice.
- **W3** — add retrieval over the local dataset, caching, and a small
  batch-mode CLI.

See `docs/architecture.md` for the full design document.
