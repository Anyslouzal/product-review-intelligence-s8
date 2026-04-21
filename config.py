"""
config.py
---------
Centralized configuration loader for the product-review-intelligence project.

Responsibility:
    Read the OPENAI_API_KEY from a local .env file (never hardcoded) and
    expose a few simple constants used across agents/, data/, and main.py.

Design notes for oral defense:
    - We use python-dotenv so the secret lives outside of version control
      (the .env file is gitignored).
    - We fail loudly at import time if the key is missing, because every
      downstream component (CrewAI agent, ChatOpenAI LLM) depends on it.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root from this file's location so config works no
# matter which subdirectory Python is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load variables declared inside <PROJECT_ROOT>/.env into os.environ.
# load_dotenv is a no-op if the file does not exist, so we check explicitly.
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Pull the OpenAI API key out of the environment. We do NOT default it
# to a literal string: an unset key should crash rather than silently
# call a bogus endpoint.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Create a .env file at the project "
        "root containing: OPENAI_API_KEY=your_key_here"
    )

# Model identifier used by LangChain's ChatOpenAI wrapper.
# gpt-4o-mini is fast, inexpensive, and handles structured JSON output
# reliably for review analysis.
OPENAI_MODEL: str = "gpt-4o-mini"

# Common paths used by other modules (keeps path logic in one place).
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR: Path = PROJECT_ROOT / "data"
RUN_LOG_PATH: Path = LOGS_DIR / "run_log.json"

# Ensure the logs directory exists at import time so writers never have
# to worry about missing folders.
LOGS_DIR.mkdir(parents=True, exist_ok=True)
