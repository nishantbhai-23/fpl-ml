"""Endpoints, paths and tunables.

Everything you might want to change lives here, so that in month four you are
not grepping for a magic number buried three modules deep.
"""

from __future__ import annotations

from pathlib import Path

# The FPL API is undocumented but stable and public.
BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{BASE_URL}/fixtures/"
ELEMENT_SUMMARY_URL = BASE_URL + "/element-summary/{element_id}/"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"

# Identify ourselves honestly rather than impersonating a browser.
USER_AGENT = "fpl-ml/0.1 (+https://github.com/nishantbhai-23/fpl-ml)"

REQUEST_TIMEOUT = 30.0

# The per-player sweep is ~700 requests against a free, unofficial API.
# Eight at a time finishes in about twenty seconds without looking like abuse.
MAX_CONCURRENCY = 8

MAX_RETRIES = 4
BACKOFF_BASE = 0.5
BACKOFF_CAP = 20.0
