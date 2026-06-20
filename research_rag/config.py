"""Central configuration for research_rag.

All paths and external endpoints are defined here so the rest of the
codebase never hardcodes them.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Filesystem layout -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"          # papers/<topic>/{raw,extracted}
LOGS_DIR = DATA_DIR / "logs"
TAXONOMY_DIR = DATA_DIR / "taxonomy"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

for _d in (DATA_DIR, PAPERS_DIR, LOGS_DIR, TAXONOMY_DIR, VECTORSTORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DOWNLOAD_LOG_PATH = LOGS_DIR / "downloads.jsonl"
TAXONOMY_PATH = TAXONOMY_DIR / "taxonomy.json"

# --- Security ----------------------------------------------------------------
# Only these hosts may ever be contacted by the downloader. Subdomains of
# these are also allowed (e.g. export.arxiv.org).
ALLOWED_DOMAINS = (
    "arxiv.org",
    "export.arxiv.org",
    "semanticscholar.org",
    "api.semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "inspirehep.net",       # HEP/quantum corpus with open fulltext/PDF links + API
    "dergipark.org.tr",     # open-access Turkish journals, serves PDFs directly
)

ALLOWED_CONTENT_TYPES = ("application/pdf",)
MAX_PDF_BYTES = 100 * 1024 * 1024  # 100 MB safety cap

# --- Rate limiting -------------------------------------------------------
ARXIV_REQUEST_DELAY_SECONDS = 3.0
SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS = 1.0
DOWNLOAD_DELAY_SECONDS = 2.0

# --- Semantic Scholar retry/backoff --------------------------------------
# Semantic Scholar's keyless API frequently returns HTTP 429 (rate limited).
# When enabled, search retries with exponential backoff instead of failing.
S2_BACKOFF_ENABLED = os.environ.get("S2_BACKOFF_ENABLED", "1") == "1"
S2_MAX_RETRIES = int(os.environ.get("S2_MAX_RETRIES", "5"))
S2_BACKOFF_BASE_SECONDS = float(os.environ.get("S2_BACKOFF_BASE", "2.0"))
S2_BACKOFF_MAX_SECONDS = float(os.environ.get("S2_BACKOFF_MAX", "60.0"))


# --- Semantic Scholar API key (optional) ---------------------------------
# A free key raises rate limits substantially. Read from the environment, or
# from a key file (default data/s2_api_key.txt; override with S2_API_KEY_FILE).
# When unset, the client falls back to keyless access.
def _load_s2_api_key() -> str:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        return key
    key_file = Path(os.environ.get("S2_API_KEY_FILE", str(DATA_DIR / "s2_api_key.txt")))
    try:
        if key_file.exists():
            # utf-8-sig so a BOM from a GUI editor (Notepad "save as UTF-8")
            # doesn't prefix the key with an invisible ﻿ that .strip()
            # leaves in place -- that would make a valid key look invalid.
            return key_file.read_text(encoding="utf-8-sig").strip()
    except OSError:
        pass
    return ""


SEMANTIC_SCHOLAR_API_KEY = _load_s2_api_key()

# --- Parallel download + rate limits -------------------------------------
DOWNLOAD_WORKERS = int(os.environ.get("DOWNLOAD_WORKERS", "4"))
# arXiv asks for <= ~1 request / 3s but tolerates a few req/s; we cap at 3/s.
ARXIV_RATE_PER_SEC = float(os.environ.get("ARXIV_RATE_PER_SEC", "3.0"))
# Semantic Scholar's documented limit WITH an API key is 1 request/second
# across all endpoints; keyless shares a small global pool. Default ~1 req/1.1s
# to sit just under the authenticated ceiling (0.6s tripped 429s). Raise only
# if S2 grants your key a higher rate.
S2_RATE_PER_MIN = float(os.environ.get("S2_RATE_PER_MIN", "55.0"))

# --- Snowball ingestion --------------------------------------------------
SNOWBALL_MAX_PAPERS = int(os.environ.get("SNOWBALL_MAX_PAPERS", "2000"))
SNOWBALL_REFS_PER_PAPER = int(os.environ.get("SNOWBALL_REFS_PER_PAPER", "50"))

# --- Nightly mode --------------------------------------------------------
NIGHTLY_BATCH_SIZE = int(os.environ.get("NIGHTLY_BATCH_SIZE", "50"))
NIGHTLY_THROTTLE_START_HOUR = int(os.environ.get("NIGHTLY_THROTTLE_START", "9"))
NIGHTLY_THROTTLE_END_HOUR = int(os.environ.get("NIGHTLY_THROTTLE_END", "22"))
NIGHTLY_CPU_TARGET = float(os.environ.get("NIGHTLY_CPU_TARGET", "0.70"))
NIGHTLY_LOG_PATH = LOGS_DIR / "nightly.log"
NIGHTLY_CHECKPOINT_PATH = LOGS_DIR / "nightly_checkpoint.json"

# --- Ollama ------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "research")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# On a CPU-only box Ollama can briefly 500 / time out while (re)loading a model
# after its keep-alive expires, or under transient memory pressure. Retry those
# so a single blip doesn't kill a long unattended classify/snowball run.
OLLAMA_MAX_RETRIES = int(os.environ.get("OLLAMA_MAX_RETRIES", "5"))
OLLAMA_BACKOFF_BASE_SECONDS = float(os.environ.get("OLLAMA_BACKOFF_BASE", "2.0"))
OLLAMA_BACKOFF_MAX_SECONDS = float(os.environ.get("OLLAMA_BACKOFF_MAX", "60.0"))

# --- GROBID (optional, best-effort metadata extraction) ----------------------
# GROBID requires running a separate Java service (see README). If it is not
# reachable, the extractor silently falls back to pymupdf/API metadata.
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070")
GROBID_ENABLED = os.environ.get("GROBID_ENABLED", "0") == "1"

# --- Qdrant ------------------------------------------------------------------
QDRANT_PATH = str(VECTORSTORE_DIR / "qdrant")
QDRANT_COLLECTION = "research_rag_papers"

# --- OCR -----------------------------------------------------------------
OCR_MIN_CHARS_PER_PAGE = 20  # below this, a page is considered "scanned"
OCR_DPI = 300
