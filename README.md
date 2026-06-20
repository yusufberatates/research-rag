# research_rag

A **local, offline** research‑paper RAG system. It downloads papers from a small
whitelist of academic sources (arXiv / Semantic Scholar / PubMed), extracts and
classifies them into a self‑updating field taxonomy, indexes them into a local
vector store, and answers questions over the corpus **with paper citations** —
all through a local [Ollama](https://ollama.com) model. **No cloud AI APIs** are
used at runtime; the only internet access is downloading papers from the
whitelisted academic sites.

Out of the box the taxonomy is seeded for **quantum radar / quantum sensing**
(see `research_rag/classifier/seed_taxonomy.py`), but the pipeline is general —
re‑seed that file for another domain.

---

## Step-by-step: from zero to first answer

Follow these in order. Steps 1–2 are one-time setup; after that you only repeat
steps 5–6. Run everything from the project folder (wherever you cloned it).

### Step 1 — Install Ollama and pull the two models (one-time)

Install [Ollama](https://ollama.com), then:

```powershell
ollama pull nomic-embed-text
ollama pull research            # the LLM; or use your own and set OLLAMA_LLM_MODEL
ollama list                     # confirm both appear
```

Make sure Ollama is running (the desktop app, or `ollama serve` in a terminal).

### Step 2 — Create the virtual environment and install deps (one-time)

```powershell
cd path\to\research_rag
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After this you just type `python main.py …` — `main.py` auto-targets `.venv`, so
you never have to activate it or remember the venv path.

### Step 3 — (Optional but recommended) add a Semantic Scholar key

Lets downloads/snowball run authenticated instead of keyless (fewer rate limits):

```powershell
notepad data\s2_api_key.txt     # paste the key on line 1, save
```

### Step 4 — Verify everything is wired up

```powershell
python main.py healthcheck
```

You want to see:

```
Ollama OK at http://localhost:11434; models present: research, nomic-embed-text.
Semantic Scholar key: detected (len 44); S2 calls will be authenticated.
```

(`Ollama OK` is required; if the key line says "NOT found" you can still proceed
keyless.)

### Step 5 — Build a corpus

Either fetch papers for a topic…

```powershell
python main.py pipeline "quantum illumination" --max-results 8 --gate
```

…or run the turnkey pilot that builds a whole tier-1 corpus (Windows):

```powershell
.\run_tier1_pilot.ps1 -Reset
```

### Step 6 — Ask questions (with citations)

```powershell
python main.py query "What is quantum illumination?"
```

The output is an answer plus the papers (title, authors, year) it was drawn from.

### Step 7 — (Optional) keep importing + classifying until you stop it

```powershell
python main.py snowball --tier 3 --max-papers 0 --nightly
```

`--max-papers 0` = unlimited; it runs until the citation queue is exhausted or
you press **Ctrl+C** (which saves a checkpoint — re-run the same command to
resume). See **Corpus growth** below for details.

### Step 8 — Inspect what you've built

```powershell
python main.py pipeline_stats     # corpus size, tiers, tree shape, embedding status
python main.py taxonomy           # the field tree
```

---

## Prerequisites

- **Ollama** running locally, with two models pulled:
  ```powershell
  ollama pull nomic-embed-text
  ollama pull research            # or set OLLAMA_LLM_MODEL to your own model
  ```
  (`research` is a user‑defined local model; any capable local LLM works — point
  `OLLAMA_LLM_MODEL` at it. On a CPU‑only / low‑RAM box, prefer a quantized
  1–7B model.)
- **Python 3.11** with the project's virtual environment (see Setup). The bare
  global Python on PATH does **not** have the dependencies — but you don't need
  to worry about that, see the note above.
- *(Optional)* **Tesseract OCR** on PATH — fallback for scanned PDFs. Without it,
  scanned pages just yield empty text instead of erroring.
- *(Optional)* A running **GROBID** service for higher‑quality metadata.
  Disabled by default; the extractor falls back to API/pymupdf metadata.

## Setup

```powershell
cd path\to\research_rag
py -3.11 -m venv .venv                 # create the venv (once)
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

After this, just use `python main.py …` (it self‑targets the venv).

---

## Semantic Scholar API key (optional, recommended)

A free key makes Semantic Scholar reliable; without it, downloads and the
citation **snowball** run keyless and hit frequent HTTP 429 rate limits.

Provide it **either** way — the env var takes precedence over the file:

```powershell
# A) environment variable (this session only):
$env:SEMANTIC_SCHOLAR_API_KEY = "<your key>"

# B) key file (session-independent; create it, run, delete when done):
notepad data\s2_api_key.txt        # paste the key on line 1, save
```

Confirm it's actually loaded (never prints the key):

```powershell
python main.py healthcheck          # -> "Semantic Scholar key: detected (len N)"
```

> Note: the `/paper/search` endpoint is rate‑limited hard by Semantic Scholar
> and may 429 in bursts **even with a valid key** — that's expected and is
> retried automatically. Retry log lines are tagged `[keyed|KEYLESS / search|refs]`
> so you can always tell whether your key was attached.

---

## Commands

Run `python main.py <command> --help` for full options.

### Core pipeline

| Command | What it does |
|---|---|
| `pipeline "<topic>" [--max-results N] [--gate] [--tier 1\|2\|3]` | download → extract → classify → index, in one go |
| `download "<topic>" [--query Q] [--max-results N] [--gate] [--tier N]` | search + security‑checked PDF download |
| `extract "<topic>"` | PDF text + metadata (pymupdf, OCR fallback) |
| `classify "<topic>"` | LLM summary + field/subfield/sub‑subfield assignment (run **before** `index`) |
| `index "<topic>"` | embed + add to the vector store (skips unclassified records with a warning) |
| `query "<question>"` | route → retrieve → answer **with citations** |

`--gate` LLM‑filters candidates against the taxonomy *before* downloading, so
off‑topic papers never get fetched. Prefer it over forcing `--tier` (let tier
derive from the assigned field).

### Corpus growth

| Command | What it does |
|---|---|
| `snowball [--tier N] [--max-papers N] [--refs N] [--dry-run] [--nightly]` | grow the corpus by following citations (LLM relevance‑judged), resumable |

`--nightly` adds checkpointing, daytime CPU throttling, file logging
(`data/logs/nightly.log`), and a completion notification — for unattended runs.

**Run it limitlessly** (import + classify until *you* stop it) with
`--max-papers 0` — no cap; it runs until the citation queue is exhausted or you
press **Ctrl+C** (which checkpoints; a plain re-run resumes). Pair it with
`--nightly` for that resumability, and use a broad `--tier` so it doesn't
converge quickly the way a narrow tier-1 frontier does:

```powershell
python main.py snowball --tier 3 --max-papers 0 --nightly
```

On a CPU-only machine expect ~3–4 min/paper, so a limitless run lasts days; set
`$env:NIGHTLY_CPU_TARGET = "1.0"` to disable the daytime throttle for full speed,
and set the S2 key first (a long keyless run will be 429-heavy).

### Inspection & maintenance

| Command | What it does |
|---|---|
| `healthcheck` | verify Ollama is reachable + models present, and report S2 key status |
| `taxonomy` | print the current field tree |
| `pipeline_stats` | corpus size, disk usage, tree shape, tiers, embedding status |
| `consolidate_taxonomy` | merge semantically redundant sibling nodes + re‑tag papers |
| `reclassify` | re‑classify the whole corpus from a fresh seed (destructive; slow) |
| `rebuild_taxonomy` | re‑attach already‑classified records to the tree (no LLM) |
| `reset_taxonomy` | overwrite the tree with the 7 seeded fields (destructive) |
| `reset_index` | drop the local Qdrant collection |

### Turnkey pilot (Windows)

`run_tier1_pilot.ps1` builds a ~300‑paper, relevance‑gated tier‑1 corpus end to
end (gated downloads → extract → classify → index → snowball → stats):

```powershell
.\run_tier1_pilot.ps1 -Reset      # first run: wipe taxonomy + index, then build
.\run_tier1_pilot.ps1             # resume after an interruption (dedup + checkpoints)
```

It runs an Ollama preflight first and aborts cleanly if the model backend is
down. Tee the output to a log: `... *>&1 | Tee-Object usermadelog.txt`.

---

## How it works (short version)

A three‑level taxonomy — **main field → subfield → sub‑subfield**, papers attach
at the leaf — with seven fixed seeded top‑level fields. Every node carries a
short LLM‑written **descriptor**, so the model never re‑reads the whole corpus:
new papers are matched against descriptors to classify, and questions are routed
down the same descriptors to a leaf before vector retrieval (falling back to a
flat search if routing finds nothing).

## Configuration

Defaults live in `research_rag/config.py`; override via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_LLM_MODEL` | `research` | LLM for summary/classify/answer |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `OLLAMA_MAX_RETRIES` / `OLLAMA_BACKOFF_BASE` / `OLLAMA_BACKOFF_MAX` | `5` / `2.0` / `60.0` | retry transient Ollama 5xx / timeouts |
| `SEMANTIC_SCHOLAR_API_KEY` | *(unset)* | S2 key (or use `data/s2_api_key.txt`) |
| `S2_MAX_RETRIES` / `S2_BACKOFF_MAX` | `5` / `60.0` | S2 429/5xx backoff |
| `GROBID_ENABLED` / `GROBID_URL` | `0` / `http://localhost:8070` | optional GROBID metadata |
| `RESEARCH_RAG_NO_REEXEC` | *(unset)* | set to `1` to disable the venv auto‑re‑exec in `main.py` |

## Data layout

```
data/
  papers/<topic_slug>/raw/<paper_id>.pdf          # downloaded PDFs
  papers/<topic_slug>/raw/<paper_id>.meta.json    # API metadata sidecar
  papers/<topic_slug>/extracted/<paper_id>.json   # text + metadata + summary + field/subfield
  taxonomy/taxonomy.json                          # field/subfield descriptors (self-updating)
  vectorstore/qdrant/                             # local embedded Qdrant DB
  logs/downloads.jsonl                            # audit log: every download + SHA-256 checksum
  logs/nightly.log                                # snowball/nightly progress
```

## Security model

- **Domain whitelist** — every outbound request (search + PDF) is checked
  against `arxiv.org`, `semanticscholar.org`, `pubmed.ncbi.nlm.nih.gov` and
  their subdomains first; anything else is refused.
- **PDF‑only downloads** — saved only if `Content-Type: application/pdf` *and*
  the bytes start with `%PDF-`, streamed under a 100 MB cap.
- **Sandboxed extraction** — PDFs are only ever parsed as data by pymupdf, never
  executed or handed to an external viewer/shell, so malicious embedded
  JS/launch actions never run.
- **Audit log** — every saved file is SHA‑256 checksummed into
  `data/logs/downloads.jsonl` with its source URL and topic.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'fitz'` | You ran the bare global Python. `main.py` now auto‑re‑execs into `.venv`; if you disabled that, use `.\.venv\Scripts\python.exe main.py …`. |
| `ConnectionRefusedError ... 11434` everywhere | Ollama isn't running. Start it (`ollama serve` / desktop app); `python main.py healthcheck` confirms. |
| 429 storms during download/snowball | Semantic Scholar rate limiting. Set a key (see above); `[keyed/search]` storms are normal S2 throttling and are retried. |
| `healthcheck` says key **NOT found** but you set a file | The file may be empty/BOM‑only (e.g. an empty paste). Re‑paste the key into `data\s2_api_key.txt` and re‑check; `detected (len N)` means it's loaded. |
| Run looks "frozen" in the Tee'd log | Output buffering; the pilot sets `PYTHONUNBUFFERED=1`. For live snowball progress watch `data\logs\nightly.log`. |
| Only one command at a time | Qdrant opens its data dir exclusively — don't run two `research_rag` commands against the same `data/vectorstore` simultaneously. |

## Known limitations

- Classification quality is bounded by the local model; occasional misroutes
  happen (e.g. LiDAR papers landing under detection/illumination). Clean up with
  `consolidate_taxonomy` and/or `reclassify`.
- GROBID and OCR are optional/best‑effort; the pipeline degrades gracefully
  without them.
