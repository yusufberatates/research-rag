# Semantic Scholar API — usable reference

Collected from the official docs:
- Tutorial: <https://www.semanticscholar.org/product/api/tutorial>
- API docs hub: <https://api.semanticscholar.org/api-docs/>

This is what `research_rag` actually needs (paper search + references for the
snowball). It is scoped to the **Academic Graph API**; the Recommendations and
Datasets APIs are noted only briefly at the end.

## Base URLs

| API | Base URL |
|---|---|
| Academic Graph | `https://api.semanticscholar.org/graph/v1` |
| Recommendations | `https://api.semanticscholar.org/recommendations/v1` |
| Datasets | `https://api.semanticscholar.org/datasets/v1` |

## Authentication

- Send the key in the **`x-api-key`** request header. No key works too (keyless),
  but shares a small global pool and gets 429s far more often.
- Get a key from the API product page (`/product/api#api-key`).
- In this repo: `research_rag/config.py:_load_s2_api_key()` reads
  `SEMANTIC_SCHOLAR_API_KEY` from the env, falling back to
  `data/s2_api_key.txt` (override path with `S2_API_KEY_FILE`).
  `semantic_scholar_client.py:_headers()` attaches the header when present.

## Rate limits

- **With an approved key: 1 request/second across *all* endpoints.** Some
  accounts are granted more after review.
- **Keyless: a shared global limit** — much lower in practice, expect 429s.
- Honor the `Retry-After` header on a 429 when present; otherwise use
  exponential backoff.
- In this repo: `config.py:S2_RATE_PER_MIN` defaults to **55/min (~0.92 req/s)**,
  intentionally just under the 1 req/s ceiling (0.6s spacing tripped 429s).
  `downloader/rate_limit.py:s2_limiter` enforces the spacing process-wide across
  the 4 download workers, and `_get_with_backoff()` already honors `Retry-After`
  then falls back to exponential backoff (`S2_BACKOFF_*` in config). **This is
  already aligned with the documented limits — no change needed.**

## Endpoints we use

### Paper relevance search — `GET /paper/search`
Best for "top N most relevant" — which is what topic downloads want.
- Params: `query` (required), `fields`, `limit` (default 100, **max 100**),
  `offset` (paginate; relevance search is capped at offset+limit ≤ 1000).
- Optional filters: `year` (e.g. `2019-2023`, `2020-`, `-2015`),
  `fieldsOfStudy`, `publicationTypes`, `venue`, `minCitationCount`,
  `openAccessPdf` (presence flag — restricts to papers with an open-access PDF).
- Response: `{ "total", "offset", "next", "data": [ {paper}, ... ] }`.
- In this repo: `semantic_scholar_client.py:search()`.

### Paper references — `GET /paper/{paper_id}/references`
Powers the citation snowball.
- Params: `fields` (applied to the nested `citedPaper`), `limit` (**max 1000**),
  `offset`.
- Response rows look like `{ "citedPaper": {paper}, ... }`.
- `{paper_id}` accepts many id forms, including `ARXIV:<id>`, `DOI:<doi>`,
  `CorpusId:<n>`, or the raw S2 `paperId`.
- In this repo: `semantic_scholar_client.py:get_references()` and
  `paper_ref()`, which maps our internal ids to a usable ref — `arxiv_<id>` →
  `ARXIV:<id>` and `s2_<paperId>` → the raw `<paperId>` (the API takes it
  directly). `arxiv_ref()` is the older arXiv-only helper kept for back-compat;
  snowball uses `paper_ref()` so Semantic-Scholar-sourced papers aren't dead
  ends for citation expansion.

### Bulk search — `GET /paper/search/bulk` (not currently used)
For sweeping large result sets (not relevance-ranked the same way).
- Token pagination: response includes a `token`; pass it back as `token=` to get
  the next page (up to ~10M results, 1000/page).
- Supports a boolean query syntax in `query` (see below).
- Use this instead of `/paper/search` only if we ever need >1000 results for a
  query; for top-15 topic downloads, relevance search is the right call.

### Batch lookup — `POST /paper/batch` (not currently used)
Fetch fields for up to **500 paper ids in one request**: body
`{"ids": ["ARXIV:...", "DOI:...", ...]}`, `fields` as a query param. Far cheaper
than N single `GET /paper/{id}` calls if we ever hydrate many known ids.

## `fields` parameter — valid values for papers

Comma-separated, **no spaces**. Request only what you need (extra fields slow the
response). Nested fields use dot paths.

```
paperId, corpusId, externalIds, url, title, abstract, venue,
publicationVenue, year, publicationDate, journal, referenceCount,
citationCount, influentialCitationCount, isOpenAccess, openAccessPdf,
fieldsOfStudy, s2FieldsOfStudy, publicationTypes, tldr,
authors            (-> authors.authorId, authors.name, ...)
citations          (-> citations.title, ...)
references         (-> references.title, ...)
embedding, embedding.specter_v2
```

- `externalIds` → `{ "DOI", "ArXiv", "PubMed", "CorpusId", ... }`.
- `openAccessPdf` → `{ "url", "status" }`; the URL is what we whitelist-check
  before downloading.
- `tldr` → `{ "model", "text" }`, an auto-generated one-sentence summary.
- In this repo: `FIELDS = "title,abstract,year,authors,openAccessPdf,externalIds"`
  (`semantic_scholar_client.py:30`). Candidate enrichments — all optional —
  are `tldr` (cheap signal for the relevance gate / summarizer),
  `fieldsOfStudy` / `publicationTypes` (could sharpen tier/field routing), and
  `publicationDate` (finer than `year`). Each added field costs response time,
  so add deliberately.

## Boolean query syntax (bulk search `query`)

| Operator | Meaning |
|---|---|
| `"exact phrase"` | match phrase in title/abstract |
| `term*` | prefix wildcard |
| `term~N` | fuzzy match, edit distance N |
| `+term` | required |
| `-term` | excluded |
| `a \| b` | OR |
| `(...)` | grouping |

Example: `((cloud computing) | virtualization) +security -privacy`

## HTTP status codes

| Code | Meaning / action |
|---|---|
| 200 | OK |
| 400 | Bad request — check params (e.g. bad `fields` name) |
| 401 | Missing/invalid key |
| 403 | Not permitted for this resource |
| 404 | Unknown endpoint/paper id |
| 429 | Rate limited — back off, honor `Retry-After` |
| 5xx | Transient server error — retry with backoff |

`research_rag` retries `{429, 500, 502, 503, 504}` and raises on the rest
(`_RETRYABLE_STATUS` in `semantic_scholar_client.py`). Note: the live run on
2026-06-17 saw a one-off `500` on `/paper/search` that succeeded on retry —
exactly what the backoff path is for.

## Other APIs (reference only)

- **Recommendations** — `POST /recommendations/v1/papers` with
  `positivePaperIds` / `negativePaperIds`; returns ranked related papers.
- **Datasets** — bulk JSON snapshots (`/release/`, `/release/{id}`,
  per-dataset download links requiring a key, and incremental `diffs`). Useful
  only for very high-volume local mirroring, not for this project's runtime.
