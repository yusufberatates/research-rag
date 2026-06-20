# Useful Resources

Curated web resources relevant to this project (a local, offline RAG over
**quantum-radar** literature). Auto-matched from the maintainer's bookmark
classifier on 2026-06-18, then hand-checked against this project's design.

> **Runtime note:** the RAG pipeline is offline / local-Ollama and contacts only
> whitelisted academic hosts (see `research_rag/config.py` `ALLOWED_DOMAINS`).
> The cloud AI tools below are **development-time helpers only** — they are not
> part of the runtime path.

---

## 1. Domain literature & field tracking
Stay current on the field and refine the 15 seed queries (`info.txt`, Phase 5).

- [QEPNT — UK Hub for Quantum Position, Navigation & Timing](https://www.qepnt.org/study/) — quantum sensing/navigation, directly adjacent to quantum radar.
- [The Quantum Insider — leading research institutions](https://thequantuminsider.com/2022/05/16/quantum-research/) — who is publishing in the field.
- [Quantum Zeitgeist](https://quantumzeitgeist.com/) — daily quantum computing/tech news.
- [Inside Quantum Technology](https://www.insidequantumtechnology.com/) — industry/research news.
- [Q-CADE Lab](https://q-cade.ai/) — quantum research group.
- [Aquark Technologies — miniaturised cold-atom hardware](https://www.aquarktechnologies.com/) — relevant to Rydberg/atomic-receiver hardware (seed query 6).
- [ML4Q](https://ml4q.de/ml4q-internship/) — Matter and Light for Quantum Information cluster.
- [arXiv — Condensed Matter (new)](https://arxiv.org/list/cond-mat/new) — primary preprint source (already whitelisted as a download host).
- [Physical Review Letters — recent](https://journals.aps.org/prl/recent) — flagship venue for the seed-query topics.
- [INSPIRE-HEP](https://inspirehep.net/) — HEP/quantum literature + API; see whitelist note below.

**Optics & photonics** (entangled photons / SPDC — seed queries 8, 11):
- [SPIE](https://spie.org/) — optics & photonics society.
- [SPIE Digital Library](https://www.spiedigitallibrary.org/) — applied optics/photonics papers.
- [Thorlabs](https://www.thorlabs.com/) — photonics components reference (lasers, SPDC optics).
- [Kaan Akşit](https://www.kaanaksit.com/) — computational light / optics researcher.
- [Prof. Hakan Altan (METU)](https://users.metu.edu.tr/haltan/index.htm) — terahertz tech, photonics, ultrafast lasers.

## 2. Paper discovery & acquisition
Feeds the `downloader` and the planned `snowball` expansion.

- [Google Scholar](https://scholar.google.com/) — discovery & citation graph (manual; not a clean PDF host).
- [INSPIRE-HEP](https://inspirehep.net/) — searchable HEP corpus with open fulltext/PDF links and an API.
- [DergiPark](https://dergipark.org.tr/en/) — open-access Turkish journals; serves PDFs directly.
- [Academic tools list (S. Albayrak)](https://soneralbayrak.com/academictrivia/websitese2.html) — points to **Elicit, Consensus, Research Rabbit, iArxiv** — AI paper-discovery tools that directly support the snowball/relevance-judging phase.
- [ResearchGate](https://www.researchgate.net/) — author networks (manual; login-walled, not for automated download).
- [EBSCO](https://research.ebsco.com/c/cdtuln/search?isDashboardExpanded=true) — institutional database (manual; behind auth).

**Shadow libraries — manual, human-discretion only (NOT in the download whitelist):**
- [Library Genesis](https://libgen.is/), [Anna's Archive](https://annas-archive.org/), [SLUM uptime monitor](https://open-slum.org/) — these host pirated content. They are intentionally **kept out of the automated `ALLOWED_DOMAINS`** because wiring them into a bulk downloader raises legal/ethical issues and contradicts this project's "whitelist of legitimate academic sources" design. Listed here only for awareness.

## 3. Building the system (dev references)
- [Colab — TensorFlow with GPU](https://colab.research.google.com/notebooks/gpu.ipynb) — quick GPU sandbox for embedding/model experiments.
- [Python docs — built-in functions](https://docs.python.org/3/library/functions.html), [pandas.DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), [seaborn](https://seaborn.pydata.org/), [Vega-Altair](https://altair-viz.github.io/index.html) — data handling & viz for `stats.py` / corpus analysis.
- [Jupyter](https://jupyter.org/install) — interactive corpus/taxonomy inspection.
- [GitHub](https://github.com/) · [Stack Overflow](https://stackoverflow.com/) — code & troubleshooting.
- [conda-forge](https://conda-forge.org/) · [pip](https://pip.pypa.io/en/stable/) — environment & the pinned `requirements.txt` (Phase 1 #3).
- [Sphinx](https://www.sphinx-doc.org/en/master/) + [furo theme](https://github.com/pradyunsg/furo) — if these `docs/` ever become a rendered site.
- [Google Cloud Storage — projects](https://cloud.google.com/storage/docs/projects) — only if corpus/index is ever moved off local disk.

**Dev-time AI (never in the runtime path):**
- [Claude](https://claude.ai/) · [NotebookLM](https://notebooklm.google.com/) — design, code, and reasoning over docs while developing.
- [Anthropic — "Building a C compiler with parallel Claudes"](https://www.anthropic.com/engineering/building-c-compiler) — multi-agent orchestration pattern, useful inspiration for the LLM classifier/snowball-judge stages.
