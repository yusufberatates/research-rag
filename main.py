#!/usr/bin/env python
"""Entry point: research_rag CLI.

Usage examples:
    python main.py pipeline "transformer attention" --max-results 5
    python main.py query "What architectures improve attention efficiency?"
"""
import os
import sys
from pathlib import Path


# --- Auto-use the project venv -------------------------------------------------
# Every dependency (pymupdf/fitz, llama-index, qdrant, ...) lives in ./.venv, NOT
# in the global Python on PATH. If main.py is
# launched with any other interpreter -- the natural `python main.py ...` -- we
# transparently re-run under the venv so imports resolve instead of dying with
# `ModuleNotFoundError: No module named 'fitz'`. Set RESEARCH_RAG_NO_REEXEC=1 to
# opt out (e.g. when you deliberately manage the environment yourself).
def _reexec_in_venv() -> None:
    if os.environ.get("RESEARCH_RAG_NO_REEXEC"):
        return
    scripts = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    venv_py = Path(__file__).resolve().parent / ".venv" / scripts / exe
    if not venv_py.exists():
        return  # no venv present; run with the current interpreter
    try:
        if Path(sys.executable).resolve() == venv_py.resolve():
            return  # already running under the venv -- nothing to do
    except OSError:
        return
    import subprocess

    completed = subprocess.run([str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(completed.returncode)


_reexec_in_venv()

# Paper titles, author names, and our own separators contain Unicode (em-dashes,
# accents, physics symbols). The Windows console defaults to a legacy code page
# (e.g. cp1252/cp857) that mangles these into "?"/"�", so force UTF-8 on the
# standard streams. Guarded for streams that predate reconfigure() or are
# redirected to something without it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from research_rag.cli import main  # noqa: E402  (after venv re-exec + stream reconfig)

if __name__ == "__main__":
    raise SystemExit(main())
