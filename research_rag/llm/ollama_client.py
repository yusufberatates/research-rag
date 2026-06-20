"""Minimal client for the local Ollama HTTP API.

This is the only place direct (non-llama-index) Ollama calls are made --
used by the classifier for cheap text-generation tasks (summaries, field
assignment) where pulling in the full llama-index LLM abstraction isn't
needed. Indexing/query use llama-index's own Ollama integrations instead.
"""
from __future__ import annotations

import time

import requests

from research_rag.config import (
    OLLAMA_BACKOFF_BASE_SECONDS,
    OLLAMA_BACKOFF_MAX_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OLLAMA_MAX_RETRIES,
)

# Server errors worth retrying. A CPU-only box returns these (or times out)
# while the model (re)loads after keep-alive expiry or under memory pressure.
_RETRYABLE_STATUS = {500, 502, 503, 504}


def generate(prompt: str, system: str | None = None, model: str | None = None, timeout: int = 300) -> str:
    """Call Ollama's /api/generate, retrying transient failures with backoff.

    Retries connection errors, timeouts, and 5xx responses (the model briefly
    reloading / OOM on a CPU box) up to ``OLLAMA_MAX_RETRIES`` times; raises on
    4xx (a real client error that won't fix itself) or once retries run out.
    This keeps a single blip from killing a long classify/snowball run.
    """
    payload = {
        "model": model or OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    url = f"{OLLAMA_BASE_URL}/api/generate"

    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= OLLAMA_MAX_RETRIES:
                raise
            _backoff_sleep(attempt, f"Ollama unreachable ({type(exc).__name__})")
            continue
        if resp.status_code in _RETRYABLE_STATUS and attempt < OLLAMA_MAX_RETRIES:
            _backoff_sleep(attempt, f"Ollama returned {resp.status_code}")
            continue
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    # Loop always returns or raises above; keeps type checkers happy.
    raise RuntimeError("unreachable")


def _backoff_sleep(attempt: int, reason: str) -> None:
    delay = min(
        OLLAMA_BACKOFF_BASE_SECONDS * (2 ** attempt), OLLAMA_BACKOFF_MAX_SECONDS
    )
    print(
        f"{reason}; retrying in {delay:.0f}s "
        f"(attempt {attempt + 1}/{OLLAMA_MAX_RETRIES})."
    )
    time.sleep(delay)


def check_health(timeout: int = 10) -> tuple[bool, str]:
    """Preflight: is Ollama reachable and are the configured models present?

    Returns ``(ok, message)``. Lets a caller (e.g. the pilot script) fail fast
    with one clear, actionable message instead of grinding every stage against
    a dead backend and emitting a wall of connection-refused tracebacks.
    """
    from research_rag.config import OLLAMA_EMBED_MODEL

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return False, (
            f"Ollama is NOT reachable at {OLLAMA_BASE_URL} ({type(exc).__name__}). "
            "Start it first: run 'ollama serve' (or launch the Ollama app), then retry."
        )

    installed = {m.get("name", "") for m in resp.json().get("models", [])}
    base_names = {n.split(":", 1)[0] for n in installed}
    missing = [
        m
        for m in (OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL)
        if m not in installed and m.split(":", 1)[0] not in base_names
    ]
    if missing:
        return False, (
            f"Ollama is up but missing model(s): {', '.join(missing)}. "
            f"Installed: {', '.join(sorted(installed)) or '(none)'}. "
            f"Pull/create them first (e.g. 'ollama pull {missing[0]}')."
        )
    return True, (
        f"Ollama OK at {OLLAMA_BASE_URL}; models present: "
        f"{OLLAMA_LLM_MODEL}, {OLLAMA_EMBED_MODEL}."
    )
