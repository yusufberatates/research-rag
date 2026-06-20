"""Generate a one-paragraph summary of a paper using the local Ollama LLM."""
from __future__ import annotations

from research_rag.llm import generate

# Keep the prompt within a reasonable context window for local models.
MAX_CHARS_FOR_SUMMARY = 12000

SYSTEM_PROMPT = (
    "You are a research assistant. Summarize academic papers accurately "
    "and concisely. Never invent facts not present in the text."
)


def summarize_paper(record: dict) -> str:
    text = record.get("abstract") or ""
    if len(text) < 200:
        text = record.get("full_text", "")[:MAX_CHARS_FOR_SUMMARY]

    prompt = (
        f"Title: {record.get('title', 'Unknown')}\n\n"
        f"Paper text (may be truncated):\n{text}\n\n"
        "Write a single concise paragraph (4-6 sentences) summarizing this "
        "paper's problem, method, and key findings. Do not use bullet points."
    )
    summary = generate(prompt, system=SYSTEM_PROMPT)
    return summary or record.get("abstract", "")
