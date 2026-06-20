"""Append-only log of every file the system has ever written to disk,
with a SHA-256 checksum, so downloads are auditable after the fact."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from research_rag.config import DOWNLOAD_LOG_PATH


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_download(*, source: str, url: str, topic: str, file_path: Path, paper_id: str) -> dict:
    record = {
        "timestamp": time.time(),
        "source": source,
        "url": url,
        "topic": topic,
        "paper_id": paper_id,
        "file_path": str(file_path),
        "sha256": sha256_of_file(file_path),
        "size_bytes": file_path.stat().st_size,
    }
    with open(DOWNLOAD_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record
