"""Nightly-mode controller: batch checkpointing, resumability, CPU throttling
during daytime hours, file logging, and a desktop notification on completion.

Designed for unattended long runs on a single laptop (see the hardware memo:
CPU-only, 12GB RAM). All behaviour is opt-in via ``enabled`` so the same
ingestion code can run interactively without any of it.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time

from research_rag.config import (
    NIGHTLY_BATCH_SIZE,
    NIGHTLY_CHECKPOINT_PATH,
    NIGHTLY_CPU_TARGET,
    NIGHTLY_LOG_PATH,
    NIGHTLY_THROTTLE_END_HOUR,
    NIGHTLY_THROTTLE_START_HOUR,
)


class NightlyController:
    def __init__(self, job: str, enabled: bool = True, batch_size: int | None = None):
        self.job = job
        self.enabled = enabled
        self.batch_size = batch_size or NIGHTLY_BATCH_SIZE

    # --- logging ---------------------------------------------------------- #
    def log(self, message: str) -> None:
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{self.job}] {message}"
        print(line)
        if self.enabled:
            try:
                with open(NIGHTLY_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    # --- checkpointing ---------------------------------------------------- #
    def save_checkpoint(self, state: dict) -> None:
        if not self.enabled:
            return
        payload = {"job": self.job, **state}
        tmp = NIGHTLY_CHECKPOINT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(NIGHTLY_CHECKPOINT_PATH)

    def load_checkpoint(self) -> dict | None:
        if not NIGHTLY_CHECKPOINT_PATH.exists():
            return None
        try:
            data = json.loads(NIGHTLY_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("job") != self.job:
            return None  # checkpoint belongs to a different job
        return data

    def clear_checkpoint(self) -> None:
        NIGHTLY_CHECKPOINT_PATH.unlink(missing_ok=True)

    # --- CPU throttling --------------------------------------------------- #
    def in_throttle_window(self, now: _dt.datetime | None = None) -> bool:
        hour = (now or _dt.datetime.now()).hour
        start, end = NIGHTLY_THROTTLE_START_HOUR, NIGHTLY_THROTTLE_END_HOUR
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # window wraps past midnight

    def throttle(self, work_seconds: float) -> None:
        """Sleep after a unit of work so CPU duty stays near the target during
        the daytime window; full speed (no sleep) outside it."""
        if not self.enabled or work_seconds <= 0:
            return
        if not self.in_throttle_window():
            return
        target = min(max(NIGHTLY_CPU_TARGET, 0.05), 1.0)
        idle = work_seconds * (1.0 / target - 1.0)
        if idle > 0:
            time.sleep(idle)

    # --- notification ----------------------------------------------------- #
    def notify(self, title: str, message: str) -> None:
        self.log(f"NOTIFY: {title} - {message}")
        if not self.enabled:
            return
        # Best-effort Windows balloon notification, fire-and-forget.
        # Double any embedded single quote so a title/message like
        # "Lloyd's bound" can't break (or inject into) the PowerShell string.
        safe_title = title.replace("'", "''")
        safe_message = message.replace("'", "''")
        ps = (
            "[void][System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
            "$n=New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon=[System.Drawing.SystemIcons]::Information;$n.Visible=$true;"
            f"$n.ShowBalloonTip(10000,'{safe_title}','{safe_message}',"
            "[System.Windows.Forms.ToolTipIcon]::Info);Start-Sleep -Seconds 6;$n.Dispose()"
        )
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass  # notification is non-essential
