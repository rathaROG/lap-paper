"""Default output locations for commands run without an explicit output path.

Each interpreter and minute gets one folder,
``benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}``, and every command
writes its own named entry inside it, so commands started in the same minute
never share a directory. Git ignores everything under ``benchmark/runs``.
"""
from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def run_folder(now: dt.datetime | None = None) -> Path:
    """For example ``benchmark/runs/windows-py3.14.10-20261030-1542``."""
    now = now or dt.datetime.now()
    return RUNS / f"{platform.system().lower()}-py{platform.python_version()}-{now:%Y%m%d-%H%M}"


def default_output(name: str, now: dt.datetime | None = None) -> Path:
    """One command's output entry inside the current run folder."""
    return run_folder(now) / name
