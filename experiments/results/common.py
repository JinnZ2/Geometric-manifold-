"""Shared helpers for experiments/results/* run scripts (2026-11-27).

Records the environment (python, numpy, scipy, torch versions, CPU, commit SHA) and
writes JSON outputs. No experiment logic lives here.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "UNKNOWN"


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "cpu": cpu_name(), "commit": git_sha()}
    for mod in ("numpy", "scipy", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None
    return env


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=float) + "\n")


def add_repo_to_path() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
