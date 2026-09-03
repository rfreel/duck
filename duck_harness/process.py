from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path


def run_logged(
    argv: Sequence[str],
    *,
    cwd: Path,
    log_dir: Path,
    stem: str,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", stem).strip("-") or "command"
    suffix = time.time_ns()
    stdout_path = log_dir / f"{safe}-{suffix}.stdout.log"
    stderr_path = log_dir / f"{safe}-{suffix}.stderr.log"
    child_env = None if env is None else {**os.environ, **dict(env)}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(
            list(argv), cwd=cwd, stdout=stdout, stderr=stderr, check=False,
            env=child_env, shell=False, text=True,
        )
    return ProcessResult(tuple(argv), proc.returncode, stdout_path, stderr_path)
