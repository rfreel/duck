from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import commands, probes
from .config import load_upstream
from .model import Status
from .paths import HarnessPaths
from .receipts import utc_now, write_receipt

_EXIT = {Status.PASS: 0, Status.FAIL: 1, Status.UNKNOWN: 2, Status.SKIP: 2}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="duck", description="Guided MicroDuck test harness")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="inspect prerequisites without installing or fetching")
    setup = sub.add_parser("setup", help="prepare pinned upstream environment")
    setup.add_argument("mode", choices=("cpu", "gpu"))
    return parser


def main(argv: Sequence[str] | None = None, *, root: Path | None = None) -> int:
    args = _parser().parse_args(argv)
    root = (Path.cwd() if root is None else root).resolve()
    paths = HarnessPaths.from_root(root)
    config = load_upstream(root / "upstream.json")
    started_at = utc_now(); host = probes.host_snapshot()
    if args.command == "doctor":
        outcome = commands.doctor(paths, config); command = ["doctor"]
    elif args.command == "setup":
        outcome = commands.setup(paths, config, args.mode); command = ["setup", args.mode]
    else:
        raise AssertionError(args.command)
    ended_at = utc_now()
    receipt = write_receipt(paths, command, outcome, host, started_at, ended_at)
    for check in outcome.checks:
        print(f"{check.status.value:<7}  {check.name}  {check.detail}")
    print(f"overall: {outcome.overall_status.value}")
    print(f"receipt: {receipt.relative_to(root).as_posix()}")
    return _EXIT[outcome.overall_status]
