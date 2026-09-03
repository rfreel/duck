from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .model import CommandOutcome
from .paths import HarnessPaths

_SCHEMA_FIELDS = {"schema_version","command","upstream_revision","started_at","ended_at","host","checks","overall_status","log_paths"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def time_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _relative(paths: HarnessPaths, path: Path) -> str:
    return path.resolve().relative_to(paths.root).as_posix()


def write_receipt(paths: HarnessPaths, command: Sequence[str], outcome: CommandOutcome, host: dict[str, object], started_at: str, ended_at: str) -> Path:
    paths.receipts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "command": list(command),
        "upstream_revision": outcome.upstream_revision,
        "started_at": started_at,
        "ended_at": ended_at,
        "host": host,
        "checks": [{"name": c.name, "status": c.status.value, "required": c.required, "detail": c.detail, "exit_code": c.exit_code, "evidence": c.evidence} for c in outcome.checks],
        "overall_status": outcome.overall_status.value,
        "log_paths": [_relative(paths, p) for p in outcome.log_paths],
    }
    name = f"{time_key()}-{os.getpid()}-{next(tempfile._get_candidate_names())}.json"
    final = paths.receipts_dir / name
    fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.", dir=paths.receipts_dir, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, final)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return final


def _validate_receipt(path: Path, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _SCHEMA_FIELDS:
        raise ValueError(f"malformed receipt {path.name}: unexpected fields")
    if payload.get("schema_version") != 1:
        raise ValueError(f"malformed receipt {path.name}: unsupported schema")
    if not isinstance(payload.get("command"), list) or not isinstance(payload.get("checks"), list):
        raise ValueError(f"malformed receipt {path.name}: invalid command/checks")
    return payload


def read_receipts(paths: HarnessPaths) -> list[dict[str, Any]]:
    if not paths.receipts_dir.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(paths.receipts_dir.glob("*.json")):
        try:
            receipts.append(_validate_receipt(path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            if isinstance(exc, ValueError) and path.name in str(exc):
                raise
            raise ValueError(f"malformed receipt {path.name}: {exc}") from exc
    return receipts
