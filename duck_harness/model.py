from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    required: bool
    detail: str
    exit_code: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandOutcome:
    checks: tuple[CheckResult, ...]
    upstream_revision: str | None = None
    log_paths: tuple[Path, ...] = ()

    @property
    def overall_status(self) -> Status:
        return aggregate_status(self.checks)


def aggregate_status(checks: Sequence[CheckResult]) -> Status:
    required = [check for check in checks if check.required]
    if any(check.status is Status.FAIL for check in required):
        return Status.FAIL
    if any(check.status in (Status.UNKNOWN, Status.SKIP) for check in required):
        return Status.UNKNOWN
    if required and all(check.status is Status.PASS for check in required):
        return Status.PASS
    return Status.UNKNOWN
