from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_REQUIRED = {"schema_version", "repository", "revision", "reference_branch"}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_AUTHORITY_REPO = "pollen-robotics/microduck_rl"


@dataclass(frozen=True)
class UpstreamConfig:
    schema_version: int
    repository: str
    revision: str
    reference_branch: str


def load_upstream(path: Path) -> UpstreamConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _REQUIRED:
        raise ValueError(f"upstream authority fields must be exactly {sorted(_REQUIRED)}")
    if data["schema_version"] != 1:
        raise ValueError("unsupported upstream schema_version")
    if data["repository"] != _AUTHORITY_REPO:
        raise ValueError(f"repository must be {_AUTHORITY_REPO}")
    revision = data["revision"]
    if not isinstance(revision, str) or not _SHA_RE.fullmatch(revision):
        raise ValueError("revision must be a lowercase 40-character git SHA")
    branch = data["reference_branch"]
    if not isinstance(branch, str) or not branch:
        raise ValueError("reference_branch must be a non-empty string")
    return UpstreamConfig(1, _AUTHORITY_REPO, revision, branch)
