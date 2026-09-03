from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessPaths:
    root: Path
    state_dir: Path
    upstream_dir: Path
    receipts_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "HarnessPaths":
        root = root.resolve()
        state = root / ".duck"
        return cls(
            root=root,
            state_dir=state,
            upstream_dir=state / "upstream" / "microduck_rl",
            receipts_dir=state / "receipts",
            logs_dir=state / "logs",
        )
