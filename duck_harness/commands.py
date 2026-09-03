from __future__ import annotations

from . import probes
from .config import UpstreamConfig
from .model import CommandOutcome
from .paths import HarnessPaths


def doctor(paths: HarnessPaths, config: UpstreamConfig) -> CommandOutcome:
    del paths, config
    return CommandOutcome(checks=(
        probes.probe_host_target(),
        probes.probe_executable("git", required=True),
        probes.probe_executable("uv", required=True),
        probes.probe_python312(required=True),
        probes.probe_nvidia(required=False),
    ))
