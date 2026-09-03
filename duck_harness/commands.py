from __future__ import annotations

from . import probes, upstream
from .config import UpstreamConfig
from .model import CheckResult, CommandOutcome, Status, aggregate_status
from .paths import HarnessPaths
from .process import run_logged


def doctor(paths: HarnessPaths, config: UpstreamConfig) -> CommandOutcome:
    del paths, config
    return CommandOutcome(checks=(probes.probe_host_target(), probes.probe_executable("git", required=True), probes.probe_executable("uv", required=True), probes.probe_python312(required=True), probes.probe_nvidia(required=False)))


def _all_required_pass(checks):
    return aggregate_status(tuple(checks)) is Status.PASS


def setup(paths: HarnessPaths, config: UpstreamConfig, mode: str, runner=run_logged) -> CommandOutcome:
    if mode not in {"cpu", "gpu"}:
        raise ValueError("setup mode must be cpu or gpu")
    checks = [probes.probe_host_target(), probes.probe_executable("git", required=True), probes.probe_executable("uv", required=True), probes.probe_python312(required=True)]
    if mode == "gpu":
        checks.append(probes.probe_nvidia(required=True))
    if not _all_required_pass(checks):
        return CommandOutcome(tuple(checks))
    python_path = next(c.evidence["path"] for c in checks if c.name in {"python3.12", "python"})
    checkout = upstream.ensure_checkout(config, paths, runner=runner)
    checks.extend(checkout.checks); logs = list(checkout.log_paths)
    if checkout.overall_status is not Status.PASS:
        return CommandOutcome(tuple(checks), checkout.upstream_revision, tuple(logs))
    sync = runner(["uv", "sync", "--python", str(python_path), "--frozen"], cwd=paths.upstream_dir, log_dir=paths.logs_dir, stem=f"setup-{mode}-uv-sync")
    logs.extend((sync.stdout_path, sync.stderr_path))
    checks.append(CheckResult("uv-sync", Status.PASS if sync.returncode == 0 else Status.FAIL, True, "frozen upstream environment synchronized" if sync.returncode == 0 else "uv sync failed", sync.returncode))
    if sync.returncode != 0:
        return CommandOutcome(tuple(checks), checkout.upstream_revision, tuple(logs))
    env_check = upstream.verify_environment(paths); checks.append(env_check)
    if env_check.status is not Status.PASS:
        return CommandOutcome(tuple(checks), checkout.upstream_revision, tuple(logs))
    if mode == "gpu":
        torch = runner(["uv", "run", "--frozen", "python", "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"], cwd=paths.upstream_dir, log_dir=paths.logs_dir, stem="torch-cuda")
        logs.extend((torch.stdout_path, torch.stderr_path))
        checks.append(CheckResult("torch-cuda", Status.PASS if torch.returncode == 0 else Status.FAIL, True, "Torch CUDA available" if torch.returncode == 0 else "Torch cannot access CUDA", torch.returncode))
    return CommandOutcome(tuple(checks), checkout.upstream_revision, tuple(logs))
