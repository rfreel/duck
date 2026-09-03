from __future__ import annotations

from pathlib import Path

from .config import UpstreamConfig
from .model import CheckResult, CommandOutcome, Status
from .paths import HarnessPaths
from .process import run_logged

CANONICAL_REMOTE = "https://github.com/pollen-robotics/microduck_rl.git"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _run(runner, argv, paths, stem, cwd=None):
    return runner(argv, cwd=paths.root if cwd is None else cwd, log_dir=paths.logs_dir, stem=stem)


def observe_checkout_revision(paths: HarnessPaths, runner=run_logged) -> CheckResult:
    if not paths.upstream_dir.exists():
        return CheckResult("upstream-revision", Status.FAIL, True, "upstream checkout is absent; run ./duck setup cpu")
    result = _run(runner, ["git", "-C", str(paths.upstream_dir), "rev-parse", "HEAD"], paths, "rev-parse")
    text = _read(result.stdout_path)
    if result.returncode != 0:
        return CheckResult("upstream-revision", Status.FAIL, True, "could not observe upstream HEAD", result.returncode)
    return CheckResult("upstream-revision", Status.PASS, True, text, 0, {"revision": text})


def ensure_checkout(config: UpstreamConfig, paths: HarnessPaths, runner=run_logged) -> CommandOutcome:
    logs: list[Path] = []; checks: list[CheckResult] = []
    paths.upstream_dir.parent.mkdir(parents=True, exist_ok=True)
    if not paths.upstream_dir.exists():
        commands = [
            (["git", "clone", "--filter=blob:none", "--no-checkout", CANONICAL_REMOTE, str(paths.upstream_dir)], "clone"),
            (["git", "-C", str(paths.upstream_dir), "fetch", "--depth=1", "origin", config.revision], "fetch-pin"),
            (["git", "-C", str(paths.upstream_dir), "checkout", "--detach", config.revision], "checkout-pin"),
            (["git", "-C", str(paths.upstream_dir), "rev-parse", "HEAD"], "rev-parse"),
        ]
        for argv, stem in commands:
            result = _run(runner, argv, paths, stem); logs.extend((result.stdout_path, result.stderr_path))
            if result.returncode != 0:
                checks.append(CheckResult(stem, Status.FAIL, True, f"{stem} failed", result.returncode)); return CommandOutcome(tuple(checks), log_paths=tuple(logs))
        observed = _read(result.stdout_path)
        checks.append(CheckResult("upstream-revision", Status.PASS if observed == config.revision else Status.FAIL, True, f"observed {observed}", evidence={"revision": observed}))
        return CommandOutcome(tuple(checks), observed or None, tuple(logs))
    status_result = _run(runner, ["git", "-C", str(paths.upstream_dir), "status", "--porcelain"], paths, "git-status"); logs.extend((status_result.stdout_path, status_result.stderr_path))
    dirty = _read(status_result.stdout_path)
    if status_result.returncode != 0:
        return CommandOutcome((CheckResult("checkout-clean", Status.FAIL, True, "git status failed", status_result.returncode),), log_paths=tuple(logs))
    if dirty:
        return CommandOutcome((CheckResult("checkout-clean", Status.FAIL, True, f"derived checkout has local changes: {paths.upstream_dir}", evidence={"status": dirty}),), log_paths=tuple(logs))
    checks.append(CheckResult("checkout-clean", Status.PASS, True, "derived checkout is clean"))
    remote_result = _run(runner, ["git", "-C", str(paths.upstream_dir), "remote", "get-url", "origin"], paths, "origin"); logs.extend((remote_result.stdout_path, remote_result.stderr_path)); remote = _read(remote_result.stdout_path)
    if remote_result.returncode != 0 or remote != CANONICAL_REMOTE:
        checks.append(CheckResult("upstream-origin", Status.FAIL, True, f"unexpected origin: {remote or '<unavailable>'}", remote_result.returncode, {"origin": remote})); return CommandOutcome(tuple(checks), log_paths=tuple(logs))
    checks.append(CheckResult("upstream-origin", Status.PASS, True, remote, evidence={"origin": remote}))
    rev_result = _run(runner, ["git", "-C", str(paths.upstream_dir), "rev-parse", "HEAD"], paths, "rev-parse"); logs.extend((rev_result.stdout_path, rev_result.stderr_path)); observed = _read(rev_result.stdout_path)
    if rev_result.returncode != 0:
        checks.append(CheckResult("upstream-revision", Status.FAIL, True, "could not observe HEAD", rev_result.returncode)); return CommandOutcome(tuple(checks), log_paths=tuple(logs))
    if observed != config.revision:
        for argv, stem in [(["git", "-C", str(paths.upstream_dir), "fetch", "--depth=1", "origin", config.revision], "fetch-pin"), (["git", "-C", str(paths.upstream_dir), "checkout", "--detach", config.revision], "checkout-pin"), (["git", "-C", str(paths.upstream_dir), "rev-parse", "HEAD"], "rev-parse-final")]:
            result = _run(runner, argv, paths, stem); logs.extend((result.stdout_path, result.stderr_path))
            if result.returncode != 0:
                checks.append(CheckResult(stem, Status.FAIL, True, f"{stem} failed", result.returncode)); return CommandOutcome(tuple(checks), observed, tuple(logs))
        observed = _read(result.stdout_path)
    checks.append(CheckResult("upstream-revision", Status.PASS if observed == config.revision else Status.FAIL, True, f"observed {observed}", evidence={"revision": observed}))
    return CommandOutcome(tuple(checks), observed or None, tuple(logs))


def verify_environment(paths: HarnessPaths, runner=run_logged) -> CheckResult:
    python = paths.upstream_dir / ".venv" / "bin" / "python"
    if not python.exists():
        return CheckResult("environment", Status.FAIL, True, "upstream Python environment is absent; run ./duck setup cpu")
    result = _run(runner, [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], paths, "venv-python", cwd=paths.upstream_dir)
    version = _read(result.stdout_path)
    if result.returncode == 0 and version == "3.12":
        return CheckResult("environment", Status.PASS, True, f"Python {version}", 0, {"path": str(python), "version": version})
    return CheckResult("environment", Status.FAIL, True, f"expected Python 3.12, observed {version or '<none>'}", result.returncode, {"path": str(python), "version": version})
