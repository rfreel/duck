from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from typing import Callable

from .model import CheckResult, Status


def host_snapshot() -> dict[str, object]:
    return {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "python_version": platform.python_version(), "python_executable": sys.executable}


def probe_host_target(*, system: str | None = None, release: str | None = None) -> CheckResult:
    system = platform.system() if system is None else system
    release = platform.release() if release is None else release
    if system == "Linux":
        target = "wsl2" if "microsoft" in release.lower() else "linux"
        return CheckResult("host-target", Status.PASS, True, f"supported target: {target}", evidence={"target": target, "system": system, "release": release})
    target = "native-windows" if system == "Windows" else "macos" if system == "Darwin" else (system.lower() or "unknown")
    return CheckResult("host-target", Status.UNKNOWN, True, f"v1 support is not established for {target}", evidence={"target": target, "system": system, "release": release})


def probe_executable(name: str, *, required: bool = True, which: Callable[[str], str | None] = shutil.which) -> CheckResult:
    path = which(name)
    if path:
        return CheckResult(name, Status.PASS, required, f"found {path}", evidence={"path": path})
    return CheckResult(name, Status.FAIL if required else Status.SKIP, required, f"{name} not found on PATH")


def probe_python312(*, required: bool = True, current_executable: str | None = None, which: Callable[[str], str | None] = shutil.which, run: Callable[..., object] = subprocess.run) -> CheckResult:
    candidates = [current_executable or sys.executable, which("python3.12"), which("python3"), which("python")]
    seen: set[str] = set(); observed: list[dict[str, object]] = []
    for path in candidates:
        if not path or path in seen: continue
        seen.add(path)
        try:
            proc = run([path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture_output=True, text=True, check=False)
        except OSError as exc:
            observed.append({"path": path, "error": str(exc)}); continue
        version = str(getattr(proc, "stdout", "")).strip()
        observed.append({"path": path, "returncode": getattr(proc, "returncode", None), "version": version})
        if getattr(proc, "returncode", 1) == 0 and version == "3.12":
            return CheckResult("python3.12", Status.PASS, required, f"Python 3.12 found at {path}", evidence={"path": path, "version": version})
    return CheckResult("python3.12", Status.FAIL if required else Status.SKIP, required, "no usable Python 3.12 interpreter observed", evidence={"candidates": observed})


def probe_nvidia(*, required: bool = False, which: Callable[[str], str | None] = shutil.which, run: Callable[..., object] = subprocess.run) -> CheckResult:
    path = which("nvidia-smi")
    if not path:
        return CheckResult("nvidia", Status.FAIL if required else Status.SKIP, required, "nvidia-smi not found")
    try:
        proc = run([path, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    except OSError as exc:
        return CheckResult("nvidia", Status.FAIL if required else Status.UNKNOWN, required, f"nvidia-smi could not run: {exc}")
    rc = int(getattr(proc, "returncode", 1)); text = str(getattr(proc, "stdout", "")).strip()
    if rc == 0 and text:
        return CheckResult("nvidia", Status.PASS, required, text, exit_code=0, evidence={"path": path, "gpus": text.splitlines()})
    return CheckResult("nvidia", Status.FAIL if required else Status.UNKNOWN, required, str(getattr(proc, "stderr", "")).strip() or "nvidia-smi returned no usable GPU data", exit_code=rc, evidence={"path": path})
