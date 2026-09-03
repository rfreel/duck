from types import SimpleNamespace

from duck_harness.model import Status
from duck_harness import probes


def test_linux_target_passes():
    r = probes.probe_host_target(system="Linux", release="6.8")
    assert r.status is Status.PASS and r.evidence["target"] == "linux"


def test_wsl2_target_passes():
    r = probes.probe_host_target(system="Linux", release="5.15.0-microsoft-standard-WSL2")
    assert r.status is Status.PASS and r.evidence["target"] == "wsl2"


def test_windows_is_unknown():
    r = probes.probe_host_target(system="Windows", release="11")
    assert r.status is Status.UNKNOWN and r.evidence["target"] == "native-windows"


def test_macos_is_unknown():
    r = probes.probe_host_target(system="Darwin", release="25")
    assert r.status is Status.UNKNOWN and r.evidence["target"] == "macos"


def test_missing_required_executable_fails():
    assert probes.probe_executable("git", which=lambda _: None).status is Status.FAIL


def fake_run_version(version):
    return lambda argv, **kwargs: SimpleNamespace(returncode=0, stdout=version + "\n", stderr="")


def test_python312_passes_with_observed_path():
    r = probes.probe_python312(current_executable="/x/current", which=lambda n: None, run=fake_run_version("3.12"))
    assert r.status is Status.PASS and r.evidence["path"] == "/x/current"


def test_python311_only_fails():
    assert probes.probe_python312(current_executable="/x/current", which=lambda n: None, run=fake_run_version("3.11")).status is Status.FAIL


def test_missing_optional_nvidia_is_skip():
    assert probes.probe_nvidia(which=lambda _: None).status is Status.SKIP


def test_failing_optional_nvidia_is_unknown():
    run = lambda argv, **kwargs: SimpleNamespace(returncode=9, stdout="", stderr="driver error")
    assert probes.probe_nvidia(which=lambda _: "/usr/bin/nvidia-smi", run=run).status is Status.UNKNOWN
