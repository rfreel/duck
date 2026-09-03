import sys

from duck_harness.process import run_logged


def test_run_logged_preserves_exit_code_and_streams(tmp_path):
    result = run_logged([sys.executable, "-c", "import sys; print('OUT'); print('ERR', file=sys.stderr); raise SystemExit(7)"], cwd=tmp_path, log_dir=tmp_path / "logs", stem="probe")
    assert result.returncode == 7
    assert result.stdout_path.read_text().strip() == "OUT"
    assert result.stderr_path.read_text().strip() == "ERR"
    result.stdout_path.relative_to(tmp_path / "logs")


def test_run_logged_does_not_use_shell_interpolation(tmp_path):
    marker = tmp_path / "SHOULD_NOT_EXIST"
    arg = f"; touch {marker}"
    result = run_logged([sys.executable, "-c", "import sys; print(sys.argv[1])", arg], cwd=tmp_path, log_dir=tmp_path / "logs", stem="literal")
    assert result.returncode == 0
    assert result.stdout_path.read_text().strip() == arg
    assert not marker.exists()
