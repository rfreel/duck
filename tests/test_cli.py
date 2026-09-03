import json

from duck_harness.cli import main
from duck_harness.model import CheckResult, CommandOutcome, Status


def write_authority(root):
    (root / "upstream.json").write_text(json.dumps({"schema_version": 1, "repository": "pollen-robotics/microduck_rl", "revision": "29e887ecfbf5d37144759e5a9f8a176dfb83d547", "reference_branch": "develop"}))


def test_doctor_persists_one_receipt_without_upstream_mutation(tmp_path, monkeypatch):
    write_authority(tmp_path)
    outcome = CommandOutcome((CheckResult("host", Status.PASS, True, "ok"),))
    monkeypatch.setattr("duck_harness.cli.commands.doctor", lambda paths, config: outcome)
    monkeypatch.setattr("duck_harness.cli.probes.host_snapshot", lambda: {"system": "Linux"})
    assert main(["doctor"], root=tmp_path) == 0
    assert len(list((tmp_path / ".duck" / "receipts").glob("*.json"))) == 1
    assert not (tmp_path / ".duck" / "upstream").exists()


def test_unknown_maps_to_exit_2(tmp_path, monkeypatch):
    write_authority(tmp_path)
    outcome = CommandOutcome((CheckResult("host", Status.UNKNOWN, True, "unknown"),))
    monkeypatch.setattr("duck_harness.cli.commands.doctor", lambda paths, config: outcome)
    monkeypatch.setattr("duck_harness.cli.probes.host_snapshot", lambda: {})
    assert main(["doctor"], root=tmp_path) == 2
