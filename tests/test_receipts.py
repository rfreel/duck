import json

import pytest

from duck_harness.model import CheckResult, CommandOutcome, Status
from duck_harness.paths import HarnessPaths
from duck_harness.receipts import read_receipts, write_receipt


def outcome(status=Status.UNKNOWN, log_paths=()):
    return CommandOutcome(checks=(CheckResult("probe", status, True, "detail"),), upstream_revision=None, log_paths=tuple(log_paths))


def test_receipt_preserves_unknown_and_relative_logs(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    log = paths.logs_dir / "x.log"
    log.parent.mkdir(parents=True)
    log.write_text("x", encoding="utf-8")
    receipt = write_receipt(paths, ["doctor"], outcome(Status.UNKNOWN, (log,)), {"system": "Linux"}, "2026-09-03T22:00:00Z", "2026-09-03T22:00:01Z")
    payload = json.loads(receipt.read_text())
    assert payload["overall_status"] == "UNKNOWN"
    assert payload["checks"][0]["status"] == "UNKNOWN"
    assert payload["upstream_revision"] is None
    assert payload["log_paths"] == [".duck/logs/x.log"]


def test_two_receipts_are_distinct(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    a = write_receipt(paths, ["doctor"], outcome(), {}, "a", "b")
    b = write_receipt(paths, ["doctor"], outcome(), {}, "a", "b")
    assert a != b


def test_read_receipts_surfaces_malformed_file(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    paths.receipts_dir.mkdir(parents=True)
    bad = paths.receipts_dir / "bad.json"
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        read_receipts(paths)


def test_read_receipts_rejects_missing_required_fields(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    paths.receipts_dir.mkdir(parents=True)
    bad = paths.receipts_dir / "missing.json"
    bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing.json"):
        read_receipts(paths)
