import json

import pytest

from duck_harness.config import load_upstream

PIN = "29e887ecfbf5d37144759e5a9f8a176dfb83d547"


def write_config(path, **overrides):
    data = {
        "schema_version": 1,
        "repository": "pollen-robotics/microduck_rl",
        "revision": PIN,
        "reference_branch": "develop",
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_upstream_accepts_exact_authority_shape(tmp_path):
    path = tmp_path / "upstream.json"
    write_config(path)
    cfg = load_upstream(path)
    assert cfg.revision == PIN
    assert cfg.repository == "pollen-robotics/microduck_rl"


@pytest.mark.parametrize("revision", ["main", "ABCDEF", "f" * 39, "g" * 40])
def test_load_upstream_rejects_non_sha_revision(tmp_path, revision):
    path = tmp_path / "upstream.json"
    write_config(path, revision=revision)
    with pytest.raises(ValueError):
        load_upstream(path)


def test_load_upstream_rejects_extra_authority_fields(tmp_path):
    path = tmp_path / "upstream.json"
    write_config(path, moving_branch="main")
    with pytest.raises(ValueError):
        load_upstream(path)


def test_load_upstream_rejects_missing_fields(tmp_path):
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_upstream(path)


def test_load_upstream_rejects_wrong_repo(tmp_path):
    path = tmp_path / "upstream.json"
    write_config(path, repository="other/repo")
    with pytest.raises(ValueError):
        load_upstream(path)


def test_load_upstream_rejects_wrong_schema(tmp_path):
    path = tmp_path / "upstream.json"
    write_config(path, schema_version=2)
    with pytest.raises(ValueError):
        load_upstream(path)
