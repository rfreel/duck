from duck_harness.paths import HarnessPaths


def test_all_mutable_harness_paths_are_confined_to_dot_duck(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    for path in (paths.state_dir, paths.upstream_dir, paths.receipts_dir, paths.logs_dir):
        path.relative_to(tmp_path / ".duck")
