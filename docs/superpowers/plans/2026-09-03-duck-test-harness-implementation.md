# Duck test harness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use the checkbox steps as the execution ledger. Do not skip the failing-test witness before implementation.

**Goal:** Build `rfreel/duck` into a thin, reproducible harness that guides a user from machine diagnosis through pinned MicroDuck CPU tests, headless MuJoCo simulation, and a bounded CUDA training smoke test while preserving PASS/FAIL/SKIP/UNKNOWN receipts.

**Architecture:** `duck` owns only orchestration, probes, logs, receipts, and documentation. The simulator/training implementation remains the pinned `pollen-robotics/microduck_rl` checkout under `.duck/upstream/microduck_rl`. A small Python standard-library package implements the harness; a POSIX `./duck` wrapper is the front door. Test commands never contact robot hardware. Each command returns explicit checks, the CLI persists one receipt, and `status` reconstructs state only from receipts.

**Tech stack:** Python 3.10+ for the harness controller; Python 3.12 for pinned upstream; Python standard library; pytest for harness tests; git; uv; upstream MuJoCo/mjlab/rsl_rl; NVIDIA CUDA only for `train-smoke`; GitHub Actions on Ubuntu for default CI.

**Spec:** `docs/superpowers/specs/2026-09-03-duck-test-harness-design.md`

## Verified implementation inputs

These are implementation facts established from the current upstream source, not assumptions:

- Authority repo: `pollen-robotics/microduck_rl`.
- Immutable pin: `29e887ecfbf5d37144759e5a9f8a176dfb83d547`.
- Human reference branch: `develop`.
- At that pin, `pyproject.toml` requires Python `>=3.12,<3.13`; `uv.lock` requires Python `==3.12.*`.
- Upstream documents `uv run --with pytest pytest tests/` as CPU-only config/reward regression tests.
- `tests/test_nan_guard.py` and `tests/test_obs_nan_guard.py` exist at the pin and are suitable as the cheapest focused CPU regression rung.
- Training uses MuJoCo Warp/CUDA; upstream supports `--env.scene.num-envs` and `--agent.max_iterations`.
- Upstream project plans use `--agent.logger tensorboard` for disposable training smoke runs, avoiding a W&B login requirement.
- `uv.lock` pins `rsl-rl-lib==5.0.1`. Its logger prints `Learning iteration {it}/{total_it}`, so the training witness can use the exact regex `Learning iteration\s+(\d+)/(\d+)`.
- `src/mjlab_microduck/robot/microduck/scene.xml` exists and contains the MicroDuck ground-contact model, a floor, named keyframes, and 14 actuators. A headless CPU MuJoCo probe can therefore establish model load plus simulation-time advancement without an ONNX policy or viewer.
- The upstream documentation does not establish native Windows or macOS as supported execution targets. Harness v1 supports Linux and WSL2 as Linux-kernel targets. Native Windows/macOS remain UNKNOWN and must not be upgraded to supported without an independent validation change.
- “CPU-safe” means no GPU hardware is required to execute that test rung. It does **not** mean the upstream dependency installation is CUDA-free or small; the pinned environment can still resolve CUDA-bearing wheels.

## Global constraints

- `upstream.json` is authority. `.duck/upstream/microduck_rl` is derived state.
- Never follow `develop` implicitly after the checkout is created. Every setup/test command verifies the observed commit equals the immutable pin.
- Never discard a dirty derived checkout silently. Report FAIL and the path; do not reset user-visible changes.
- `doctor` performs no install, fetch, checkout, configuration, or external-state mutation. Its only write is the required append-only receipt under `.duck/receipts/`.
- No `test` command may run setup implicitly. Missing setup is a witnessed precondition failure with `./duck setup cpu` or `./duck setup gpu` as the next action.
- No command in this plan invokes `robotctl`, `duckctl`, serial buses, Bluetooth robot control, flashing, policy deployment, or physical hardware control.
- Overall PASS requires every required check to PASS. Required FAIL dominates; otherwise required UNKNOWN or SKIP produces UNKNOWN.
- Optional missing GPU capability is SKIP during `doctor`; GPU capability is required and therefore FAIL for `setup gpu` and `test train-smoke`.
- Use `subprocess` argument arrays only. Never use `shell=True` for harness-controlled commands.
- Paths written by the harness are confined to `.duck/` and the disposable upstream checkout, except for normal uv cache behavior outside the repository.
- Default CI runs harness tests plus upstream CPU unit/smoke checks only. It does not run simulation, CUDA training, Hugging Face Jobs, or hardware tests.
- Each task below ends with a commit. Do not combine tasks into one large commit.

---

## Task 1: Establish authority, status semantics, and path confinement

**Files:**
- Create: `upstream.json`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `duck_harness/__init__.py`
- Create: `duck_harness/model.py`
- Create: `duck_harness/config.py`
- Create: `duck_harness/paths.py`
- Create: `tests/test_model.py`
- Create: `tests/test_config.py`
- Create: `tests/test_paths.py`

**Interfaces:**
- `Status`: `PASS | FAIL | SKIP | UNKNOWN`
- `CheckResult(name, status, required, detail, exit_code, evidence)`
- `CommandOutcome(checks, upstream_revision, log_paths)`
- `aggregate_status(checks) -> Status`
- `UpstreamConfig(schema_version, repository, revision, reference_branch)`
- `load_upstream(path) -> UpstreamConfig`
- `HarnessPaths.from_root(root) -> HarnessPaths`

- [ ] **Step 1: Write the failing status/config/path tests**

Create `tests/test_model.py` with these cases:

```python
from duck_harness.model import CheckResult, Status, aggregate_status


def check(status: Status, required: bool = True) -> CheckResult:
    return CheckResult("x", status, required, status.value)


def test_required_passes_aggregate_to_pass():
    assert aggregate_status((check(Status.PASS), check(Status.PASS))) is Status.PASS


def test_required_fail_dominates_unknown():
    assert aggregate_status((check(Status.UNKNOWN), check(Status.FAIL))) is Status.FAIL


def test_required_unknown_is_preserved():
    assert aggregate_status((check(Status.PASS), check(Status.UNKNOWN))) is Status.UNKNOWN


def test_optional_skip_does_not_block_pass():
    assert aggregate_status((check(Status.PASS), check(Status.SKIP, required=False))) is Status.PASS


def test_required_skip_is_not_silently_passed():
    assert aggregate_status((check(Status.SKIP),)) is Status.UNKNOWN
```

Create `tests/test_config.py`:

```python
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
```

Create `tests/test_paths.py`:

```python
from duck_harness.paths import HarnessPaths


def test_all_mutable_harness_paths_are_confined_to_dot_duck(tmp_path):
    paths = HarnessPaths.from_root(tmp_path)
    for path in (paths.state_dir, paths.upstream_dir, paths.receipts_dir, paths.logs_dir):
        path.relative_to(tmp_path / ".duck")
```

- [ ] **Step 2: Run tests and confirm import failures**

```bash
python -m pytest tests/test_model.py tests/test_config.py tests/test_paths.py -q
```

Expected witness: collection fails because `duck_harness.model`, `duck_harness.config`, and `duck_harness.paths` do not exist.

- [ ] **Step 3: Add the exact upstream authority file**

Create `upstream.json`:

```json
{
  "schema_version": 1,
  "repository": "pollen-robotics/microduck_rl",
  "revision": "29e887ecfbf5d37144759e5a9f8a176dfb83d547",
  "reference_branch": "develop"
}
```

Create `.gitignore`:

```gitignore
.duck/
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "duck-harness"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
test = ["pytest>=8,<9"]

[tool.setuptools.packages.find]
include = ["duck_harness*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Implement the status model**

Create `duck_harness/model.py` with this contract:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    required: bool
    detail: str
    exit_code: int | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandOutcome:
    checks: tuple[CheckResult, ...]
    upstream_revision: str | None = None
    log_paths: tuple[Path, ...] = ()

    @property
    def overall_status(self) -> Status:
        return aggregate_status(self.checks)


def aggregate_status(checks: Sequence[CheckResult]) -> Status:
    required = [check for check in checks if check.required]
    if any(check.status is Status.FAIL for check in required):
        return Status.FAIL
    if any(check.status in (Status.UNKNOWN, Status.SKIP) for check in required):
        return Status.UNKNOWN
    if required and all(check.status is Status.PASS for check in required):
        return Status.PASS
    return Status.UNKNOWN
```

- [ ] **Step 5: Implement strict authority parsing and paths**

`duck_harness/config.py` must reject missing fields, extra fields, non-lowercase/non-40-character SHA values, schema versions other than `1`, and repositories other than `pollen-robotics/microduck_rl`.

Use this validation shape:

```python
_REQUIRED = {"schema_version", "repository", "revision", "reference_branch"}
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
```

`duck_harness/paths.py` must define:

```python
@dataclass(frozen=True)
class HarnessPaths:
    root: Path
    state_dir: Path
    upstream_dir: Path
    receipts_dir: Path
    logs_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "HarnessPaths":
        root = root.resolve()
        state = root / ".duck"
        return cls(
            root=root,
            state_dir=state,
            upstream_dir=state / "upstream" / "microduck_rl",
            receipts_dir=state / "receipts",
            logs_dir=state / "logs",
        )
```

- [ ] **Step 6: Run the focused tests**

```bash
python -m pytest tests/test_model.py tests/test_config.py tests/test_paths.py -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add upstream.json .gitignore pyproject.toml duck_harness tests

git commit -m "feat: establish duck harness authority model"
```

---

## Task 2: Add deterministic subprocess logging and append-only receipts

**Files:**
- Create: `duck_harness/process.py`
- Create: `duck_harness/receipts.py`
- Create: `tests/test_process.py`
- Create: `tests/test_receipts.py`

**Interfaces:**
- `ProcessResult(argv, returncode, stdout_path, stderr_path)`
- `run_logged(argv, cwd, log_dir, stem, env=None) -> ProcessResult`
- `utc_now() -> str`
- `write_receipt(paths, command, outcome, host, started_at, ended_at) -> Path`
- `read_receipts(paths) -> list[dict]`

- [ ] **Step 1: Write failing subprocess and receipt tests**

`tests/test_process.py` must prove stdout/stderr separation, exit-code preservation, no shell interpolation, and log confinement. Use `sys.executable` as the child process so the test is portable.

```python
import sys

from duck_harness.process import run_logged


def test_run_logged_preserves_exit_code_and_streams(tmp_path):
    result = run_logged(
        [sys.executable, "-c", "import sys; print('OUT'); print('ERR', file=sys.stderr); raise SystemExit(7)"],
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        stem="probe",
    )
    assert result.returncode == 7
    assert result.stdout_path.read_text().strip() == "OUT"
    assert result.stderr_path.read_text().strip() == "ERR"
    result.stdout_path.relative_to(tmp_path / "logs")
```

`tests/test_receipts.py` must assert:

- receipt JSON contains exactly the observed status values;
- `UNKNOWN` stays `UNKNOWN` after serialization/deserialization;
- `upstream_revision` may be `null` before a checkout exists;
- log paths are stored relative to the repo root;
- two writes produce two distinct files;
- a malformed receipt is surfaced by `read_receipts` instead of silently ignored.

- [ ] **Step 2: Run and confirm missing-module failures**

```bash
python -m pytest tests/test_process.py tests/test_receipts.py -q
```

- [ ] **Step 3: Implement logged subprocess execution**

`duck_harness/process.py` must use `subprocess.run(argv, ...)` with an argument list, `check=False`, and file handles for stdout/stderr. Create log files under the provided `log_dir`; do not invoke a shell.

Use a collision-resistant stem suffix based on `time.time_ns()` so multiple commands in one second cannot overwrite each other.

- [ ] **Step 4: Implement receipt serialization and atomic writes**

Receipt schema version is `1`. Persist this shape:

```json
{
  "schema_version": 1,
  "command": ["doctor"],
  "upstream_revision": null,
  "started_at": "2026-09-03T22:00:00.000000Z",
  "ended_at": "2026-09-03T22:00:00.010000Z",
  "host": {},
  "checks": [],
  "overall_status": "UNKNOWN",
  "log_paths": []
}
```

`write_receipt` must:

1. create `.duck/receipts/` only when the receipt is being persisted;
2. build the payload from `CommandOutcome` without altering any status;
3. convert log paths to repository-relative POSIX strings;
4. write a temporary file in the same directory;
5. call `os.replace(temp_path, final_path)` for atomic publication.

`read_receipts` must sort by filename and raise `ValueError` naming the malformed file if parsing or required-field validation fails.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_process.py tests/test_receipts.py -q
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add duck_harness/process.py duck_harness/receipts.py tests/test_process.py tests/test_receipts.py

git commit -m "feat: add execution logs and immutable receipts"
```

---

## Task 3: Build `doctor` and the executable front door

**Files:**
- Create: `duck_harness/probes.py`
- Create: `duck_harness/commands.py`
- Create: `duck_harness/cli.py`
- Create: `duck_harness/__main__.py`
- Create: `duck`
- Create: `tests/test_probes.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- `host_snapshot() -> dict[str, object]`
- `probe_host_target() -> CheckResult`
- `probe_executable(name, required=True) -> CheckResult`
- `probe_python312(required=True) -> CheckResult`
- `probe_nvidia(required=False) -> CheckResult`
- `doctor(paths, config) -> CommandOutcome`
- `main(argv=None) -> int`

- [ ] **Step 1: Write failing probe tests for all materially distinct host paths**

`tests/test_probes.py` must cover at least these host cases:

1. native Linux → PASS, evidence target `linux`;
2. WSL2-style Linux release containing `microsoft` → PASS, evidence target `wsl2`;
3. native Windows → required UNKNOWN;
4. macOS → required UNKNOWN.

Also test:

- missing `git` → FAIL when required;
- missing `uv` → FAIL when required;
- Python 3.12 found through `sys.executable` or an executable candidate → PASS with exact executable path in evidence;
- only Python 3.11 available → FAIL;
- missing `nvidia-smi` in optional doctor mode → SKIP;
- present `nvidia-smi` returning nonzero → UNKNOWN when optional, not fabricated PASS.

Inject `system`, `release`, executable lookup, and subprocess runner into probe helpers so these tests do not depend on the developer machine.

- [ ] **Step 2: Run the tests and confirm failure**

```bash
python -m pytest tests/test_probes.py tests/test_cli.py -q
```

- [ ] **Step 3: Implement read-only probes**

`probe_host_target` must use this decision table:

| Observed host | Status | Evidence `target` |
|---|---|---|
| `platform.system() == "Linux"` and release contains `microsoft` | PASS | `wsl2` |
| `platform.system() == "Linux"` otherwise | PASS | `linux` |
| Windows | UNKNOWN | `native-windows` |
| macOS | UNKNOWN | `macos` |
| any other system | UNKNOWN | lowercase system name |

`probe_python312` checks these candidates without installing anything:

1. `sys.executable`;
2. `python3.12` from `PATH`;
3. `python3` from `PATH`;
4. `python` from `PATH`.

For each unique executable, run:

```bash
EXE -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
```

PASS only on exactly `3.12`.

`probe_nvidia` runs only when `nvidia-smi` exists:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

- [ ] **Step 4: Implement `doctor`**

Required doctor checks:

- supported harness host target;
- `git` executable;
- `uv` executable;
- Python 3.12 interpreter availability.

Optional doctor check:

- NVIDIA GPU visibility.

Do not create upstream or log directories. The CLI may create only `.duck/receipts/` when it persists the doctor receipt.

- [ ] **Step 5: Implement the CLI receipt boundary**

`duck_harness/cli.py` owns timing and receipt persistence exactly once per command. For `doctor`:

1. capture `started_at`;
2. call `commands.doctor`;
3. capture `ended_at`;
4. persist one receipt;
5. print each check as `STATUS  name  detail`;
6. print `overall: STATUS` and the receipt path;
7. return exit code `0` for PASS, `1` for FAIL, `2` for UNKNOWN.

Do not map UNKNOWN to success.

- [ ] **Step 6: Add the POSIX front door and executable bit**

Create `duck`:

```sh
#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
cd "$ROOT"
exec "$PYTHON" -m duck_harness "$@"
```

Set the git mode to executable:

```bash
chmod +x duck
git update-index --chmod=+x duck
```

The executable bit is an acceptance requirement because the documented interface is `./duck`, not `sh duck`.

- [ ] **Step 7: Run focused tests and a real local doctor**

```bash
python -m pytest tests/test_probes.py tests/test_cli.py -q
./duck doctor
```

On a host lacking one required prerequisite, `./duck doctor` is expected to return FAIL or UNKNOWN with a specific witness; that is a valid test of the reporting path. Confirm a receipt exists and no `.duck/upstream/` directory was created.

- [ ] **Step 8: Commit**

```bash
git add duck duck_harness tests

git commit -m "feat: add read-only duck doctor"
```

---

## Task 4: Fetch the immutable upstream checkout and create the pinned environment

**Files:**
- Create: `duck_harness/upstream.py`
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Modify: `duck_harness/probes.py`
- Create: `tests/test_upstream.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- `observe_checkout_revision(paths) -> CheckResult`
- `ensure_checkout(config, paths, runner=run_logged) -> CommandOutcome`
- `verify_environment(paths) -> CheckResult`
- `setup(paths, config, mode: str) -> CommandOutcome`

- [ ] **Step 1: Write failing checkout/setup tests**

Test the command sequence with a fake `run_logged` function. Required cases:

1. absent checkout issues a clone/fetch/checkout sequence for the immutable SHA;
2. an observed exact SHA is PASS;
3. a different observed SHA is FAIL, never accepted because it is on `develop`;
4. a dirty checkout is FAIL and no checkout/reset command follows;
5. wrong `origin` URL is FAIL;
6. setup stops before network mutation if host/git/uv/Python 3.12 prerequisites fail;
7. `setup cpu` does not require NVIDIA;
8. `setup gpu` requires NVIDIA and, after sync, requires `torch.cuda.is_available()` to be true;
9. failed `uv sync` preserves its nonzero exit code in the check.

- [ ] **Step 2: Implement immutable checkout verification**

Canonical remote URL:

```text
https://github.com/pollen-robotics/microduck_rl.git
```

For an absent checkout, use these exact operations, with every subprocess logged:

```bash
git clone --filter=blob:none --no-checkout https://github.com/pollen-robotics/microduck_rl.git .duck/upstream/microduck_rl
git -C .duck/upstream/microduck_rl fetch --depth=1 origin 29e887ecfbf5d37144759e5a9f8a176dfb83d547
git -C .duck/upstream/microduck_rl checkout --detach 29e887ecfbf5d37144759e5a9f8a176dfb83d547
git -C .duck/upstream/microduck_rl rev-parse HEAD
```

For an existing checkout, first run:

```bash
git -C .duck/upstream/microduck_rl status --porcelain
git -C .duck/upstream/microduck_rl remote get-url origin
git -C .duck/upstream/microduck_rl rev-parse HEAD
```

If status is non-empty, stop with FAIL. If origin differs from the canonical remote, stop with FAIL. If HEAD differs and the checkout is clean, fetch and detach-checkout the immutable SHA, then re-observe HEAD. PASS only on exact equality.

- [ ] **Step 3: Implement environment readiness and setup**

The environment lives at:

```text
.duck/upstream/microduck_rl/.venv
```

Use the Python 3.12 executable returned by `probe_python312`; do not invent a new interpreter path. Run from the upstream checkout:

```bash
uv sync --python /observed/python3.12/path --frozen
```

In code, `/observed/python3.12/path` is the exact `evidence["path"]` from the passing probe, passed as a subprocess argument rather than string interpolation through a shell.

`verify_environment` requires `.venv/bin/python` on Linux/WSL2 and verifies:

```bash
.venv/bin/python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
```

The observed output must be `3.12`.

For `setup gpu`, add both GPU checks:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uv run --frozen python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

`setup cpu` omits both GPU requirements.

- [ ] **Step 4: Add CLI shape**

Support:

```bash
./duck setup cpu
./duck setup gpu
```

Accepted modes are exactly `cpu` and `gpu`; argparse rejects other values before any mutation.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_upstream.py tests/test_cli.py -q
```

Then, on a disposable Linux/WSL2 development environment:

```bash
./duck setup cpu
```

Verify the receipt's `upstream_revision` equals the pin and `git -C .duck/upstream/microduck_rl rev-parse HEAD` prints the same SHA.

- [ ] **Step 6: Commit**

```bash
git add duck_harness tests

git commit -m "feat: pin and prepare microduck upstream"
```

---

## Task 5: Add focused and full upstream CPU test rungs

**Files:**
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Create: `tests/test_cpu_tiers.py`

**Interfaces:**
- `test_unit(paths, config, runner=run_logged) -> CommandOutcome`
- `test_smoke(paths, config, runner=run_logged) -> CommandOutcome`

- [ ] **Step 1: Write failing tier tests**

With an injected fake runner, assert the exact argv for each tier.

Focused unit rung:

```text
uv run --frozen --with pytest pytest -q tests/test_nan_guard.py tests/test_obs_nan_guard.py
```

Full CPU smoke rung:

```text
uv run --frozen --with pytest pytest -q tests/
```

Both commands run with `cwd=.duck/upstream/microduck_rl`.

Test these failures separately:

- checkout absent;
- checkout SHA mismatch;
- environment absent;
- pytest process returns nonzero.

The command must not call setup in any of those cases.

- [ ] **Step 2: Implement explicit readiness checks**

Before either test subprocess:

1. verify exact upstream SHA;
2. verify `.venv/bin/python` reports Python 3.12;
3. if either fails, return immediately with the next action `./duck setup cpu` in the check detail.

Do not call `uv sync` from a test command.

- [ ] **Step 3: Implement the two CPU tiers**

A subprocess exit code of `0` yields PASS for the test check. Any nonzero exit code yields FAIL with the numeric exit code and log paths in the receipt. Do not parse pytest text to upgrade a nonzero process.

- [ ] **Step 4: Add CLI commands**

Support exactly:

```bash
./duck test unit
./duck test smoke
```

- [ ] **Step 5: Verify**

```bash
python -m pytest tests/test_cpu_tiers.py tests/test_cli.py -q
./duck test unit
./duck test smoke
```

A real PASS on the latter two is evidence of the pinned upstream CPU regression surface only; it is not evidence of simulation, training, or hardware behavior.

- [ ] **Step 6: Commit**

```bash
git add duck_harness tests

git commit -m "feat: add upstream cpu test ladder"
```

---

## Task 6: Add a policy-free headless MuJoCo simulation witness

**Files:**
- Create: `scripts/probes/sim_step.py`
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Create: `tests/test_sim_tier.py`

**Interface:**
- `test_sim(paths, config, runner=run_logged) -> CommandOutcome`

- [ ] **Step 1: Write failing simulation-tier tests**

Use a fake runner whose stdout file contains one JSON object. Required PASS payload:

```json
{
  "steps": 50,
  "initial_time": 0.0,
  "final_time": 0.1,
  "nq": 21,
  "nu": 14
}
```

Test independent FAIL cases for:

- subprocess nonzero;
- invalid JSON;
- `final_time <= initial_time`;
- `steps != 50`;
- `nu != 14`;
- upstream SHA mismatch;
- missing environment.

- [ ] **Step 2: Create the deterministic MuJoCo probe**

Create `scripts/probes/sim_step.py`:

```python
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco


def main() -> int:
    scene = Path(sys.argv[1]).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    initial_time = float(data.time)
    steps = 50
    for _ in range(steps):
        mujoco.mj_step(model, data)
    payload = {
        "steps": steps,
        "initial_time": initial_time,
        "final_time": float(data.time),
        "nq": int(model.nq),
        "nu": int(model.nu),
    }
    print(json.dumps(payload, sort_keys=True))
    if payload["final_time"] <= payload["initial_time"]:
        return 1
    if payload["nu"] != 14:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Implement the harness command**

Run from the upstream checkout:

```text
uv run --frozen python ABSOLUTE_DUCK_ROOT/scripts/probes/sim_step.py ABSOLUTE_UPSTREAM_ROOT/src/mjlab_microduck/robot/microduck/scene.xml
```

The harness must parse the final non-empty stdout line as JSON and independently verify the same acceptance conditions as the probe. Record the numeric payload as receipt evidence.

This tier intentionally has no viewer and no policy file. Its claim is limited to: **the pinned MicroDuck MJCF loads under CPU MuJoCo and physics time advances for 50 steps with 14 actuators.**

- [ ] **Step 4: Add CLI command and tests**

```bash
./duck test sim
python -m pytest tests/test_sim_tier.py tests/test_cli.py -q
```

- [ ] **Step 5: Run a real simulation witness**

After `./duck setup cpu`:

```bash
./duck test sim
```

Inspect the receipt. PASS requires `final_time > initial_time` and `nu == 14` in recorded evidence.

- [ ] **Step 6: Commit**

```bash
git add scripts/probes/sim_step.py duck_harness tests

git commit -m "feat: add headless microduck simulation witness"
```

---

## Task 7: Add the bounded CUDA training-path witness

**Files:**
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Create: `tests/test_train_tier.py`

**Interfaces:**
- `parse_learning_iterations(text: str) -> list[tuple[int, int]]`
- `test_train_smoke(paths, config, runner=run_logged) -> CommandOutcome`

- [ ] **Step 1: Write failing iteration-parser tests**

Use the exact rsl_rl 5.0.1 console form:

```python
from duck_harness.commands import parse_learning_iterations


def test_parse_learning_iterations_uses_pinned_rsl_rl_format():
    text = "\x1b[1m Learning iteration 0/5 \x1b[0m\n Learning iteration 4/5 "
    assert parse_learning_iterations(text) == [(0, 5), (4, 5)]


def test_no_iteration_witness_is_empty_not_assumed():
    assert parse_learning_iterations("trainer exited") == []
```

The parser regex is exactly:

```python
re.compile(r"Learning iteration\s+(\d+)/(\d+)")
```

- [ ] **Step 2: Write failing training command tests**

Required preconditions:

1. exact pinned checkout;
2. ready Python 3.12 environment;
3. passing `nvidia-smi` probe;
4. passing in-environment Torch CUDA probe.

Torch CUDA probe:

```text
uv run --frozen python -c import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))
```

Represent the `-c` program as one argv element in code.

Exact bounded training argv:

```text
uv run --frozen train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard
```

Test separate failures for missing GPU, Torch CUDA false/nonzero, trainer nonzero, and trainer exit `0` with no iteration witness.

- [ ] **Step 3: Implement the training witness**

PASS requires **both**:

- bounded trainer exit code `0`;
- parsed iteration evidence contains at least one tuple and the maximum observed iteration index is at least `4` with total `5`.

This prevents “process started” from being mislabeled as “training advanced through the bounded run.” Record the observed tuples in receipt evidence.

Do not inspect reward value as a quality criterion. Do not claim the resulting policy is useful or transferable.

- [ ] **Step 4: Add CLI command**

```bash
./duck test train-smoke
```

The command must return FAIL rather than SKIP when CUDA is missing because the user explicitly selected a GPU-required rung.

- [ ] **Step 5: Verify unit behavior**

```bash
python -m pytest tests/test_train_tier.py tests/test_cli.py -q
```

On a compatible CUDA machine after `./duck setup gpu`, run:

```bash
./duck test train-smoke
```

If no CUDA host is available during implementation, mark the live witness NOT RUN; do not convert the unit-level parser/command tests into a claim that the GPU path was executed.

- [ ] **Step 6: Commit**

```bash
git add duck_harness tests

git commit -m "feat: add bounded cuda training smoke test"
```

---

## Task 8: Implement receipt-only `status` and write the guided user/agent documentation

**Files:**
- Modify: `duck_harness/receipts.py`
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Modify: `README.md`
- Create: `TESTING.md`
- Create: `AGENTS.md`
- Create: `tests/test_status.py`

**Interfaces:**
- `latest_by_command(receipts) -> dict[str, dict]`
- `status(paths, config) -> CommandOutcome`

- [ ] **Step 1: Write failing status tests**

Tests must prove:

- no receipts → status reports UNKNOWN rather than inferring from `.venv`, checkout files, or generated models;
- latest receipt per canonical command wins by receipt timestamp/filename ordering;
- a prior FAIL remains FAIL in the displayed summary until a later receipt for the same command supersedes it;
- malformed receipt produces FAIL for status parsing;
- invoking status does not rerun probes, git, uv, MuJoCo, or training.

Canonical summary keys:

```text
doctor
setup cpu
setup gpu
test unit
test smoke
test sim
test train-smoke
```

- [ ] **Step 2: Implement `status`**

`status` reads receipt JSON only. Its printed table has columns:

```text
COMMAND | STATUS | ENDED_AT | UPSTREAM
```

Absent commands display UNKNOWN with `ENDED_AT` and `UPSTREAM` as `-`.

After reading and rendering the prior receipts, status itself writes its required receipt. Do not let the newly written status receipt affect the summary printed during the same invocation.

- [ ] **Step 3: Rewrite README as the shortest successful path**

README must lead with:

```bash
git clone https://github.com/rfreel/duck.git
cd duck
./duck doctor
./duck setup cpu
./duck test unit
./duck test smoke
./duck test sim
./duck status
```

Then state:

- v1 target: Linux or WSL2;
- upstream Python requirement: 3.12;
- native Windows/macOS: UNKNOWN, not supported by implication;
- CPU-safe tests may still download CUDA-bearing upstream dependencies;
- CUDA training is separate: `./duck setup gpu` then `./duck test train-smoke`;
- no command in the default ladder controls a physical robot.

- [ ] **Step 4: Create TESTING.md as a rung-by-rung guide**

For each rung include exactly these fields:

1. **Prerequisites**
2. **Command**
3. **PASS witness**
4. **What PASS does not prove**
5. **Common failure classes**
6. **Next discriminating action**

Document the claim boundary explicitly:

| Rung | PASS establishes | PASS does not establish |
|---|---|---|
| doctor | required host/tool prerequisites observed | dependencies installed |
| setup cpu | pinned checkout + Python 3.12 upstream environment | tests pass |
| unit | two focused upstream CPU regressions pass | full suite/sim |
| smoke | full documented upstream CPU suite passes | dynamics/training/hardware |
| sim | MicroDuck model loads and advances 50 CPU MuJoCo steps | learned walking behavior |
| train-smoke | CUDA training advances through 5 bounded iterations | reward quality/sim2real |
| hardware | outside v1 | nothing in v1 substitutes for it |

- [ ] **Step 5: Create AGENTS.md**

Agent rules:

- read `upstream.json` before touching derived upstream state;
- never change the pin implicitly;
- run the cheapest relevant failing witness first;
- preserve PASS/FAIL/SKIP/UNKNOWN exactly;
- do not infer test success from artifacts;
- do not treat simulation or short training as hardware validation;
- never run physical robot control from default tests;
- when updating the pin, rerun harness tests, `setup cpu`, `test unit`, `test smoke`, and `test sim`; GPU training remains separately evidenced.

- [ ] **Step 6: Verify docs and status tests**

```bash
python -m pytest tests/test_status.py tests/test_cli.py -q
./duck status
```

- [ ] **Step 7: Commit**

```bash
git add README.md TESTING.md AGENTS.md duck_harness tests

git commit -m "docs: make duck a guided testing ladder"
```

---

## Task 9: Add default CPU CI and execute the acceptance matrix

**Files:**
- Create: `.github/workflows/test.yml`
- Modify: `tests/test_cli.py`
- Modify: `README.md` only if execution reveals a concrete documented mismatch

- [ ] **Step 1: Add a failing local executable-mode assertion before CI**

Add a test or verification script that asserts the repository `duck` file is executable on POSIX. Also verify:

```bash
./duck --help
```

If the executable bit is absent, repair the git mode before proceeding.

- [ ] **Step 2: Create GitHub Actions workflow with two jobs**

Use `.github/workflows/test.yml`:

```yaml
name: test

on:
  push:
  pull_request:

jobs:
  harness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install -e ".[test]"
      - run: python -m pytest -q
      - run: ./duck --help
      - run: ./duck doctor

  upstream-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: python -m pip install uv
      - run: ./duck doctor
      - run: ./duck setup cpu
      - run: ./duck test unit
      - run: ./duck test smoke
```

Do not add `test sim` to default CI; the approved design keeps simulation integration opt-in/local. Do not add `train-smoke` because default GitHub-hosted runners do not satisfy the required CUDA witness.

- [ ] **Step 3: Run the complete harness test suite locally**

```bash
python -m pytest -q
```

Expected: all harness tests PASS.

- [ ] **Step 4: Run the real CPU acceptance ladder on Linux/WSL2**

```bash
./duck doctor
./duck setup cpu
./duck test unit
./duck test smoke
./duck test sim
./duck status
```

Capture the generated receipt paths. Verify each receipt's observed upstream SHA is either the exact pin or `null` only where no checkout had yet been observed. No receipt may claim another revision.

- [ ] **Step 5: Run the GPU acceptance rung only when a qualifying CUDA host is actually available**

```bash
./duck setup gpu
./duck test train-smoke
```

Acceptance requires the live train-smoke receipt to show observed iteration evidence through iteration `4/5`. If a qualifying CUDA host is unavailable, record the GPU acceptance witness as NOT RUN in the implementation report; harness unit tests for the GPU logic may PASS, but they are not a substitute for execution.

- [ ] **Step 6: Verify safety boundary by code search**

From repository root:

```bash
grep -R -n -E 'robotctl|duckctl|rustypot|/dev/tty|serial|bluetooth' duck_harness scripts .github || true
```

Expected: no physical-control invocation in harness/default test code. Documentation may mention prohibited tools only in explanatory text outside the searched implementation paths.

- [ ] **Step 7: Inspect GitHub Actions result**

After pushing the workflow commit, verify both `harness` and `upstream-cpu` jobs complete successfully. If the upstream CPU job fails, use its exact log witness; do not weaken the CPU-safety claim or skip the failing test without identifying the cause.

- [ ] **Step 8: Final repository-level verification**

Run:

```bash
git status --short
python -m pytest -q
./duck status
```

Required final conditions:

- tracked worktree clean after commits;
- harness tests PASS;
- `upstream.json` still pins `29e887ecfbf5d37144759e5a9f8a176dfb83d547`;
- default CI has no GPU or hardware requirement;
- CPU ladder execution has real receipts;
- simulation receipt, if run, contains time advancement and `nu=14`;
- GPU execution is labeled according to actual execution state, never inferred.

- [ ] **Step 9: Commit**

```bash
git add .github/workflows/test.yml tests README.md

git commit -m "ci: verify duck harness and upstream cpu tests"
```

---

## Plan self-review checklist

Before implementation begins, verify these statements against this file:

- Every approved command has an implementation task and a test task.
- Every design acceptance criterion maps to at least one executable witness.
- Upstream authority is a fixed SHA, not a moving branch.
- Native Windows/macOS remain UNKNOWN rather than being generalized from Linux/WSL2.
- CPU tests, CPU simulation, CUDA training, and physical validation remain distinct claims.
- `doctor`'s only mutation is its append-only receipt.
- No setup action is hidden inside a `test` command.
- The training parser is grounded in the pinned `rsl-rl-lib==5.0.1` console format.
- The simulation witness proves physics advancement without requiring a learned policy.
- A missing live CUDA execution remains NOT RUN rather than PASS.
