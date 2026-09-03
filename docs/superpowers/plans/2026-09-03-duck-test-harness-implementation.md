# Duck test harness implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Use the checkbox steps as the execution ledger. Each implementation task starts from a failing test or failing witness.

**Goal:** Turn `rfreel/duck` into a thin, reproducible guide and test harness for the Pollen Robotics MicroDuck stack, from machine diagnosis through pinned CPU regressions, headless MuJoCo simulation, and a bounded CUDA training smoke test.

**Architecture:** `duck` owns orchestration, probes, logs, receipts, and documentation only. The authoritative simulator/training code remains an immutable `pollen-robotics/microduck_rl` checkout under `.duck/upstream/microduck_rl`. A dependency-free Python controller exposes commands through an executable POSIX `./duck` wrapper. Every command returns explicit checks; the CLI persists one append-only receipt; `status` reconstructs prior state from receipts rather than artifacts.

**Tech stack:** Python 3.10+ for the harness controller; Python 3.12 for upstream; Python standard library; pytest for harness tests; git; uv; upstream MuJoCo/mjlab/rsl_rl; NVIDIA CUDA only for `train-smoke`; GitHub Actions on Ubuntu for default CI.

**Spec:** `docs/superpowers/specs/2026-09-03-duck-test-harness-design.md`

## Verified implementation inputs

The following were established from current upstream source during planning:

- Repository: `pollen-robotics/microduck_rl`.
- Immutable revision: `29e887ecfbf5d37144759e5a9f8a176dfb83d547`.
- Reference branch: `develop`; this branch is descriptive only and is never used as runtime authority.
- At the pin, `pyproject.toml` requires Python `>=3.12,<3.13`; `uv.lock` requires `==3.12.*`.
- Upstream documents `uv run --with pytest pytest tests/` as CPU-only config-invariant and reward-function regression tests.
- `tests/test_nan_guard.py` and `tests/test_obs_nan_guard.py` exist at the pin and form the focused CPU unit rung.
- Training uses MuJoCo Warp/CUDA; upstream accepts `--env.scene.num-envs` and `--agent.max_iterations`.
- Upstream project plans use `--agent.logger tensorboard` for disposable training smoke runs, avoiding a W&B login dependency.
- `uv.lock` pins `rsl-rl-lib==5.0.1`. Its logger emits `Learning iteration {it}/{total_it}`. Training progress is therefore witnessed with `Learning iteration\s+(\d+)/(\d+)`.
- `src/mjlab_microduck/robot/microduck/scene.xml` exists, loads the ground-contact MicroDuck model, contains a floor and named keyframes, and has 14 actuators.
- Upstream source inspected during planning does not establish native Windows or macOS execution support. Harness v1 accepts Linux and WSL2 as Linux-kernel targets. Native Windows and macOS remain UNKNOWN until separately validated.
- “CPU-safe” means no GPU hardware is required to execute that rung. It does not imply a small or CUDA-free dependency installation; upstream resolution can still pull CUDA-bearing wheels.

## Global constraints

- `upstream.json` is authority. `.duck/upstream/microduck_rl` is disposable derived state.
- Every setup/test command verifies the observed checkout SHA equals the immutable pin.
- A dirty derived checkout is never reset or discarded silently; report FAIL and stop.
- `doctor` performs no installation, fetch, checkout, configuration, or external-state mutation. Its only write is its append-only receipt under `.duck/receipts/`.
- Test commands never invoke setup implicitly. Missing setup is a precondition failure whose next action names `./duck setup cpu` or `./duck setup gpu`.
- No v1 command invokes `robotctl`, `duckctl`, a serial bus, Bluetooth robot control, flashing, policy deployment, or physical robot control.
- Required FAIL dominates overall status. Otherwise a required UNKNOWN or SKIP yields UNKNOWN. Overall PASS requires all required checks to PASS.
- Optional missing GPU capability is SKIP in `doctor`. GPU capability is required and therefore FAIL for `setup gpu` and `test train-smoke`.
- Use subprocess argument arrays and `shell=False` only.
- Harness-owned mutable paths stay under `.duck/` or the disposable upstream checkout, apart from ordinary uv cache behavior.
- Default CI runs harness tests plus upstream CPU unit/smoke tests. Simulation, CUDA training, Hugging Face Jobs, and hardware are excluded from default CI.
- Each task below ends in a separate commit.

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

```python
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
```

- [ ] **Step 1: Write failing tests**

`tests/test_model.py` covers: all required PASS → PASS; any required FAIL → FAIL; required UNKNOWN → UNKNOWN; optional SKIP does not block PASS; required SKIP → UNKNOWN.

`tests/test_config.py` covers: exact valid authority shape; missing field rejected; extra field rejected; repository other than `pollen-robotics/microduck_rl` rejected; revision not matching `^[0-9a-f]{40}$` rejected; schema version other than `1` rejected.

`tests/test_paths.py` verifies every mutable path can be resolved relative to `root/.duck`.

Run:

```bash
python -m pytest tests/test_model.py tests/test_config.py tests/test_paths.py -q
```

Expected initial witness: import/collection failure because the modules do not yet exist.

- [ ] **Step 2: Create authority and package metadata**

`upstream.json`:

```json
{
  "schema_version": 1,
  "repository": "pollen-robotics/microduck_rl",
  "revision": "29e887ecfbf5d37144759e5a9f8a176dfb83d547",
  "reference_branch": "develop"
}
```

`.gitignore`:

```gitignore
.duck/
.venv/
__pycache__/
.pytest_cache/
*.py[cod]
```

`pyproject.toml`:

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

- [ ] **Step 3: Implement model/config/paths**

`aggregate_status(checks)` uses this exact precedence:

```python
required = [check for check in checks if check.required]
if any(check.status is Status.FAIL for check in required):
    return Status.FAIL
if any(check.status in (Status.UNKNOWN, Status.SKIP) for check in required):
    return Status.UNKNOWN
if required and all(check.status is Status.PASS for check in required):
    return Status.PASS
return Status.UNKNOWN
```

`HarnessPaths.from_root(root)` resolves:

```text
state_dir    = ROOT/.duck
upstream_dir = ROOT/.duck/upstream/microduck_rl
receipts_dir = ROOT/.duck/receipts
logs_dir     = ROOT/.duck/logs
```

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/test_model.py tests/test_config.py tests/test_paths.py -q
git add upstream.json .gitignore pyproject.toml duck_harness tests
git commit -m "feat: establish duck harness authority model"
```

---

## Task 2: Add deterministic subprocess logs and append-only receipts

**Files:**
- Create: `duck_harness/process.py`
- Create: `duck_harness/receipts.py`
- Create: `tests/test_process.py`
- Create: `tests/test_receipts.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path

def run_logged(argv: Sequence[str], *, cwd: Path, log_dir: Path, stem: str,
               env: Mapping[str, str] | None = None) -> ProcessResult: ...

def utc_now() -> str: ...
def write_receipt(paths: HarnessPaths, command: Sequence[str], outcome: CommandOutcome,
                  host: dict[str, Any], started_at: str, ended_at: str) -> Path: ...
def read_receipts(paths: HarnessPaths) -> list[dict[str, Any]]: ...
```

- [ ] **Step 1: Write failing tests**

`tests/test_process.py` runs `sys.executable -c` and proves stdout/stderr separation, exact nonzero exit-code propagation, log confinement, and no shell interpolation.

`tests/test_receipts.py` proves:

- PASS/FAIL/SKIP/UNKNOWN survive serialization exactly;
- `upstream_revision` may be null before an observed checkout exists;
- log paths are repository-relative POSIX strings;
- two receipt writes create distinct files;
- malformed JSON or missing required receipt fields raises `ValueError` naming the file.

- [ ] **Step 2: Implement subprocess logging**

Use `subprocess.run(list(argv), cwd=cwd, stdout=stdout_handle, stderr=stderr_handle, check=False, env=env)` and no shell. Create unique log filenames using a sanitized stem plus `time.time_ns()`.

- [ ] **Step 3: Implement receipt schema and atomic write**

Schema version `1` fields:

```text
schema_version
command
upstream_revision
started_at
ended_at
host
checks
overall_status
log_paths
```

Persist through a temporary file in `.duck/receipts/` followed by `os.replace`. The writer may create `.duck/receipts/`; it must not create unrelated state directories.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/test_process.py tests/test_receipts.py -q
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

```python
def host_snapshot() -> dict[str, object]: ...
def probe_host_target() -> CheckResult: ...
def probe_executable(name: str, *, required: bool = True) -> CheckResult: ...
def probe_python312(*, required: bool = True) -> CheckResult: ...
def probe_nvidia(*, required: bool = False) -> CheckResult: ...
def doctor(paths: HarnessPaths, config: UpstreamConfig) -> CommandOutcome: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

- [ ] **Step 1: Write failing host/tool tests**

Cover four distinct host paths:

| Observation | Required status | `evidence.target` |
|---|---|---|
| Linux, release contains `microsoft` | PASS | `wsl2` |
| other Linux | PASS | `linux` |
| Windows | UNKNOWN | `native-windows` |
| macOS | UNKNOWN | `macos` |

Also cover missing `git` → FAIL, missing `uv` → FAIL, exact Python 3.12 found → PASS with path evidence, no Python 3.12 → FAIL, missing optional `nvidia-smi` → SKIP, and failing optional `nvidia-smi` invocation → UNKNOWN.

Inject platform strings, executable lookup, and subprocess runner into probe helpers so tests are deterministic.

- [ ] **Step 2: Implement probes without mutation**

Python 3.12 candidates, in order:

```text
sys.executable
python3.12
python3
python
```

For each unique existing executable run:

```text
EXE -c import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')
```

PASS only for output `3.12`.

If `nvidia-smi` exists, run:

```text
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

- [ ] **Step 3: Implement doctor and CLI receipt boundary**

Required doctor checks: host target, `git`, `uv`, Python 3.12. NVIDIA is optional.

CLI sequence for a normal command:

1. load root `upstream.json`;
2. capture start time and host snapshot;
3. run command implementation;
4. capture end time;
5. write exactly one receipt;
6. print `STATUS  check-name  detail` lines plus overall status and receipt path.

Exit code mapping for test/setup/doctor commands:

```text
PASS    -> 0
FAIL    -> 1
UNKNOWN -> 2
```

- [ ] **Step 4: Create executable `duck`**

```sh
#!/usr/bin/env sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
cd "$ROOT"
exec "$PYTHON" -m duck_harness "$@"
```

Then:

```bash
chmod +x duck
git update-index --chmod=+x duck
```

- [ ] **Step 5: Verify and commit**

```bash
python -m pytest tests/test_probes.py tests/test_cli.py -q
./duck doctor
```

A local doctor may legitimately return FAIL/UNKNOWN when a prerequisite is absent. Verify that it creates a receipt but does not create `.duck/upstream/`.

```bash
git add duck duck_harness tests
git commit -m "feat: add read-only duck doctor"
```

---

## Task 4: Fetch the immutable upstream checkout and create its Python 3.12 environment

**Files:**
- Create: `duck_harness/upstream.py`
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Create: `tests/test_upstream.py`

**Interfaces:**

```python
def observe_checkout_revision(paths: HarnessPaths) -> CheckResult: ...
def ensure_checkout(config: UpstreamConfig, paths: HarnessPaths,
                    runner=run_logged) -> CommandOutcome: ...
def verify_environment(paths: HarnessPaths) -> CheckResult: ...
def setup(paths: HarnessPaths, config: UpstreamConfig, mode: str) -> CommandOutcome: ...
```

- [ ] **Step 1: Write failing checkout/setup tests**

Using a fake logged runner, cover:

1. absent checkout creates clone/fetch/detached-checkout command sequence;
2. exact observed SHA → PASS;
3. other SHA → FAIL, regardless of branch name;
4. dirty checkout → FAIL with no reset/checkout afterward;
5. wrong origin URL → FAIL;
6. failed host/git/uv/Python prerequisite prevents clone;
7. `setup cpu` does not require NVIDIA;
8. `setup gpu` requires NVIDIA and Torch CUDA visibility;
9. failed `uv sync` preserves the numeric exit code.

- [ ] **Step 2: Implement checkout operations**

Canonical remote:

```text
https://github.com/pollen-robotics/microduck_rl.git
```

Absent checkout operations:

```text
git clone --filter=blob:none --no-checkout https://github.com/pollen-robotics/microduck_rl.git .duck/upstream/microduck_rl
git -C .duck/upstream/microduck_rl fetch --depth=1 origin 29e887ecfbf5d37144759e5a9f8a176dfb83d547
git -C .duck/upstream/microduck_rl checkout --detach 29e887ecfbf5d37144759e5a9f8a176dfb83d547
git -C .duck/upstream/microduck_rl rev-parse HEAD
```

Existing checkout probes, before mutation:

```text
git -C .duck/upstream/microduck_rl status --porcelain
git -C .duck/upstream/microduck_rl remote get-url origin
git -C .duck/upstream/microduck_rl rev-parse HEAD
```

Non-empty status or wrong origin stops with FAIL. A clean wrong SHA may fetch and detach-checkout the immutable pin. Final PASS requires exact SHA equality.

- [ ] **Step 3: Create upstream environment**

Use the exact Python 3.12 path recorded by `probe_python312`:

```text
uv sync --python OBSERVED_PYTHON_3_12_PATH --frozen
```

The code passes the observed path as one argv element; it is not shell-expanded text.

Environment readiness requires `.duck/upstream/microduck_rl/.venv/bin/python` and exact output `3.12` from:

```text
.venv/bin/python -c import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')
```

For `setup gpu`, also require:

```text
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uv run --frozen python -c import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))
```

- [ ] **Step 4: Add CLI commands and verify**

Only these modes are accepted:

```bash
./duck setup cpu
./duck setup gpu
```

Run:

```bash
python -m pytest tests/test_upstream.py tests/test_cli.py -q
./duck setup cpu
```

On a qualifying host, verify receipt `upstream_revision` and `git rev-parse HEAD` both equal the immutable pin.

- [ ] **Step 5: Commit**

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

```python
def test_unit(paths: HarnessPaths, config: UpstreamConfig,
              runner=run_logged) -> CommandOutcome: ...
def test_smoke(paths: HarnessPaths, config: UpstreamConfig,
               runner=run_logged) -> CommandOutcome: ...
```

- [ ] **Step 1: Write failing exact-argv tests**

Focused unit argv:

```text
uv run --frozen --with pytest pytest -q tests/test_nan_guard.py tests/test_obs_nan_guard.py
```

Full CPU smoke argv:

```text
uv run --frozen --with pytest pytest -q tests/
```

Both run with `cwd=.duck/upstream/microduck_rl`.

Independently test missing checkout, SHA mismatch, missing environment, and nonzero pytest exit. None may trigger setup.

- [ ] **Step 2: Implement readiness and CPU tiers**

Before running pytest, require exact checkout SHA and verified Python 3.12 environment. A missing precondition reports FAIL with `./duck setup cpu` as the next action.

Pytest exit `0` → PASS. Any nonzero → FAIL with exact exit code and stdout/stderr log paths. Do not parse pytest prose to upgrade a nonzero exit.

- [ ] **Step 3: Add commands and verify**

```bash
./duck test unit
./duck test smoke
python -m pytest tests/test_cpu_tiers.py tests/test_cli.py -q
```

PASS establishes the pinned upstream CPU regression surface only.

- [ ] **Step 4: Commit**

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

```python
def test_sim(paths: HarnessPaths, config: UpstreamConfig,
             runner=run_logged) -> CommandOutcome: ...
```

- [ ] **Step 1: Write failing simulation tests**

A passing fake stdout payload is:

```json
{"steps": 50, "initial_time": 0.0, "final_time": 0.1, "nq": 21, "nu": 14}
```

Test separate failures for nonzero subprocess exit, invalid JSON, non-advancing time, steps other than 50, actuator count other than 14, SHA mismatch, and missing environment.

- [ ] **Step 2: Create deterministic probe**

`scripts/probes/sim_step.py`:

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
    return 0 if payload["final_time"] > initial_time and payload["nu"] == 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Implement harness validation**

Run:

```text
uv run --frozen python ABSOLUTE_DUCK_ROOT/scripts/probes/sim_step.py ABSOLUTE_UPSTREAM_ROOT/src/mjlab_microduck/robot/microduck/scene.xml
```

Parse the final non-empty stdout line as JSON and independently require:

```text
returncode == 0
steps == 50
final_time > initial_time
nu == 14
```

Record the payload as receipt evidence.

Claim boundary: PASS proves the pinned MicroDuck model loads and advances 50 CPU MuJoCo steps. It does not prove learned walking behavior.

- [ ] **Step 4: Verify and commit**

```bash
python -m pytest tests/test_sim_tier.py tests/test_cli.py -q
./duck test sim
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

```python
def parse_learning_iterations(text: str) -> list[tuple[int, int]]: ...
def test_train_smoke(paths: HarnessPaths, config: UpstreamConfig,
                     runner=run_logged) -> CommandOutcome: ...
```

- [ ] **Step 1: Write failing parser and command tests**

Parser regex:

```python
re.compile(r"Learning iteration\s+(\d+)/(\d+)")
```

Required parser test:

```python
text = "\x1b[1m Learning iteration 0/5 \x1b[0m\n Learning iteration 4/5 "
assert parse_learning_iterations(text) == [(0, 5), (4, 5)]
```

Training tests independently cover: missing GPU, nonzero Torch CUDA probe, trainer nonzero, trainer exit `0` with no progress witness, and progress ending before `4/5`.

- [ ] **Step 2: Implement exact preconditions and argv**

Require pinned checkout, Python 3.12 environment, passing `nvidia-smi`, then:

```text
uv run --frozen python -c import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))
```

Bounded trainer:

```text
uv run --frozen train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard
```

PASS requires trainer exit `0` and parsed evidence containing total `5` with maximum observed iteration index at least `4`. Record all parsed iteration tuples.

Do not use reward value as an acceptance criterion. A five-iteration run is a training-path witness, not a policy-quality or sim-to-real witness.

- [ ] **Step 3: Verify unit logic**

```bash
python -m pytest tests/test_train_tier.py tests/test_cli.py -q
```

If a qualifying CUDA host exists:

```bash
./duck setup gpu
./duck test train-smoke
```

If no qualifying CUDA host exists, the live GPU witness remains NOT RUN. Unit tests do not substitute for it.

- [ ] **Step 4: Commit**

```bash
git add duck_harness tests
git commit -m "feat: add bounded cuda training smoke test"
```

---

## Task 8: Add receipt-only status and guided documentation

**Files:**
- Modify: `duck_harness/receipts.py`
- Modify: `duck_harness/commands.py`
- Modify: `duck_harness/cli.py`
- Modify: `README.md`
- Create: `TESTING.md`
- Create: `AGENTS.md`
- Create: `tests/test_status.py`

**Interfaces:**

```python
def latest_by_command(receipts: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]: ...
def status(paths: HarnessPaths, config: UpstreamConfig) -> CommandOutcome: ...
```

- [ ] **Step 1: Write failing status tests**

Canonical summary rows:

```text
doctor
setup cpu
setup gpu
test unit
test smoke
test sim
test train-smoke
```

Tests prove:

- absent prior receipts display UNKNOWN for every row;
- latest valid receipt for a row supersedes earlier receipts;
- prior FAIL remains visible until superseded by a later receipt for the same row;
- malformed receipt makes the `status` command itself FAIL;
- status does not call probes, git, uv, MuJoCo, or training.

Distinguish two levels explicitly: the **summary rows** may be UNKNOWN while the **status command outcome** is PASS when the receipt store was parsed successfully. If receipt parsing fails, status command outcome is FAIL.

- [ ] **Step 2: Implement status**

Print:

```text
COMMAND | STATUS | ENDED_AT | UPSTREAM
```

Use `-` for absent timestamp/SHA. Read and render prior receipts first; then persist status's own required receipt. Never infer success from `.venv`, checkout files, logs, checkpoints, or model artifacts.

- [ ] **Step 3: Rewrite README as the shortest path**

Lead with:

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

State: Linux/WSL2 v1 target; Python 3.12 upstream requirement; native Windows/macOS UNKNOWN; CPU-safe tests can still install CUDA-bearing dependencies; CUDA training is separate via `setup gpu` and `test train-smoke`; no default command controls a physical robot.

- [ ] **Step 4: Create TESTING.md**

Every rung contains six fields: Prerequisites, Command, PASS witness, What PASS does not prove, Common failure classes, Next discriminating action.

Required claim table:

| Rung | PASS establishes | PASS does not establish |
|---|---|---|
| doctor | required host/tool prerequisites observed | dependencies installed |
| setup cpu | pinned checkout + Python 3.12 upstream environment | tests pass |
| unit | focused upstream CPU guards pass | full suite/simulation |
| smoke | full documented upstream CPU suite passes | dynamics/training/hardware |
| sim | MicroDuck model loads and advances 50 CPU MuJoCo steps | learned walking behavior |
| train-smoke | CUDA training advances through five bounded iterations | reward quality/sim-to-real |
| hardware | outside v1 | no v1 rung substitutes for hardware validation |

- [ ] **Step 5: Create AGENTS.md**

Rules: read `upstream.json`; never move the pin implicitly; run cheapest discriminating witness first; preserve four status values; do not infer test success from artifacts; do not equate simulation/training with hardware; never run physical control from default tests; pin updates require harness tests plus CPU setup/unit/smoke/sim compatibility checks.

- [ ] **Step 6: Verify and commit**

```bash
python -m pytest tests/test_status.py tests/test_cli.py -q
./duck status
git add README.md TESTING.md AGENTS.md duck_harness tests
git commit -m "docs: make duck a guided testing ladder"
```

---

## Task 9: Add default CPU CI and execute the acceptance matrix

**Files:**
- Create: `.github/workflows/test.yml`
- Modify: `tests/test_cli.py`
- Modify: `README.md` only if a real execution witness reveals a concrete documentation mismatch

- [ ] **Step 1: Verify executable contract locally**

```bash
test -x duck
./duck --help
```

Repair the tracked executable bit if either fails.

- [ ] **Step 2: Create two-job GitHub Actions workflow**

`.github/workflows/test.yml`:

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
      - run: python -m pip install -e ".[test]" uv
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

`uv` is deliberately installed in **both** jobs because it is a required doctor probe. Do not add simulation or CUDA training to default CI.

- [ ] **Step 3: Run complete harness tests**

```bash
python -m pytest -q
```

- [ ] **Step 4: Execute the real CPU acceptance ladder on Linux/WSL2**

```bash
./duck doctor
./duck setup cpu
./duck test unit
./duck test smoke
./duck test sim
./duck status
```

Inspect generated receipts. After setup, every command that observes the checkout must record exactly `29e887ecfbf5d37144759e5a9f8a176dfb83d547`; pre-checkout doctor may record null because no revision was observed.

- [ ] **Step 5: Execute the GPU acceptance rung only with a real qualifying CUDA host**

```bash
./duck setup gpu
./duck test train-smoke
```

Live acceptance requires receipt iteration evidence through `4/5`. If the host is unavailable, implementation reporting must say NOT RUN.

- [ ] **Step 6: Verify physical-control boundary**

```bash
grep -R -n -E 'robotctl|duckctl|rustypot|/dev/tty|serial|bluetooth' duck_harness scripts .github || true
```

Expected: no physical-control invocation in implementation/default-test paths.

- [ ] **Step 7: Push and inspect CI**

Both `harness` and `upstream-cpu` jobs must PASS. If upstream CPU CI fails, diagnose the exact log witness rather than weakening or skipping the check.

- [ ] **Step 8: Final verification**

```bash
git status --short
python -m pytest -q
./duck status
```

Required final state:

- tracked worktree clean after commits;
- harness tests PASS;
- `upstream.json` still contains the immutable pin;
- default CI requires neither GPU nor robot hardware;
- CPU ladder has real receipts;
- a run simulation receipt contains advancing time and `nu=14`;
- GPU execution state reflects actual execution, never inference.

- [ ] **Step 9: Commit**

```bash
git add .github/workflows/test.yml tests README.md
git commit -m "ci: verify duck harness and upstream cpu tests"
```

---

## Plan self-review

This plan is ready for execution only while all statements below remain true:

- Each approved command is mapped to implementation and tests.
- Each design acceptance criterion has an executable witness.
- The upstream authority is a SHA, not a moving branch.
- Linux, WSL2, native Windows, and macOS are not collapsed into one support claim.
- CPU unit tests, CPU full suite, CPU dynamics, CUDA training, and physical validation remain distinct claims.
- `doctor` has no mutation beyond its receipt.
- No test command performs hidden setup.
- Status summary state is distinguished from status-command execution success.
- The training parser is grounded in pinned `rsl-rl-lib==5.0.1` output.
- The simulation witness does not require a learned policy.
- Missing live CUDA execution remains NOT RUN rather than PASS.
