# Duck test harness design

Date: 2026-09-03
Status: approved design; implementation not yet started
Repository: `rfreel/duck`

## Purpose

`duck` is a thin, reproducible control and test harness for Pollen Robotics MicroDuck. It does not fork or duplicate the simulator. The authoritative simulator/training implementation remains `pollen-robotics/microduck_rl`; `duck` pins an upstream revision and supplies an ergonomic path from an unprepared machine to verified local tests, simulation, and later training.

The primary interaction is progressive and inspectable:

`doctor -> setup -> unit/smoke -> simulation -> training -> explicit hardware/deployment`

Each rung must report what was actually checked. A failed or unavailable check remains FAIL or UNKNOWN; later commands must not silently reinterpret it as PASS.

## Scope

### In scope for the first implementation

1. A pinned upstream manifest identifying the exact `microduck_rl` repository and revision used by `duck`.
2. A machine diagnostic command that checks required local capabilities without installing, fetching, configuring, or changing system/dependency/checkout state. Its sole permitted local mutation is the append-only diagnostic receipt required by this design under `.duck/receipts/`.
3. A setup command that fetches the pinned upstream source and installs only the dependencies required for the selected rung.
4. CPU-safe unit/config smoke tests where upstream supports them.
5. A local simulation test that launches the MicroDuck model or policy inference when its prerequisites are present.
6. An optional GPU training smoke test with deliberately small workload; this proves the training path starts, not that a policy is good.
7. Human-readable testing documentation organized by progressive rungs and failure recovery.
8. GitHub Actions for checks that are demonstrably CPU-safe and independent of robot hardware.
9. Machine-readable receipts for each harness command so future automation can distinguish PASS, FAIL, SKIP, and UNKNOWN.
10. An `AGENTS.md` that tells coding agents to preserve the upstream pin, run the cheapest relevant test first, and never treat simulator success as hardware validation.

### Explicitly out of scope for the first implementation

- Copying or vendoring the complete `microduck_rl` source tree.
- Automatically flashing, commanding, or deploying to a physical MicroDuck.
- Treating a short RL smoke run as evidence of policy quality or sim-to-real transfer.
- Reimplementing Pollen's robot runtime, policy contract, actuator physics, or training environments.
- Making Hugging Face, Weights & Biases, CUDA, or a physical robot mandatory for the CPU-safe rung.

## Architecture

### 1. Upstream authority boundary

`upstream.json` at the repository root is the canonical local declaration of the external dependency. Its first implementation has exactly these authority fields:

- `schema_version`
- `repository`: `pollen-robotics/microduck_rl`
- `revision`: an immutable commit SHA selected during implementation planning
- `reference_branch`: the upstream branch from which that immutable revision was selected, retained for human reference only

The harness clones or updates a disposable working copy under a gitignored directory such as `.duck/upstream/microduck_rl`. The working copy is derived state. The manifest is authority.

Changing the pin is an explicit repository change and must run the compatibility tests before merge.

### 2. Command surface

The repository exposes one small front door, implemented with ordinary shell/Python rather than a custom daemon:

- `./duck doctor` — capability inspection whose probes are read-only; the command persists only its required append-only diagnostic receipt under `.duck/receipts/`.
- `./duck setup [cpu|gpu]` — prepare the requested rung.
- `./duck test unit` — run the cheapest upstream tests known to be CPU-safe.
- `./duck test smoke` — configuration/environment smoke checks.
- `./duck test sim` — start a real MicroDuck simulation/inference path.
- `./duck test train-smoke` — launch a bounded RL training run and verify it advances through initial iterations.
- `./duck status` — summarize the most recent receipts without rerunning tests.

The front door delegates to focused scripts in `scripts/`; it should contain almost no logic itself.

### 3. Receipts

Every command writes a timestamped JSON receipt under `.duck/receipts/` and prints a concise human summary. Receipts are ignored by git.

Minimum receipt fields:

- command and arguments
- upstream revision actually observed
- start/end timestamps
- host/platform facts used by the decision
- checks performed
- status per check: `PASS`, `FAIL`, `SKIP`, or `UNKNOWN`
- subprocess exit codes where applicable
- artifact/log paths

A command only reports overall PASS when its required checks pass. Missing optional hardware or GPU capability is SKIP when that capability is outside the selected rung; it is FAIL when the user explicitly selected a rung requiring it. UNKNOWN is preserved when the harness cannot establish a fact safely.

### 4. Documentation

`README.md` is the shortest successful path and links to `TESTING.md`.

`TESTING.md` is organized as a ladder rather than a tool catalog:

1. Diagnose the machine.
2. Establish a CPU-safe baseline.
3. See the robot in simulation.
4. Exercise the training path.
5. Understand what has and has not been proven.
6. Only then follow separately documented hardware/deployment instructions.

Each rung states prerequisites, exact command, expected PASS witness, common failure classes, and the next safe action.

`AGENTS.md` gives agents the same ladder and forbids skipping directly to expensive training when a cheaper failing witness exists.

## Data flow

1. User invokes `./duck <command>`.
2. Harness reads `upstream.json`.
3. Harness inspects local capabilities needed by that command.
4. If source is required, harness verifies the checkout resolves to the pinned SHA; it does not silently follow upstream branch movement.
5. Harness invokes the narrow upstream command corresponding to the rung.
6. Raw stdout/stderr is retained under `.duck/logs/`.
7. The harness derives a receipt from observed exit status and explicit probes.
8. `./duck status` reads receipts only; it does not infer success from the mere existence of generated files.

## Failure handling

The harness fails early on mismatched upstream revisions, missing required executables, or incompatible Python/runtime prerequisites.

Setup may create an isolated project environment only inside `.duck/` or the disposable upstream checkout. The exact environment mechanism and path are implementation inputs to be selected only after verifying the pinned upstream tooling. `doctor` performs no installation, fetch, checkout, configuration, or external-state mutation; its sole local write is the append-only receipt required under `.duck/receipts/`.

Commands must print the failed precondition and the smallest next command that can resolve or further discriminate the failure.

No cleanup command may delete paths outside `.duck/` or the disposable upstream checkout. No test command may contact or control robot hardware implicitly.

## Testing strategy for `duck` itself

The harness has its own tests separate from upstream MicroDuck tests.

1. **Static/contract tests:** manifest schema, command parsing, receipt schema, path confinement, and status aggregation.
2. **Probe tests:** capability detection against synthetic PATH/environment fixtures so PASS/FAIL/UNKNOWN behavior is deterministic.
3. **Subprocess tests:** fake upstream executables validate exit-code propagation and log capture without requiring CUDA or MuJoCo.
4. **Integration smoke:** on a host meeting the documented prerequisites, fetch the pinned upstream revision and run the cheapest real CPU-safe upstream test set.
5. **Simulation integration:** opt-in/local; verifies the selected viewer/inference process starts and advances rather than merely importing modules.
6. **Training integration:** opt-in/GPU; bounded iteration count and environment count. Acceptance is successful initialization plus observed iteration advancement, not reward quality.
7. **Hardware tests:** outside the first implementation and never part of default CI.

GitHub Actions initially runs only tests 1-4 when their prerequisites are verified to be CPU-safe. Simulation, GPU training, and hardware validation remain explicit separate tiers.

## Acceptance criteria

The first implementation is accepted when all of the following are true:

- A fresh host meeting the implementation-documented prerequisites can clone `duck`, run `./duck doctor`, and receive an unambiguous capability report without installation, dependency, checkout, configuration, or external-state mutation; the required append-only receipt under `.duck/receipts/` is the only permitted local write.
- `./duck setup cpu` produces a pinned upstream checkout and isolated environment, or fails with a specific prerequisite witness.
- `./duck test unit` and `./duck test smoke` execute real checks and emit receipts.
- On a host satisfying the documented simulation prerequisites, `./duck test sim` proves that a MicroDuck simulation process starts and advances.
- On a host satisfying the documented CUDA/GPU prerequisites, `./duck test train-smoke` proves a bounded training run advances through initial iterations without claiming policy quality.
- `./duck status` reconstructs only observed prior results from receipts.
- Default CI does not require a GPU or physical robot.
- No default or test command can command, flash, or deploy to a physical robot.
- Documentation makes the boundary between unit validation, simulator validation, training-path validation, and physical validation explicit.

## Alternatives preserved

### A. Pinned upstream harness — selected

Lowest duplication and clearest authority boundary. Costs: the harness must explicitly manage compatibility as upstream changes.

### B. Full fork or vendored simulator — not selected

Gives maximum ability to patch simulator internals but creates a second source of truth and ongoing merge burden. This remains a valid future choice if upstream changes become incompatible with the intended experiments.

### C. Container-first harness — not selected initially

Can make CUDA/system dependency reproduction stronger, but introduces container runtime and GPU pass-through complexity before the CPU-safe rung. It remains a future execution backend, not an architectural dead end.

### D. Pure documentation with no harness — not selected

Lowest implementation cost but cannot produce machine-readable receipts, enforce an immutable upstream revision, or distinguish observed test execution from user-reported state.

## Non-claims and unresolved implementation inputs

This design does not establish which host operating systems satisfy the harness prerequisites, which exact upstream tests are CPU-safe, which environment mechanism the pinned upstream revision requires, or which immutable upstream commit should be pinned. Those remain materially distinct unresolved inputs. They must be established from current upstream code during implementation planning and recorded as verified inputs rather than filled from assumption.
