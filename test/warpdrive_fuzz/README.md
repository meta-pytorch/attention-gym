# warpdrive fuzz harness

The harness tests the [safety contract](../../docs/warpdrive/guarantees.md)
and the linked topic guides in two ways:

- **Declaration and mutation fuzzing** (`static.py`, `mutate.py`, `mutate_tma.py`): invalid
  declarations and compile-only source mutations of valid templates must raise the intended
  error. Host/compile only; nothing is launched. Report labels (`E03`, `G2`, ...) name the cases
  in `static.py` and `mutate.py`, each next to the rejection it expects; they are test labels,
  not library error codes. `test_static.py` holds their minimized regressions.
- **Mutant check** (`runtime_mutants.py`, `test_mutants.py`): each mutant breaks one layer
  derivation (a join count, a wait parity, a schedule stride, ...), and the ordinary GPU test
  listed for it must then fail or hang. A mutant that survives means the test suite is blind
  to that failure class.

The random valid-program generator and its runtime oracle stay runnable in
transformer_nuggets' frozen copy of warpdrive; they are not part of this repository.

## Verdicts

Keep these classes separate:

- **Required rejection:** an invalid declaration or traced protocol must fail for the intended
  reason. Inspect the exception/cause, not just whether compilation failed.
- **Documented obligation or unverified:** compilation may accept it, but nothing established
  a valid runtime program. Never launch a negative compile mutation merely because it compiled.

Run compile probes in isolated processes, not threads sharing CuTeDSL tracing state. Keep
source fixed throughout a run. Preserve source and seed on a finding, minimize it, and
distinguish layer bugs from harness bugs, toolchain limits and invalid programs.

## Run

Use the [repository tiers](../../AGENTS.md#warpdrive-testing-tiers). The normal tier runs the
unmarked regressions in `test_static.py` (compile-only, a few seconds) and skips every test
marked `fuzz`. Quick fuzz (`-m fuzz`) is explicit: the seeded cases in `test_static.py` and the
mutant check, which needs a GPU reservation. Long campaigns require the scope described in
those tiers; these commands are not an instruction to run every campaign.

```bash
# regressions (normal tier), then the seeded quick fuzz
CUTE_DSL_ARCH=sm_100a .venv/bin/pytest -n 6 -q test/warpdrive_fuzz/test_static.py
CUTE_DSL_ARCH=sm_100a .venv/bin/pytest -n 6 -q test/warpdrive_fuzz/test_static.py -m fuzz
.venv/bin/python tools/warpdrive/fuzz_warpdrive.py static --seed 0 --budget 10000
.venv/bin/python tools/warpdrive/fuzz_warpdrive.py mutate --seed 0 --budget 1000
```

Mutation sources are saved under `$WARPDRIVE_FUZZ_MUTATIONS_DIR` (a temporary directory by
default). Use the recorded source and the same revision for reproduction: a seed alone is not
a stable program across harness changes.

## Coverage boundaries

A mutant names one failure class, not a proof that the class is covered everywhere: its test
is one hand-written kernel exercising that derivation. Value-based tests can miss a missing
async wait when the operation happens to complete first (for example an MMA consumer's release
replaced by a plain arrival).

Do not interpret a surviving mutant as safe or a clean campaign as a race-freedom proof.
When adding a layer derivation, add a mutant for it and a test that catches it; when changing
a lifetime or ordering path, add a discriminating test.
