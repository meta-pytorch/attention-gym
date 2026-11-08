# warpdrive fuzz harness

The harness tests the [safety contract](../../docs/warpdrive/guarantees.md)
and the linked topic guides. [CASES.md](CASES.md) maps the labels in generated reports to
rejection cases and protocol topics. They are test labels, not library error codes.
[Findings](FINDINGS.md) records failure mechanisms and the regressions that protect them.

## Verdicts and execution

Keep these three classes separate:

- **Required rejection:** an invalid declaration or traced protocol must fail for the intended
  reason. Inspect the exception/cause, not just whether compilation failed.
- **Valid abstract program:** matching epochs, participants and live dependencies; compile
  and run, then check values, ownership, sequence and completion.
- **Documented obligation or unverified:** compilation may accept it, but the harness has not
  established a valid runtime program. Never launch a negative compile mutation merely
  because it compiled.

Run compile probes in isolated processes, not threads sharing CuTeDSL tracing state. Keep
source fixed throughout a run. Preserve source/spec/seed on timeout or mismatch, minimize
it, and distinguish layer bugs from generator bugs, toolchain limits and invalid programs.
An implementation-derived scope oracle checks consistency with those rules, not the rules'
correctness; use independent runtime observations too.

## Run

Use the [repository tiers](../../AGENTS.md#warpdrive-testing-tiers). The normal tier skips fuzz;
quick fuzz is explicit. On shared GPU hosts reserve a device and put a process timeout around
runtime/compile jobs that use it. Long campaigns, sanitizer and CUTracer work require the
scope described in those tiers; these commands are not an instruction to run every campaign.

Host/compile probes with an explicit target:

```bash
CUTE_DSL_ARCH=sm_100a .venv/bin/pytest -n 6 -q test/warpdrive_fuzz/test_static.py -m fuzz
```

For a requested campaign, from the repository root:

```bash
.venv/bin/python tools/warpdrive/fuzz_warpdrive.py static --seed 0 --budget 10000
CUTE_DSL_ARCH=sm_100a .venv/bin/python tools/warpdrive/fuzz_warpdrive.py mutate --seed 0 --budget 1000
.venv/bin/python tools/warpdrive/fuzz_warpdrive.py runtime --help
```

Mutation sources are saved under `$WARPDRIVE_FUZZ_MUTATIONS_DIR` (a temporary directory by
default). Use the recorded source/spec and the same revision for reproduction: a seed alone
is not a stable program across generator changes.

## Coverage boundaries

Runtime oracles check progress, values, per-handle epochs, role placement, per-unit work
sequences and coverage, plus bitwise repeatability where the generated program requires it.
Generator cases include Rings/Slots, joins, turns, raw barriers/roles, overrides, storage
bindings and overlaps, try-token and preview waits, built-in/custom schedules, ready device
lists and concurrent launches with separate workspaces. See `gen.py` for the enabled combinations rather than assuming their
full Cartesian product is covered.

Current gaps to consider when extending a kernel or the generator:

- The generated MMA data path is single-CTA; example tests cover two-CTA MMA data.
- Round-robin/named channels and IKET captures are not generated. IKET compile/run is covered.
- Try tokens are probed immediately before their wait and previews read nothing, so the
  generator exercises both probe outcomes and the non-advancing wait, not preview reads or
  work between the probe and the wait (the unit toys in `test_warpdrive_tokens.py` do).
- Multicast generation does not combine both modes in one channel. Manual signals, overrides
  and unclosed handles skip multicast and `fills` because the local-only manual protocol is
  invalid there. `Tma(fills=...)` channels copy the same tile into every CTA they fill (a
  misdirected copy changes no value), and only pair fills are multicast. The scope model
  restates the documented TMA rule, including the signal-unit superset of releases.
- Concurrent launches deliberately sharing a workspace are outside the supported contract.
- A missing async wait can escape a value oracle when the operation happens to complete first.
  In particular, replacing only the UMMA consumer's release with a plain arrival was not
  detected; the MMA finished reading before the next TMA write landed. A work-ring early
  release and a dropped pair tail also escaped the earlier mutation campaign.

Do not interpret a surviving mutant as safe or a clean campaign as a race-freedom proof.
Add a discriminating lifetime/ordering test when changing one of these paths.
