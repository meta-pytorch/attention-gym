# SASS equivalence gate

Compiles every vendored GDN and KDA kernel variant through the Attention Gym drivers and
compares the generated SASS of two trees. Use it to gate a mechanical step of the cudnn-frontend
upgrade (restyle, codemod, host refactor): identical SASS, or at most offset noise, with the
shared-memory layout, the mbarrier offsets and the launch resources unchanged.

```bash
# Reference: the tree before the change (a git archive works; it must expose the current
# driver API: cudnn_fe.gdn/kda/summary and their plan builders).
mkdir -p agent_space/sass/ref_tree && git archive <rev> attn_gym | tar -x -C agent_space/sass/ref_tree
gpu-run --timeout 900 auto -- timeout -k 10 2400 .venv/bin/python -m tools.cudnn_fe.sass snapshot agent_space/sass/ref --tree agent_space/sass/ref_tree
# Candidate: the live checkout.
gpu-run --timeout 900 auto -- timeout -k 10 2400 .venv/bin/python -m tools.cudnn_fe.sass snapshot agent_space/sass/new
python -m tools.cudnn_fe.sass diff agent_space/sass/ref agent_space/sass/new --quiet   # exit 1 on real changes
python -m tools.cudnn_fe.sass diff agent_space/sass/ref agent_space/sass/new --strict  # also fail on noise
```

`--tree DIR` names a checkout that contains `attn_gym/`; it is put ahead of the installed
package and the run aborts if `attn_gym` resolves elsewhere (see `tools/cudnn_fe/_common.py`).

A full snapshot (44 cases, 55 artifacts) takes about five minutes on GB200. Both CuTeDSL caches
are pointed at fresh temporary directories for the run, so every kernel really recompiles.
`--cases SUBSTR ...` runs a subset, but kernel caches credit each kernel to the first case that
compiles it, so compare full runs for the gate. Two runs of one tree are byte-identical.

The snapshot is the device code that runs: `cute.compile` embeds the cubin in the IR module
(CuTeDSL `OptLevel` applies there), and the host object export that `jit_cache` publishes uses
the same `export_module_to_bytes(..., opt_level=3)` call as the capture (`opt_level` is the host
LLVM level). Every run checks this by carving the cubins out of the published `.o` files and
matching them byte for byte against the snapshot (`N/N cubins ... match` in `snapshot.log`); a
mismatch fails the run.

## Cases

Per kind (`gdn`, `kda`), on two sequences of a few chunks with two heads, plans pinned through
the drivers' own builders (chain floors, `ForwardPlan.build`, `PREP_TILE_FRACTION`):

| case | kernels |
|---|---|
| `fwd_uncut{,_nostate,_f16,_i64}` | prefill + prologue (uncut table) |
| `fwd_dv`, `kda_fwd_prep` | d_v split; KDA shared prep (`kda_prep`, `kda_prep_prefill`) |
| `fwd_chain{,_i64}` | chain prologue, prefill, tinv, summary, state chain |
| `fwd_warmup` | approximate split: `split_k_*` plan/scan/walk + prefill |
| `fwd_paged{,_mask,_dv}`, `kda_fwd_paged_prep` | paged-state routes and fresh-slot mask |
| `bwd_uncut{,_nostate,_f16,_i64}` | recompute, bprop, tinv + prologues |
| `bwd_chain{,_i64}` | chain backward head/main/tail bundles, bprop summary |
| `bwd_warmup` | approximate split backward |
| `kda_summary_{fwd,fwd_f16,fwd_i64,bwd,bwd_i64,probe}` | CP summaries (`kda_summary`, `kda_bprop_summary`) |

## Snapshot files

Per label `<case>__<compiled fn>_<k>`: `.cubin`, `.sass` (nvdisasm), `.res`
(`<kernel> REG= STACK= SHARED= LOCAL= DSMEM= MBAR=`), `.fns` (kernel stem to symbol) and two
derived views, `.ops` (`<kernel> <OPCODE.MODS> <count>`) and `.layout`
(`<kernel> <OPCODE.MODS> <smem immediate> <count>` over `LDS`/`STS`/`LDSM`/`STSM`/`SYNCS*`).
`DSMEM` is the dynamic shared-memory size of the launch, read from the host IR; `MBAR` lists the
`SYNCS.EXCH` (mbarrier init) shared-memory immediates. `ARTIFACTS.txt`, `CASES.txt`, `META.txt`
and `snapshot.log` describe the run. `diff` re-parses the `.sass` text (resources come from
`.res`), so snapshots taken by an older version of the tool classify under the current rules.

Kernels pair by stem (`gdn_prefill`, `state_chain`, ...) derived from the `frost_<name>` symbol
up to the first mangled argument; labels pair by the host function name, so a host rename shows
every label as "only in a/b" (loud, but the gate cannot see across it).

## Verdicts

Per kernel, paired by label and stem, from strongest to weakest:

- **identical**: normalized instruction text (addresses and label numbers stripped) and
  resources equal. Two runs of one tree give this everywhere.
- **same-histogram**: text differs (scheduling, register allocation) but the full-mnemonic
  histogram, the shared-memory layout, the mbarrier offsets and the resources are equal.
- **noise**: histogram deltas only in `NOP`/`UMOV`/`UIADD3`/`IMAD`/`LOP3`/`CS2R`/`ULOP3`
  (any modifiers), at most 8 instructions in total.
- **real**: `REG`, `STACK`, `DSMEM`, `SHARED`, `LOCAL` or the mbarrier offsets differ, or a
  resource is unknown (`?`) on either side; the shared-memory layout (multiset of
  `(mnemonic, immediate)` of `LDS`/`STS`/`LDSM`/`STSM`/`SYNCS*`) differs; any change to a
  `UTMALDG`/`UTMASTG`/`BSSY`/`SYNCS`/`NANOSLEEP`/`FMUL2`/`FADD2`/`FFMA2` count; any other
  histogram delta outside the noise band; a label or kernel present on one side only; a failed
  case.

`--quiet` prints noise and real labels only; without `--strict` the gate passes identical,
same-histogram and noise labels.

The layout check exists because SMEM relocations leave the histogram intact: the GDN prefill
SharedStorage restyle moved the mbarrier block from `+0x24000` to `+0x0` and cost ~3 % (fixed in
"Restore the v1.30 SMEM order in the GDN prefill, tinv and summary kernels"), and the KDA
restyle moved the `kda_recompute` tiles (`STS.128 +0x1f000 -> +0x1ec00`, `LDS +0x1ec00 ->
+0x2ec00`) with a six-instruction noise delta, which the earlier base-mnemonic histogram passed.
Keying the histogram on the full mnemonic makes width (`LDS.128` vs `LDS.64`), rounding
(`FFMA.FTZ`) and signedness (`IMAD.WIDE.U32`) changes real. Constant-bank slot changes and
instruction reordering are `same-histogram`; use `--strict` plus a `.sass` diff when a step
must be byte-exact.
