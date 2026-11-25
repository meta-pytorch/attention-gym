---
name: adding-megakernel-model
description: Adds a model's decode to the warpdrive megakernels on the shared examples/megakernel package, rung by rung from a bitwise oracle to a chat backend. Use when porting a new model to megakernel launches, adding a chat family, or moving model-agnostic code into the package.
---

# Adding a megakernel model

The package contracts (what belongs in `examples/megakernel/`, compile-key hygiene, kernel
rules) live in [`attn_gym/warpdrive/examples/megakernel/AGENTS.md`](../../../attn_gym/warpdrive/examples/megakernel/AGENTS.md);
read it first. Read [`authoring-warpdrive`](../authoring-warpdrive/SKILL.md) before writing
kernels, and use the closest model directory (`qwen_megakernel/`, `inkling_megakernel/`) as
the worked example; for a dense sliding-window GQA layer the closest is the DSpark drafter's
`inkling_megakernel/draft_backbone_kernel.py`.

Routing: a new model's decode, its chat family, or moving a piece between a model directory and
the package uses this skill. Tuning one existing kernel (`cutedsl-performance`) or changing a
warpdrive layer feature does not.

## Setup

- Work in a worktree with its own `.venv` ([`worktree-env-setup`](../worktree-env-setup/SKILL.md)).
- Reserve GPUs with `gpu-run` and bound every launch with a process timeout:
  `gpu-run --timeout 120 auto -- timeout --signal=TERM --kill-after=5s <N>s ...`. A timeout
  is a hang to diagnose, not a slow run.
- Real checkpoints come from the local Hugging Face cache; tests skip without them, so also
  build a tiny random model that the CPU and GPU tests always run.

## The ladder

Climb in order; each rung is tested against the previous ones before the next starts.

0. **Shapes.** Check the model against the package's shape preconditions (AGENTS.md: head
   dimension 128, hidden size a multiple of 256, 32-row GEMV tiles, SM100) and write that check
   into the kernels' constructor first; a model that fails one needs a package change, not a
   model directory.
1. **Oracle.** `<model>/ref.py`: PyTorch only, HF's dtype and rounding boundaries, checkpoint
   loading. Gate: bit-identical to HF's eager model on the tiny model and on short real prompts;
   its docstring states the scope and any deliberate difference.
2. **Capturable torch step.** Device token and position, static shapes, no host sync. Gate:
   equal to the oracle's decode, and a CUDA graph of several steps replays correctly. It is the
   baseline backend and bench row.
3. **One-token kernels.** Per-layer launches composed from the package (tagged words, GEMV
   engine, ring loads, instruction table); op codes, row schedule and scratch stay in the model
   directory. Gates: each work-item kind alone, then whole steps, against the oracle within a
   stated error budget at every static specialization and at runtime-size boundaries (AGENTS.md,
   exposed sizes), with compiled-kernel reuse across runtime sizes; NaN-filled unused cache rows;
   repeated launches bit for bit; the kernels' required modules in
   `test_warpdrive_megakernel_package.py`'s `REQUIRED`.
4. **Quantized weights** (when the checkpoint stores them). Stream them as stored: pack them to
   `megakernel.formats`' layout (`inkling_megakernel/quant.py` is a worked loader) and decode
   with `quant_gemv`. When the checkpoint declares its experts' activation quantization (a
   `hf_quant_config.json`), the oracle and the kernels compute those numerics
   (AGENTS.md, expert numerics; `megakernel.act_quant`), oracle first, its quantizers
   bit-matched against the reference serving stack's. Gate: every launch output equals the same
   launch on the weights' bf16 block values bit for bit, on random weights over every supported
   code and scale byte and on the checkpoint.
5. **Chat family.** `<model>/chat.py` provides `DEFAULT_BACKEND`, `OPTIONS`, `check_options`,
   `checkpoint_dir` and `load_chat_model`; register its `model_type` in `megakernel/chat.py`'s
   `FAMILIES`. A new flag also goes into `ChatOptions` (default None: family-specific), into
   `main()`'s options and its `ChatOptions(...)` call, and into the family's `OPTIONS`. Each
   backend's `kernels()` lists the `(compile_fn, kernel)` calls its steps and prefill launch,
   which the chat compiles in parallel before the first prompt; test the list against the calls
   a step makes (the `record_compiles` fixture). Gate: a real prompt answers coherently through
   `python -m attn_gym.warpdrive.examples.megakernel.chat --model <repo>`.
6. **Bench.** `<model>/bench.py` on `examples/bench_support.py` (`capture`, `time_graphs`,
   `provenance`): CUDA-graph replay of the torch step and the megakernel at fixed contexts.
   Record the bandwidth floor next to the measurement.
7. **Speculative decoding** (when the checkpoint ships a drafter). Implement
   `megakernel.spec`'s `TargetBlock` and `DraftModel` for the model and run `SpecStep` as a chat
   `MultiTokenBackend`. Gates: the drafter against its reference implementation; the step,
   eager and replayed from a CUDA graph, equal to a host loop over the same verify forward and
   drafter; acceptance rates measured on varied prompts before claiming a speedup.
8. **Multi-token verify.** A block form of the per-layer kernels on `megakernel.multi_gemv` as
   the `TargetBlock`, and the drafter as launches, so draft and verify are megakernel launches
   (acceptance stays a few torch kernels); prefill through `spec.prefill_blocks`. Gates: every
   exposed block size against the oracle, NaN-filled masked rows, the one-token kernels' PTX
   unchanged, and the verify cost per `k` measured against the one-token step.

## Moving code into the package

- Move only what the second user needs or what has no model semantics (AGENTS.md). Update every
  importer; leave no alias in the model directory.
- A move must not change kernels: dump each affected kernel's PTX before and after with
  `attn_gym.warpdrive.ptx_check.capture_ptx` (compile the uncached function,
  `compile_fn.__wrapped__(op)`) and compare; any difference needs an explanation. PTX covers
  device code only: a move that touches host code (constructors, runners, sizing) still needs
  the models' GPU suites.

## Before review

- `ruff check && ruff format --check`, then
  `pytest -n 6 test/warpdrive/test_warpdrive_megakernel_package.py` and the model's CPU tests.
- Each model's GPU suites and chat smoke on every commit that touches the package.
- Bench A/B against the base commit for every model whose kernels or runners changed.
- Extend `megakernel/AGENTS.md` for new shared pieces, and `docs/warpdrive/examples.md` for
  the model.
