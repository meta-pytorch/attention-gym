"""A differentiated warpdrive KDA forward runs the kernels its backward recomputes.

Small packs share a decay/factor prep pass between the d_v split's two value tiles when the
forward is inference-only. The backward recomputes the per-tile formulation, which rounds the
chunk states apart from the shared prep, so a forward that records a graph runs per-tile.
``kernel_options["exact_recompute"]`` overrides the inference, for ``chunk_kda`` and for the
staged handles that context parallelism differentiates inside its own autograd function.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
# Shapes whose inference forward shares the prep on a GB300 (152 SMs): the first pairs it with a
# chain backward, the second with the warmup backward.
PREP_SHAPES = [pytest.param(2048, 8, id="1x2048xH8"), pytest.param(272, 2, id="1x272xH2")]
PRECISIONS = [
    *(pytest.param(torch.bfloat16, value, id=value) for value in ("bf16", "tf32+bf16", "tf32")),
    pytest.param(torch.float16, None, id="fp16"),
]


def _inputs(tokens, heads, dtype=torch.bfloat16, seed=0, gate_scale=5.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    shape = (1, tokens, heads, 128)
    q, k = (F.normalize(normal(*shape), dim=-1).to(dtype) for _ in range(2))
    v = normal(*shape).to(dtype)
    gate = -gate_scale * torch.rand(*shape, device="cuda", generator=gen)
    beta = torch.rand(*shape[:3], device="cuda", generator=gen)
    return q, k, v, gate, beta


def _leaves(inputs, requires_grad=True):
    return [tensor.detach().clone().requires_grad_(requires_grad) for tensor in inputs]


def _options(value):
    return WARPDRIVE if value is None else {**WARPDRIVE, "intra_chunk_dtypes": value}


@pytest.mark.parametrize("gate_scale", [5.0, 0.01], ids=["strong_gates", "weak_gates"])
@pytest.mark.parametrize("dtype, value", PRECISIONS)
@pytest.mark.parametrize("tokens, heads", PREP_SHAPES)
def test_handoff_gradients_match_one_call(tokens, heads, dtype, value, gate_scale) -> None:
    """The last 16 tokens' outputs and gradients of one differentiated call equal those of a
    call started from the final state of a differentiated call over the first ``T - 16``: the
    backward's checkpoints are the states the forward ran. The first shape's uncut forward hands
    its chain backward the pieces' entry states (NOTE [Piece entry states]); piece summaries
    composed in FP32 would round apart from it, which weak gates do not decay away."""
    options = _options(value)
    inputs = _inputs(tokens, heads, dtype, gate_scale=gate_scale)
    cut = tokens - 16
    d_output = torch.randn_like(inputs[2])

    leaves = _leaves(inputs)
    output, _ = chunk_kda(*leaves, kernel_options=options)
    expected = torch.autograd.grad(output, leaves, d_output)

    head = _leaves(t[:, :cut] for t in inputs)
    _, handoff = chunk_kda(*head, output_final_state=True, kernel_options=options)
    tail = _leaves(t[:, cut:] for t in inputs)
    tail_output, _ = chunk_kda(*tail, initial_state=handoff.detach(), kernel_options=options)
    actual = torch.autograd.grad(tail_output, tail, d_output[:, cut:].contiguous())

    assert torch.equal(tail_output, output[:, cut:])
    for name, got, want in zip(("dq", "dk", "dv", "dgate", "dbeta"), actual, expected):
        assert torch.equal(got, want[:, cut:]), name


def _paired_pieces(lengths, heads):
    """Pieces per sequence of the chain backward this GPU pairs with an uncut forward over this
    pack (NOTE [Piece entry states]), 0 when its plans do not pair that way."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    return kda.entry_state_pieces(sum(lengths), len(lengths), heads, 128, torch.device("cuda"))


def _ceil_div(a, b):
    return -(-a // b)


def _respans(lengths, pieces):
    """Whether the piece table recomputes its span for this pack: the per-sequence slot
    ceilings of ``ceil(chunks / wave)``-chunk pieces overflow the wave (``piece_table_body``)."""
    wave = len(lengths) * pieces
    span = max(1, _ceil_div(_ceil_div(sum(lengths), 16), wave))
    return sum(max(1, _ceil_div(_ceil_div(n, 16), span)) for n in lengths) > wave


# Packs to draw fixtures from, most heads first. The plans follow the SM budget, and fewer heads
# leave each sequence more slots, so every pool pairs on the budgets SM100 parts expose
# (test_every_sm_budget_has_fixtures).
ONE_SEQUENCE = [([512], heads) for heads in (8, 4, 2, 1)]
LONG_SEQUENCE = [([2048], heads) for heads in (8, 4, 2, 1)]
RAGGED = [*(([300, 700, 0, 1000], heads) for heads in (4, 2, 1)), ([600, 1400, 0, 2000], 1)]
# A long sequence behind empty ones, whose slot ceilings overflow the wave (``_respans``): the
# forward's serial mirror (``piece_chain.sequence_pieces``) must respan as the table does.
RESPAN = [*(([0, 0, 0, 1792], heads) for heads in (4, 2, 1)), ([0, 0, 0, 3584], 1)]
POOLS = {
    "one_sequence": (ONE_SEQUENCE, None),
    "long_sequence": (LONG_SEQUENCE, None),
    "ragged": (RAGGED, None),
    "respan": (RESPAN, _respans),
}


def _find(pool, require=None):
    """The first ``(lengths, heads, pieces)`` of ``pool`` whose plans pair on this GPU and that
    ``require(lengths, pieces)`` accepts, None when there is none."""
    for lengths, heads in pool:
        pieces = _paired_pieces(lengths, heads)
        if pieces and (require is None or require(lengths, pieces)):
            return lengths, heads, pieces
    return None


def _paired(name):
    """``_find`` over the named pool; skip when this GPU's SM budget pairs none of it."""
    found = _find(*POOLS[name])
    if found is None:
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        pytest.skip(f"no {name} pack pairs an uncut forward with a chain backward on {num_sm} SMs")
    return found


def _seed_probe(tokens, heads, dtype):
    """An FP32 entry state the 16-bit operands cannot hold, ``S[0, 0] = 1 + eps``, read through
    ``q`` at the last token (inside the last backward piece), with ``dO`` there: the forward
    gives ``o[-1] = eps``, and the backward's checkpoint gives ``dq[-1] = eps`` only when it
    starts from the forward's own states. Piece states composed from summaries lose it
    (``dq[-1] = 0``)."""
    eps = 2**-12 if dtype == torch.float16 else 2**-10
    shape = (1, tokens, heads, 128)
    q, k, v, d_output = (torch.zeros(shape, device="cuda", dtype=dtype) for _ in range(4))
    gate = torch.zeros(shape, device="cuda")
    beta = torch.zeros(shape[:3], device="cuda")
    k[0, 0, :, 0] = 1
    beta[0, 0] = 1
    q[0, -1, :, 0] = 1
    d_output[0, -1, :, 0] = 1
    state = torch.zeros(1, heads, 128, 128, device="cuda")
    state[:, :, 0, 0] = 1 + eps
    return (q, k, v, gate, beta), d_output, state, eps


@pytest.mark.parametrize("dtype, value", PRECISIONS)
def test_fp32_initial_state_reaches_the_chain_backward(dtype, value) -> None:
    """``_seed_probe`` through ``chunk_kda``."""
    lengths, heads, _ = _paired("one_sequence")
    (q, k, v, gate, beta), d_output, state, eps = _seed_probe(lengths[0], heads, dtype)
    leaves = _leaves((q, k, v))
    output, _ = chunk_kda(
        *leaves, gate, beta, initial_state=state, scale=1.0, kernel_options=_options(value)
    )
    (dq,) = torch.autograd.grad(output, leaves[:1], d_output)

    assert output[0, -1, :, 0].eq(eps).all()
    assert torch.equal(dq[0, -1, :, 0], output[0, -1, :, 0])


def test_fp32_initial_state_reaches_a_staged_fallback_backward() -> None:
    """``_seed_probe`` through the staged handles when their pieces cannot launch: an FP32 gate
    whose singleton head stride needs 64-bit offsets. The summaries may split (on a GB300 they
    do), but ``run`` and the backward then take the public op's plans, so ``run`` must record
    the entry states (it did not: ``dq[-1] = 0``) and the tape read before it is rejected."""
    from attn_gym._backends.cute.utils import requires_int64_abi
    from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward

    tokens, heads = 512, 1
    if not _paired_pieces([tokens], heads):
        pytest.skip("this GPU does not pair an uncut forward with a chain backward here")
    (q, k, v, gate, beta), d_output, state, eps = _seed_probe(tokens, heads, torch.bfloat16)
    gate = gate.as_strided(gate.shape, (tokens * 128, 128, 2**31, 1))
    assert requires_int64_abi(gate)
    prepared = chunk_kda_prepare(
        *_leaves((q, k, v)), gate, beta, scale=1.0, kernel_options=WARPDRIVE
    )
    assert prepared.pieces is None
    before = prepared.saved
    with torch.no_grad():
        output, _ = prepared.run(state)
    dq = chunk_kda_prepare_backward(
        prepared.saved, d_output, state, scale=prepared.scale, kernel_options=WARPDRIVE
    ).run()[0]

    assert output[0, -1, :, 0].eq(eps).all()
    assert torch.equal(dq[0, -1, :, 0], output[0, -1, :, 0])
    with pytest.raises(ValueError, match="no piece entry states"):
        chunk_kda_prepare_backward(
            before, d_output, state, scale=prepared.scale, kernel_options=WARPDRIVE
        )


@pytest.mark.parametrize("dtype, value", [PRECISIONS[1], PRECISIONS[3]])
@pytest.mark.parametrize("pool", ["one_sequence", "ragged", "respan"])
def test_entry_states_are_the_forward_states(pool, dtype, value) -> None:
    """The forward stores, for each piece of the chain backward's piece table, the FP32 state
    of the prefix before it: the final state of a forward over that prefix. The ``respan``
    pack checks the forward's serial span against the table's overflow correction."""
    from attn_gym.linear._delta_rule.warpdrive import piece_chain
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    lengths, heads, pieces = _paired(pool)
    tokens, num_seqs = sum(lengths), len(lengths)
    q, k, v, gate, beta = (t[0] for t in _inputs(tokens, heads, dtype, gate_scale=0.01))
    gen = torch.Generator(device="cuda").manual_seed(1)
    state = torch.randn(num_seqs, heads, 128, 128, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
    cu = cu.to(torch.int32)
    options = {"scale": 128**-0.5, "intra_chunk_dtypes": value or "bf16", "exact_recompute": True}
    rows = num_seqs * pieces
    entry = torch.full(kda.entry_state_shape(rows, heads, 128, 128), torch.nan, device="cuda")
    output, _ = kda.kda_forward(q, k, v, gate, beta, cu, initial_state=state, **options)
    stored, _ = kda.kda_forward(
        q, k, v, gate, beta, cu, initial_state=state, entry_states=entry, **options
    )
    assert torch.equal(stored, output)

    table = piece_chain.piece_table_layout(num_seqs, pieces, heads)
    words = torch.zeros(table.nbytes // 4, dtype=torch.int32, device="cuda")
    cu_pieces = words[table.cu_pieces // 4 :][: rows + 1]
    main_rows = words[table.main_rows // 4 :][: num_seqs + 1]
    piece_chain.build_chain_tables(
        cu,
        {
            "cu_pieces": cu_pieces,
            "main_rows": main_rows,
            "summary_rows": words[table.summary_rows // 4 :][: num_seqs + 1],
            "main_count": words[table.main_count // 4 :][:1],
            "summary_count": words[table.summary_count // 4 :][:1],
            "work_items": torch.empty(rows * heads, 10, dtype=torch.int32, device="cuda"),
            "scheduler": torch.zeros(1, dtype=torch.int32, device="cuda"),
        },
        pieces=pieces,
        heads_out=heads,
        b_t=16,
    )
    entry = entry.permute(0, 1, 3, 2, 4).reshape(rows, heads, 128, 128)
    for seq in range(num_seqs):
        first, last = (main_rows[seq : seq + 2] // heads).tolist()
        start = int(cu[seq])
        if int(cu[seq + 1]) == start:
            continue  # an empty sequence runs no piece
        for slot in range(first, last):
            stop = int(cu_pieces[slot])
            prefix = slice(start, stop)
            _, want = kda.kda_forward(
                *(t[prefix] for t in (q, k, v, gate, beta)),
                torch.tensor([0, stop - start], dtype=torch.int32, device="cuda"),
                initial_state=state[seq : seq + 1].clone(),
                output_final_state=True,
                **options,
            )
            assert torch.equal(entry[slot], want[0]), (seq, slot)


@pytest.mark.parametrize(
    "batch, tokens, heads",
    [(1, 512, 8), (1, 2048, 8), (2, 2048, 4), (1, 8192, 16), (8, 2048, 16), (1, 2048, 64)],
)
def test_entry_state_buffer_holds_one_state_per_backward_piece(batch, tokens, heads) -> None:
    """The extra memory a differentiated forward keeps for its backward: one FP32
    ``[H, V, K]`` state per chain backward piece, and nothing when the forward chains too or
    the backward runs uncut."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda
    from attn_gym.linear.kda.impl.warpdrive import entry_state_rows
    from attn_gym.linear.kda.impl.warpdrive_ops import chunk_warpdrive_packed_fwd_entry_states_op

    q, k, v, gate, beta = _inputs(batch * tokens, heads)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * tokens
    rows = entry_state_rows(gate, v, cu)
    device = q.device
    forward = kda.ForwardPlan.build(batch * tokens, batch, heads, 128, device)
    backward = kda.BackwardPlan.build(batch * tokens, batch, heads, 128, device)
    if forward.chain or not backward.chain:
        assert rows is None
        return
    _, _, entry = chunk_warpdrive_packed_fwd_entry_states_op(
        q, k, v, gate, beta, None, cu, 0.088, True, False, False, "tf32+bf16", rows
    )
    assert entry.dtype == torch.float32
    assert entry.numel() * 4 == batch * backward.pieces * heads * 128 * 128 * 4


@pytest.mark.parametrize("num_sm", [16, 20, 24, 32, 48, 64, 96, 120, 132, 148, 152, 160])
def test_every_sm_budget_has_fixtures(num_sm, monkeypatch) -> None:
    """The tests here skip when no pack of their pool pairs on this GPU; the pools pair on every
    SM budget an SM100 part or partition is likely to expose (patched into the planners), so
    those skips cannot hide."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import plan

    monkeypatch.setattr(plan, "_num_sm", lambda device: num_sm)
    for name, (pool, require) in POOLS.items():
        assert _find(pool, require) is not None, name
    assert _compile_fixture() is not None


def test_compiled_entry_state_rows_hold_every_backward_piece() -> None:
    """Under ``torch.compile`` the rows come from the sequence count alone: enough for any
    token count's backward pieces, and none when no backward of that pack can chain."""
    from attn_gym.linear._delta_rule.warpdrive import piece_chain
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    device = torch.device("cuda")
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    for num_seqs, heads in ((1, 8), (1, 16), (1, 64), (4, 8), (8, 16)):
        rows = kda.entry_state_rows_bound(num_seqs, heads, device)
        for tokens in (96, 512, 1040, 2048, 8192, 32768):
            pieces = kda.BackwardPlan.build(num_seqs * tokens, num_seqs, heads, 128, device).pieces
            assert num_seqs * pieces <= rows, (num_seqs, heads, tokens)
        if num_sm // (num_seqs * heads) < piece_chain.CHAIN_MIN_PIECES_REVERSE:
            assert rows == 0, (num_seqs, heads)


def _compile_fixture():
    """``(heads, packs, crowded)`` for the compiled graphs, each pack a list of sequence lengths:
    two whose plans pair (seeded backwards), one whose backward runs uncut and one long sequence
    whose forward chains too; then two packs of more sequences than leave two slots each (0-row
    buffers). None when no head count gives them all on this GPU."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda, plan

    device = torch.device("cuda")
    num_sm = plan._num_sm(device)
    for heads in (8, 4, 2, 1):
        if not all(_paired_pieces([tokens], heads) for tokens in (512, 1040)):
            continue
        if kda.BackwardPlan.build(96, 1, heads, 128, device).chain:
            continue
        for tokens in (4096, 8192, 16384):
            if kda.ForwardPlan.build(tokens, 1, heads, 128, device).chain and (
                kda.BackwardPlan.build(tokens, 1, heads, 128, device).chain
            ):
                crowded = num_sm // (2 * heads) + 1
                return (
                    heads,
                    [[512], [1040], [96], [tokens]],
                    [[64] * crowded, [48] * (crowded + 3)],
                )
    return None


def test_entry_states_under_fullgraph_compile() -> None:
    """The buffer is sized from the sequence count alone, so one ``torch.compile`` graph serves
    every token count's pairing of the plans: packs whose plans pair seed the backward, while an
    uncut backward and a forward that chains too leave the buffer unread (a stale read would
    move the gradients). Packs with under two slots per sequence allocate 0 rows; PyTorch's
    contiguity metadata guards on an empty buffer, so they share a second graph. The gradients
    keep the eager bits."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    fixture = _compile_fixture()
    if fixture is None:
        pytest.skip("this GPU's SM budget gives no head count every pairing")
    heads, packs, crowded = fixture
    for lengths in crowded:
        assert kda.entry_state_rows_bound(len(lengths), heads, torch.device("cuda")) == 0
    torch._dynamo.reset()
    compiled = torch.compile(chunk_kda, fullgraph=True, dynamic=True)

    def check(lengths):
        inputs = _inputs(sum(lengths), heads, gate_scale=0.01)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
        cu = cu.to(torch.int32)
        d_output = torch.randn_like(inputs[2])

        def grads(run):
            leaves = _leaves(inputs)
            output = run(*leaves, cu_seqlens=cu, kernel_options=WARPDRIVE)[0]
            return torch.autograd.grad(output, leaves, d_output)

        expected = grads(chunk_kda)
        actual = grads(compiled)
        for name, got, want in zip(("dq", "dk", "dv", "dgate", "dbeta"), actual, expected):
            assert torch.equal(got, want), (lengths[:2], name)

    with torch._dynamo.config.patch(error_on_recompile=True):
        for lengths in packs:
            check(lengths)
    check(crowded[0])
    with torch._dynamo.config.patch(error_on_recompile=True):
        check(crowded[1])


@pytest.mark.parametrize("pool", ["long_sequence", "ragged"])
def test_seeded_backward_keeps_the_state_cotangent(pool) -> None:
    """Seeding changes the recomputed states, not the reverse chain: ``d_initial_state`` keeps
    the bits of the backward that composes piece summaries, through several pieces, weak gates
    and a nonzero exit cotangent, while the token gradients move to the forward's states."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    lengths, heads, pieces = _paired(pool)
    tokens, num_seqs = sum(lengths), len(lengths)
    q, k, v, gate, beta = (t[0] for t in _inputs(tokens, heads, gate_scale=0.01))
    gen = torch.Generator(device="cuda").manual_seed(3)
    state, d_final = (
        torch.randn(num_seqs, heads, 128, 128, device="cuda", generator=gen) for _ in range(2)
    )
    d_output = torch.randn(v.shape, device="cuda", generator=gen).to(v.dtype)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
    cu = cu.to(torch.int32)
    common = {"scale": 128**-0.5, "initial_state": state, "intra_chunk_dtypes": "tf32+bf16"}
    entry = torch.empty(kda.entry_state_shape(num_seqs * pieces, heads, 128, 128), device="cuda")
    kda.kda_forward(q, k, v, gate, beta, cu, exact_recompute=True, entry_states=entry, **common)

    def backward(entry_states):
        return kda.kda_backward(
            q, k, v, gate, beta, d_output, cu, d_final_state=d_final, entry_states=entry_states,
            **common,
        )  # fmt: skip

    seeded, composed = backward(entry), backward(None)
    assert torch.equal(seeded[5], composed[5])
    assert not all(torch.equal(a, b) for a, b in zip(seeded[:5], composed[:5]))


def test_entry_state_ops_pass_opcheck() -> None:
    """The forward op that returns ``entry_states`` and the backward op that reads them."""
    from attn_gym.linear.kda.impl.warpdrive import entry_state_rows
    from attn_gym.linear.kda.impl.warpdrive_ops import (
        chunk_warpdrive_packed_fwd_entry_states_op,
        chunk_warpdrive_packed_local_bwd_op,
    )

    (tokens,), heads, _ = _paired("one_sequence")
    q, k, v, gate, beta = _inputs(tokens, heads, gate_scale=0.01)
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    rows = entry_state_rows(gate, v, cu)
    state = torch.randn(1, heads, 128, 128, device="cuda")
    utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    for initial_state, final in ((None, False), (None, True), (state, True)):
        torch.library.opcheck(
            chunk_warpdrive_packed_fwd_entry_states_op,
            (q, k, v, gate, beta, initial_state, cu, 0.088, True, False, final, "bf16", rows),
            test_utils=utils,
        )
    _, _, entry = chunk_warpdrive_packed_fwd_entry_states_op(
        q, k, v, gate, beta, None, cu, 0.088, True, False, False, "tf32+bf16", rows
    )
    d_output = torch.randn_like(v)
    torch.library.opcheck(
        chunk_warpdrive_packed_local_bwd_op,
        (q, k, v, gate, beta, d_output, cu, False, 0.088, True, "tf32+bf16", False, entry),
        test_utils=utils,
    )


def test_entry_states_must_be_a_buffer_on_the_device() -> None:
    """The driver checks the buffer before a kernel reads or writes through its pointer: one on
    another device or of another rank raises ``ValueError``."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    (tokens,), heads, pieces = _paired("one_sequence")
    q, k, v, gate, beta = (t[0] for t in _inputs(tokens, heads))
    cu = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    common = {"scale": 128**-0.5, "intra_chunk_dtypes": "bf16"}
    on_host = torch.empty(kda.entry_state_shape(pieces, heads, 128, 128))
    for entry in (on_host, torch.empty((), device="cuda")):
        with pytest.raises(ValueError, match="entry_states"):
            kda.kda_forward(
                q, k, v, gate, beta, cu, exact_recompute=True, entry_states=entry, **common
            )
        with pytest.raises(ValueError, match="entry_states"):
            kda.kda_backward(q, k, v, gate, beta, v, cu, entry_states=entry, **common)


def test_saved_tensor_hooks_may_restore_other_strides() -> None:
    """``saved_tensors_hooks`` preserve values, not strides: restoring every saved tensor with
    its last two modes column-major, the piece entry states included, keeps the gradients'
    bits (the backward rejected such entry states)."""
    (tokens,), heads, _ = _paired("one_sequence")
    inputs = _inputs(tokens, heads, gate_scale=0.01)
    d_output = torch.randn_like(inputs[2])

    def unpack(tensor):
        if tensor.ndim < 2:
            return tensor
        return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)

    def grads():
        leaves = _leaves(inputs)
        output = chunk_kda(*leaves, kernel_options=WARPDRIVE)[0]
        return torch.autograd.grad(output, leaves, d_output)

    expected = grads()
    with torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, unpack):
        actual = grads()
    for name, got, want in zip(("dq", "dk", "dv", "dgate", "dbeta"), actual, expected):
        assert torch.equal(got, want), name


def test_only_a_recorded_forward_stores_entry_states(monkeypatch) -> None:
    """``exact_recompute=True`` without an autograd tape (``no_grad``, or no input requiring
    grad) keeps the per-tile math but stores no piece entry states, which no backward reads."""
    from attn_gym.linear.kda.impl import warpdrive as impl

    (tokens,), heads, _ = _paired("one_sequence")
    inputs = _inputs(tokens, heads)
    options = {**WARPDRIVE, "exact_recompute": True}
    stores = []
    entry_states_op = impl.chunk_warpdrive_packed_fwd_entry_states_op

    def recording_op(*args):
        stores.append(args[-1])
        return entry_states_op(*args)

    monkeypatch.setattr(impl, "chunk_warpdrive_packed_fwd_entry_states_op", recording_op)
    recorded = chunk_kda(*_leaves(inputs), kernel_options=options)[0].detach()
    assert len(stores) == 1
    with torch.no_grad():
        no_grad = chunk_kda(*_leaves(inputs), kernel_options=options)[0]
    without_grad_inputs = chunk_kda(*_leaves(inputs, False), kernel_options=options)[0]

    assert len(stores) == 1
    assert torch.equal(no_grad, recorded)
    assert torch.equal(without_grad_inputs, recorded)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_only_inference_forwards_share_the_prep(compiled, monkeypatch) -> None:
    """Under grad the forward returns the per-tile bits; under ``no_grad``, ``inference_mode``
    or without inputs that require grad it keeps the shared prep's. ``torch.compile`` decides
    while tracing."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    inputs = _inputs(2048, 8)
    run = torch.compile(chunk_kda, fullgraph=True) if compiled else chunk_kda

    def forward(requires_grad=True):
        return run(*_leaves(inputs, requires_grad), kernel_options=WARPDRIVE)[0].detach()

    training = forward()
    without_grad_inputs = forward(requires_grad=False)
    with torch.no_grad():
        no_grad = forward()
    with torch.inference_mode():
        inference = chunk_kda(*inputs, kernel_options=WARPDRIVE)[0]
    monkeypatch.setattr(kda, "PREP_TILE_FRACTION", 0.0)
    with torch.no_grad():
        per_tile = chunk_kda(*inputs, kernel_options=WARPDRIVE)[0]

    assert torch.equal(training, per_tile)
    assert not torch.equal(no_grad, per_tile)
    assert torch.equal(without_grad_inputs, no_grad)
    assert torch.equal(inference, no_grad)


def _per_tile(monkeypatch):
    """Make every later warpdrive KDA forward run per-tile, as the backward recomputes."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    monkeypatch.setattr(kda, "PREP_TILE_FRACTION", 0.0)


def test_exact_recompute_overrides_the_grad_mode_inference(monkeypatch) -> None:
    """``True`` runs per-tile under ``no_grad``; ``False`` shares the prep under grad."""
    inputs = _inputs(2048, 8)

    def forward(exact_recompute=None, requires_grad=True):
        options = dict(WARPDRIVE)
        if exact_recompute is not None:
            options["exact_recompute"] = exact_recompute
        return chunk_kda(*_leaves(inputs, requires_grad), kernel_options=options)[0].detach()

    exact_under_no_grad = forward(True, requires_grad=False)
    inexact_under_grad = forward(False)
    with torch.no_grad():
        shared = forward()
    _per_tile(monkeypatch)
    with torch.no_grad():
        per_tile = forward()

    assert not torch.equal(shared, per_tile)
    assert torch.equal(exact_under_no_grad, per_tile)
    assert torch.equal(inexact_under_grad, shared)


# One sequence whose staged summaries run uncut on a GB300, so ``run`` takes the public op's
# plan, and whose inference forward shares the prep.
STAGED_SHAPE = (256, 8)


def _staged_run(inputs, kernel_options=WARPDRIVE):
    """Prepare and run one staged forward from a zero entry state under ``no_grad``, as one CP
    rank does inside its autograd function."""
    from attn_gym.linear.kda.stages import chunk_kda_prepare

    with torch.no_grad():
        prepared = chunk_kda_prepare(*inputs, kernel_options=kernel_options)
        assert prepared.pieces is None
        entry = torch.zeros(1, inputs[0].shape[2], 128, 128, device="cuda")
        return prepared.run(entry, output_final_state=True)


def test_staged_run_infers_exact_recompute_at_prepare(monkeypatch) -> None:
    """A handle prepared under autograd runs per-tile; ``False`` keeps the shared prep and
    ``True`` drops it, whatever the grad mode."""
    from attn_gym.linear.kda.stages import chunk_kda_prepare

    inputs = _inputs(*STAGED_SHAPE)
    entry = torch.zeros(1, inputs[0].shape[2], 128, 128, device="cuda")

    def run(exact_recompute=None):
        options = dict(WARPDRIVE)
        if exact_recompute is not None:
            options["exact_recompute"] = exact_recompute
        prepared = chunk_kda_prepare(*_leaves(inputs), kernel_options=options)
        assert prepared.pieces is None
        with torch.no_grad():
            return prepared.run(entry, output_final_state=True)

    inferred_under_grad = run()
    inexact_under_grad = run(False)
    exact_under_no_grad = _staged_run(inputs, {**WARPDRIVE, "exact_recompute": True})
    shared = _staged_run(inputs)
    _per_tile(monkeypatch)
    per_tile = _staged_run(inputs)

    assert not torch.equal(shared[0], per_tile[0])
    for got, want in (
        (inferred_under_grad, per_tile),
        (inexact_under_grad, shared),
        (exact_under_no_grad, per_tile),
    ):
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_context_parallel_kda_runs_the_exact_forward(monkeypatch) -> None:
    """One rank owning the whole document: its staged forward runs inside an autograd function,
    where grad mode is off, so ``context_parallel_kda`` must ask for the per-tile kernels."""
    import attn_gym.linear.context_parallel as cp_impl
    from attn_gym.linear.context_parallel import ContextParallelPlan
    from attn_gym.linear.kda import context_parallel_kda

    tokens, _ = STAGED_SHAPE
    inputs = _inputs(*STAGED_SHAPE)
    monkeypatch.setattr(cp_impl.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(cp_impl.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(cp_impl, "_all_gather_slots", lambda slots, group: slots.unsqueeze(0))
    routing = ContextParallelPlan.from_fragments((0, tokens), [[(0, tokens)]], 0).routing("cuda")
    output, _ = context_parallel_kda(
        *_leaves(inputs), routing=routing, group=None, kernel_options=WARPDRIVE
    )
    shared = _staged_run(inputs)
    _per_tile(monkeypatch)
    per_tile = _staged_run(inputs)

    assert not torch.equal(shared[0], per_tile[0])
    assert torch.equal(output.detach(), per_tile[0])


def test_staged_handles_hand_entry_states_to_the_backward() -> None:
    """Each ``run`` binds the tape to its own entry states (NOTE [Staged entry states]): two runs
    from different initial states each give, from the tape read after them (rebuilt as context
    parallelism does), the gradients of ``chunk_kda`` bit for bit. The tape read before any
    ``run`` holds none, and the backward rejects it rather than composing piece summaries."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import staged
    from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward

    # A pack whose staged summaries run uncut on a GB300 while its backward chains.
    num_seqs, length, heads = 4, 1536, 12
    tokens = num_seqs * length
    if not _paired_pieces([length] * num_seqs, heads):
        pytest.skip("this GPU does not pair an uncut forward with a chain backward here")
    inputs = _inputs(tokens, heads, gate_scale=0.01)
    cu = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32) * length
    if staged.plan(staged.KDA, inputs[2][0], cu) is not None:
        pytest.skip("this GPU splits the staged summaries here")
    gen = torch.Generator(device="cuda").manual_seed(2)
    states = [torch.randn(num_seqs, heads, 128, 128, device="cuda", generator=gen) for _ in "ab"]
    d_output = torch.randn_like(inputs[2])
    d_final = torch.randn(states[0].shape, device="cuda", generator=gen)

    def backward(tape, state):
        return chunk_kda_prepare_backward(
            tape, d_output, state, scale=prepared.scale, kernel_options=WARPDRIVE
        )

    prepared = chunk_kda_prepare(*_leaves(inputs), cu_seqlens=cu, kernel_options=WARPDRIVE)
    assert prepared.pieces is None
    before = prepared.saved
    runs = []
    for state in states:
        with torch.no_grad():
            staged_output, _ = prepared.run(state, output_final_state=True)
        runs.append((state, staged_output, type(prepared.saved)._make(tuple(prepared.saved))))
    with pytest.raises(ValueError, match="no piece entry states"):
        backward(before, states[0])

    for state, staged_output, tape in runs:
        leaves = _leaves(inputs)
        entry = state.clone().requires_grad_()
        output, final = chunk_kda(
            *leaves, initial_state=entry, cu_seqlens=cu, output_final_state=True,
            kernel_options=WARPDRIVE,
        )  # fmt: skip
        expected = torch.autograd.grad((output, final), (*leaves, entry), (d_output, d_final))
        grads = backward(tape, state).run(d_final)

        assert torch.equal(staged_output, output.detach())
        names = ("dq", "dk", "dv", "dgate", "dbeta", "dstate")
        for name, got, want in zip(names, grads, expected):
            assert torch.equal(got, want), name
