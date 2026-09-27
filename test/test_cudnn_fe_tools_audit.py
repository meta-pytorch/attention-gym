"""CPU-only tests for the cuDNN vendoring audit (tools/cudnn_fe/audit.py) and codemod (restyle.py)."""

import ast
import subprocess
import textwrap

import pytest

from tools.cudnn_fe import audit, closure, restyle

PR_A = "88eb5ce"  # the verbatim v1.30 drop

HEADER = """\
import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as nvvm
from ..tile_dsl.swizzle import swizzle_xor_128b
"""


def src(body: str) -> str:
    return HEADER + textwrap.dedent(body)


def rules(text: str) -> list[str]:
    return [f.rule for f in audit.audit_source(text, "kernel/x.py")]


@pytest.fixture(scope="module")
def house():
    """The current tree's tile_dsl helpers (house-style tma, swizzle and pointwise)."""
    return restyle.tile_helpers(audit.default_root())


@pytest.mark.parametrize(
    ("rule", "body"),
    [
        ("raw-nvvm-wrapper", "def arrive(mb):\n    nvvm.mbarrier_arrive(mb)\n"),
        ("raw-nvvm-wrapper", "def commit():\n    nvvm.cp_async_bulk_commit_group()\n"),
        (
            "inline-ptx-packed-f32",
            'def fmul2(a, b):\n    return cute.arch.inline_ptx("mul.f32x2 $0, $1, $2;")\n',
        ),
        ("rmem-array", "def f():\n    acc = cutlass.Array(cutlass.Float32, 8, alignment=16)\n"),
        (
            "smem-data-array",
            (
                "def f(cfg):\n    SMEM = cutlass.AddressSpace.smem\n"
                "    sK_raw = cutlass.Array(cutlass.Float16, 4096, space=SMEM, alignment=1024)\n"
            ),
        ),
        (
            "inline-swizzle-box",
            (
                "def f(r, c, cfg):\n"
                "    return (c // 64) * (cfg.b_t * 64) + r * 64 + swizzle_xor_128b(r, c % 64)\n"
            ),
        ),
        (
            "inline-swizzle-box",
            (
                "def f(r, c, cfg):\n    seg = c // 32\n    d = c - seg * 32\n"
                "    idx = seg * (cfg.b_t * 32) + r * 32 + swizzle_xor_128b(r, d, elem_bytes=4)\n"
            ),
        ),
        (
            "compile-outside-jit-cache",
            "def run(t):\n    return cute.compile(host, from_dlpack(t))\n",
        ),
        ("upstream-ref", "from cudnn.frost.device import current_device\n"),
        ("upstream-ref", "def run(gdp_lo):\n    pass\n"),
        ("pruned-knob", "def build_cfg(expand_num=1):\n    pass\n"),
        ("pruned-knob", "def f(cfg):\n    return cfg.safe_gate\n"),
        ("tma-helper-args", "def f(t, s, mb):\n    tma_load_tile(t, s, mb, acquire=False)\n"),
        (
            "sched-arrival-literal",
            "bars = Bars(mb_scheduler_done=MBarrier(alloc(2), stages=2, init_count=15))\n",
        ),
    ],
)
def test_rule_detects_upstream_form(rule, body):
    assert rule in rules(src(body))


# Aliased, fully qualified, positional and keyword spellings of the same upstream forms.
ALIASED_FORMS = {
    "nvvm_symbol_alias": (
        "raw-nvvm-wrapper",
        "from cutlass.experimental.primitives import mbarrier_arrive as arrive\narrive(mb)\n",
    ),
    "nvvm_attribute_chain": (
        "raw-nvvm-wrapper",
        "import cutlass._mlir.dialects.nvvm\ncutlass._mlir.dialects.nvvm.mbarrier_init(mb, 1)\n",
    ),
    "ptx_keyword": (
        "inline-ptx-packed-f32",
        'cute.arch.inline_ptx(ptx_code="mul.f32x2 $0, $1, $2;")\n',
    ),
    "rmem_alias": ("rmem-array", "from cutlass import Array as A\na = A(cutlass.Float32, 8)\n"),
    "smem_key_tile": (
        "smem-data-array",
        "sKeys = cutlass.Array(cutlass.Float16, 4096, space=cutlass.AddressSpace.smem)\n",
    ),
    "swizzle_alias": (
        "inline-swizzle-box",
        (
            "from ..tile_dsl.swizzle import swizzle_xor_128b as swz\n"
            "x = (c // 64) * (rows * 64) + r * 64 + swz(r, c % 64)\n"
        ),
    ),
    "compile_alias": (
        "compile-outside-jit-cache",
        "import cutlass.cute as c\nx = c.compile(h, 1)\n",
    ),
    "compile_bogus_decorator": (
        "compile-outside-jit-cache",
        "@not_jit_cache\ndef compile_host():\n    return cute.compile(host, 1)\n",
    ),
    "upstream_relative_alias": (
        "upstream-ref",
        "from ..kernel.gdp_bprop_v64_f16 import build_cfg as cfg\n",
    ),
    "knob_kwargs": ("pruned-knob", "def f(**safe_gate):\n    pass\n"),
    "tma_alias": (
        "tma-helper-args",
        "from ..tile_dsl.tma import tma_load_tile as load\nload(t, s, mb, acquire=False)\n",
    ),
    "sched_positional": (
        "sched-arrival-literal",
        "mb_scheduler_done = MBarrier(alloc(2), 2, 15)\n",
    ),
}


@pytest.mark.parametrize("case", sorted(ALIASED_FORMS))
def test_rule_resolves_imported_identities(case):
    rule, body = ALIASED_FORMS[case]
    assert rule in rules(body)


def test_house_style_forms_are_not_reported():
    text = src(
        """
        from attn_gym._backends.cute import jit_cache

        def arrive(mb):
            cute.arch.mbarrier_arrive(mb.data_ptr())
            while not nvvm.mbarrier_try_wait_parity(mb, 0, time_limit=1):  # waits stay raw
                pass

        def kernel(cfg):
            SMEM = cutlass.AddressSpace.smem
            acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            bars = cutlass.Array(cutlass.Int64, 4, space=SMEM, alignment=8)
            sScheduler = cutlass.Array(cutlass.Int32, 2, space=SMEM, alignment=16)
            # row * 64 + swizzle with col < 64 is a value-range judgment, not an exact box form
            k_off = r * 64 + swizzle_xor_128b(r, c)
            tma_load_tile(t, s, mb)
            done = MBarrier(alloc(2), init_count=cfg.threads_per_cta // 32 - 1)

        @jit_cache
        def _compile(n):
            return cute.compile(host, n)

        def doc():
            "Mentions cudnn.frost, expand_num and safe_gate only in a docstring."
        """
    )
    assert rules(text) == []


def test_allowlist_matches_rule_path_and_key(tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_text("rmem-array kernel/x\\.py (acc|tmp) -- documented exception\n")
    text = src(
        """
        def f():
            acc = cutlass.Array(cutlass.Float32, 8)
            other = cutlass.Array(cutlass.Float32, 8)
        """
    )
    entries = audit.load_allowlist(allow)
    remaining, allowed = audit.apply_allowlist(audit.audit_source(text, "kernel/x.py"), entries)
    assert [f.key for f in allowed] == ["acc"]
    assert [f.key for f in remaining] == ["other"]
    allow.write_text("rmem-array kernel/x\\.py acc\n")
    with pytest.raises(ValueError, match="reason"):
        audit.load_allowlist(allow)
    allow.write_text("rmem-array kernel/x\\.py .* -- every key\n")
    with pytest.raises(ValueError, match="evidenced keys"):
        audit.load_allowlist(allow)


CODEMOD_CASES = {
    "tma": (
        """
        from ..tile_dsl.tma import tma_load_tile, tma_store_tile

        def f(t, s, mb):
            tma_load_tile(t, s, mb, acquire=False)
            tma_store_tile(
                t,
                s,
                acquire=False,
            )
        """,
        """
        from ..tile_dsl.tma import tma_load_tile, tma_store_tile

        def f(t, s, mb):
            tma_load_tile(t, s, mb)
            tma_store_tile(
                t,
                s,
            )
        """,
    ),
    "barrier": (
        """
        def arrive(mb, n):
            nvvm.mbarrier_arrive(mb)
            nvvm.mbarrier_arrive_expect_tx(mb, n)
            nvvm.mbarrier_init(mb, n)
            nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)
            nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_2)
            nvvm.cp_async_bulk_commit_group()
            nvvm.cp_async_bulk_wait_group(0, read=True)
        """,
        """
        def arrive(mb, n):
            cute.arch.mbarrier_arrive(mb.data_ptr())
            cute.arch.mbarrier_arrive_and_expect_tx(mb.data_ptr(), n)
            cute.arch.mbarrier_init(mb.data_ptr(), n)
            tcgen05.commit(mb.data_ptr())
            nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_2)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        """,
    ),
    "rmem": (
        """
        def f(cfg, ptr):
            pack = cutlass.Array(cutlass.Int32, (cfg.b_t // 2), alignment=16)
            regs = cutlass.Array(cutlass.Float32, 4, space=cutlass.AddressSpace.rmem)
            half = cutlass.Array(cutlass.Int32, 8, alignment=16)
            pack[0] = regs[1]
            nvvm.tcgen05_st("16x128b", ptr, pack[0:(cfg.b_t // 2)])
            st(regs.data_ptr().load(count=4, alignment=4))
            st(half[0:4])
        """,
        """
        def f(cfg, ptr):
            pack = cute.make_rmem_tensor(((cfg.b_t // 2),), cutlass.Int32)
            regs = cute.make_rmem_tensor((4,), cutlass.Float32)
            half = cutlass.Array(cutlass.Int32, 8, alignment=16)
            pack[0] = regs[1]
            nvvm.tcgen05_st("16x128b", ptr, pack.load())
            st(regs.load())
            st(half[0:4])
        """,
    ),
    "swizzle": (
        """
        def f(r, c, v, dk, base, cfg):
            seg = c // 64
            seg_dim = c - seg * 64
            idx = seg * (cfg.b_t * 64) + r * 64 + swizzle_xor_128b(r, seg_dim, elem_bytes=2)
            addr = (
                base
                + (dk // 64) * (cfg.d_v * 64)
                + v * 64
                + swizzle_xor_128b(v, dk % 64)
            )
            f32 = (c // 32) * (cfg.b_t * 32) + r * 32 + swizzle_xor_128b(r, c % 32, elem_bytes=4)
            keep = (c // 64) * (cfg.b_t * 64) + r * 64 + swizzle_xor_128b(r ^ 1, c % 64)
            return idx, addr, f32, keep
        """,
        """
        def f(r, c, v, dk, base, cfg):
            idx = swizzle_box_offset_128b(r, c, box_rows=cfg.b_t)
            addr = base + swizzle_box_offset_128b(v, dk, box_rows=cfg.d_v)
            f32 = swizzle_box_offset_128b(r, c, box_rows=cfg.b_t, elem_bytes=4)
            keep = (c // 64) * (cfg.b_t * 64) + r * 64 + swizzle_xor_128b(r ^ 1, c % 64)
            return idx, addr, f32, keep
        """,
    ),
    "fadd2_fold": (
        """
        def f(a, b, c):
            vec = nvvm.add_packed_f32x2(
                cutlass.Vector.from_elements((a, b), cutlass.Float32),
                cutlass.Vector.from_elements((b, c), cutlass.Float32),
                ftz=False,
                rnd="rn",
            )
            x, y = cutlass.Float32(vec[0]), cutlass.Float32(vec[1])
            return x, y
        """,
        """
        def f(a, b, c):
            x, y = fadd2(a, b, b, c)
            return x, y
        """,
    ),
}

IMPORT_FIXES = {
    "barrier": ("from cutlass.cute.nvgpu import tcgen05\n", ""),
    "swizzle": ("", "swizzle_box_offset_128b, swizzle_xor_128b"),
}


@pytest.mark.parametrize("case", sorted(CODEMOD_CASES))
def test_codemod_rewrites_fixture_and_is_idempotent(case, house):
    before, after = (src(text) for text in CODEMOD_CASES[case])
    new_line, swizzle_names = IMPORT_FIXES.get(case, ("", ""))
    after = after.replace("from cutlass.experimental", new_line + "from cutlass.experimental")
    if swizzle_names:
        after = after.replace("import swizzle_xor_128b", f"import {swizzle_names}")
    if case == "fadd2_fold":
        before = before.replace(HEADER, HEADER + "from ..tile_dsl.pointwise import fmul2\n")
        after = after.replace(HEADER, HEADER + "from ..tile_dsl.pointwise import fadd2, fmul2\n")

    result = restyle.restyle_source(before, "kernel/x.py", helpers=house)
    assert result.text == after
    again = restyle.restyle_source(result.text, "kernel/x.py", helpers=house)
    assert again.text == result.text
    assert not again.applied


V130_POINTWISE = '''\
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm as nvvm_ops


def _packed_f32x2(op, vec_a, vec_b):
    """``op(vec_a, vec_b)`` for nvvm.{mul,add}_packed_f32x2 on f32x2 IR values."""
    if _PACKED_RES_FIRST:
        return op(vec_a.type, vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
    return op(vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)


def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add."""
    vec_a = cutlass.Vector.from_elements((a_lo, a_hi), cutlass.Float32)
    vec_b = cutlass.Vector.from_elements((b_lo, b_hi), cutlass.Float32)
    res = cutlass.Vector(_packed_f32x2(nvvm_ops.add_packed_f32x2, vec_a.ir_value(), vec_b.ir_value()), dtype=cutlass.Float32)
    return cutlass.Float32(res[0]), cutlass.Float32(res[1])
'''


def test_codemod_rewrites_v130_fadd2_body():
    result = restyle.restyle_source(V130_POINTWISE, "tile_dsl/pointwise.py")
    assert result.text.endswith(
        '    """Packed fp32 add."""\n'
        "    return cute.arch.add_packed_f32x2((a_lo, a_hi), (b_lo, b_hi))\n"
    )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("vec_b.ir_value()), dtype", "vec_a.ir_value()), dtype"),  # a + a
        ("((b_lo, b_hi), cutlass", "((b_hi, b_lo), cutlass"),  # swapped lanes
        ("return op(vec_a, vec_b, rnd", "return op(vec_b, vec_b, rnd"),  # helper body changed
    ],
)
def test_codemod_leaves_other_fadd2_bodies_manual(old, new):
    """``fadd2`` is rewritten only when its dataflow is the packed lane-wise add."""
    before = V130_POINTWISE.replace(old, new)
    assert before != V130_POINTWISE
    assert restyle.restyle_source(before, "tile_dsl/pointwise.py").text == before


SWIZZLE_MUTATED = """\
from ..tile_dsl.swizzle import swizzle_xor_128b


def f(r, c, rows):
    seg = c // 64
    seg_col = c % 64
    c += 64
    return seg * (rows * 64) + r * 64 + swizzle_xor_128b(r, seg_col)
"""


@pytest.mark.parametrize(
    "between",
    ["    c += 64\n", "    if r:\n        c = 0\n", "    seg_col = 3\n", "    (seg := 1)\n"],
)
def test_codemod_keeps_swizzle_when_bindings_are_not_straight_line(between, house):
    """A write to ``c``/``seg``/``seg_col`` or control flow before the chain keeps it manual."""
    before = SWIZZLE_MUTATED.replace("    c += 64\n", between)
    assert restyle.restyle_source(before, "kernel/x.py", helpers=house).text == before


def test_codemod_swizzle_rewrite_preserves_offsets(house):
    before = SWIZZLE_MUTATED.replace("    c += 64\n", "    unrelated = 1\n")
    after = restyle.restyle_source(before, "kernel/x.py", helpers=house).text
    assert "swizzle_box_offset_128b(r, c, box_rows=rows)" in after

    def swizzle_xor_128b(r, c, elem_bytes=2):
        return c ^ ((r & 7) * (16 // elem_bytes))

    def swizzle_box_offset_128b(r, c, *, box_rows):
        return (c // 64) * (box_rows * 64) + r * 64 + swizzle_xor_128b(r, c % 64)

    results = []
    for text in (before, after):
        ns = {
            "swizzle_xor_128b": swizzle_xor_128b,
            "swizzle_box_offset_128b": swizzle_box_offset_128b,
        }
        exec(text.split("\n", 1)[1], ns)  # noqa: S102 -- fixture minus its import
        results.append([ns["f"](r, c, 16) for r in range(9) for c in range(0, 256, 7)])
    assert results[0] == results[1]


def test_codemod_needs_the_helper_in_the_target_tree(house):
    """Without swizzle_box_offset_* in tile_dsl/swizzle.py the swizzle rewrite stays pending."""
    before = src(CODEMOD_CASES["swizzle"][0])
    assert restyle.restyle_source(before, "kernel/x.py").text == before
    helpers = dict(house, swizzle=audit.analyze("def swizzle_xor_128b(r, c): pass\n", "s.py"))
    assert restyle.restyle_source(before, "kernel/x.py", helpers=helpers).text == before
    unimported = before.replace("from ..tile_dsl.swizzle import swizzle_xor_128b\n", "")
    unimported = "def swizzle_xor_128b(r, c, elem_bytes=2):\n    pass\n" + unimported
    assert restyle.restyle_source(unimported, "kernel/x.py", helpers=house).text == unimported


V130_TMA = """\
def tma_load_tile(smem_tile, gmem_slice, mbar, *, cta_group=1, mcast_mask=None, acquire=True):
    pass
"""


@pytest.mark.parametrize(
    ("call", "helpers_tma", "rewritten"),
    [
        ("tma_load_tile(t, s, mb, acquire=False)", "house", True),
        ("tma_load_tile(t, s, mb, acquire=False)", V130_TMA, False),  # helper still fences
        ("tma_load_tile(t, s, mb, acquire=True)", "house", False),
        ("tma_load_tile(t, s, mb, cta_group=2)", "house", False),
        ("tma_load_tile(t, s, mb, mcast_mask=3)", "house", False),
        ("tma_load_tile(t, s, mb, cta_group=1)", "house", True),
    ],
)
def test_codemod_drops_only_tma_args_the_helper_hardcodes(call, helpers_tma, rewritten, house):
    before = f"from ..tile_dsl.tma import tma_load_tile\n\n\ndef f(t, s, mb):\n    {call}\n"
    helpers = dict(house)
    if helpers_tma != "house":
        helpers["tma"] = audit.analyze(helpers_tma, "tile_dsl/tma.py")
    after = restyle.restyle_source(before, "kernel/x.py", helpers=helpers).text
    assert (after != before) == rewritten
    if rewritten:
        assert "tma_load_tile(t, s, mb)\n" in after


def _defined_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names |= {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return names


def test_codemod_on_the_verbatim_drop_only_calls_existing_helpers(tmp_path):
    """Every ``from ..tile_dsl.X import name`` in the restyled 88eb5ce drop resolves."""
    has = subprocess.run(
        ["git", "-C", str(closure.repo_root()), "cat-file", "-e", f"{PR_A}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if has.returncode:
        pytest.skip(f"{PR_A} is not in this clone")
    root = audit.extract_rev(PR_A, tmp_path)
    results = restyle.restyle_tree(root, [], write=True)
    assert any(r.applied for r in results)
    missing = []
    for path in audit.iter_sources(root):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level and node.module):
                continue
            target = path.parent.joinpath(*[".."] * (node.level - 1), *node.module.split("."))
            module = target.with_suffix(".py").resolve()
            if not module.is_file():
                continue
            defined = _defined_names(ast.parse(module.read_text()))
            missing += [
                f"{path.relative_to(root)}: {node.module}.{a.name}"
                for a in node.names
                if a.name not in defined
            ]
    assert missing == []


def test_codemod_skips_allowlisted_findings(tmp_path):
    allow = tmp_path / "allow.txt"
    allow.write_text("rmem-array kernel/x\\.py acc -- SASS-driven exception\n")
    text = src(
        """
        def f():
            acc = cutlass.Array(cutlass.Float32, 8)
            other = cutlass.Array(cutlass.Float32, 8)
        """
    )
    result = restyle.restyle_source(text, "kernel/x.py", audit.load_allowlist(allow))
    assert "acc = cutlass.Array(cutlass.Float32, 8)" in result.text
    assert "other = cute.make_rmem_tensor((8,), cutlass.Float32)" in result.text
