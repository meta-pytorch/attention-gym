"""Paged fresh-empty routes must remain in the native work queue for clearing."""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym.linear._delta_rule.cudnn_fe.common import split_k
from attn_gym.linear._delta_rule.cudnn_fe.common.tvm_ffi import (
    make_counter_signature,
    make_cu_seqlens_signature,
    make_paged_route_signatures,
    make_work_items_signature,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="cuDNN v1.30 helpers require Blackwell",
)


@cute.kernel
def paged_order_kernel(cu, routes, seeds, items, count):
    tidx = cute.arch.thread_idx()[0]
    keys = cutlass.Array(cutlass.Int32, 256, space=cutlass.AddressSpace.smem)
    indices = cutlass.Array(cutlass.Int32, 256, space=cutlass.AddressSpace.smem)
    spread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem)
    split_k.order_body(
        True,
        16,
        128,
        2,
        tidx,
        cutlass.Int32(2),
        cutlass.Int32(10),
        cu,
        None,
        count,
        items,
        None,
        keys,
        indices,
        spread,
        mStateIndices=routes,
        mHasInitialState=seeds,
    )


@cute.jit
def paged_order_launch(cu, routes, seeds, items, count, stream):
    paged_order_kernel(cu, routes, seeds, items, count).launch(
        grid=(1, 1, 1), block=(128, 1, 1), stream=stream
    )


@jit_cache
def compile_paged_order():
    routes, seeds = make_paged_route_signatures(5, has_initial_state=True)
    return compile_tvm_ffi(
        paged_order_launch,
        make_cu_seqlens_signature(6),
        routes,
        seeds,
        make_work_items_signature(10),
        make_counter_signature(),
        name="test_paged_order",
    )


def test_paged_order_keeps_only_nonempty_and_fresh_empty_routes():
    cu = torch.tensor([0, 0, 0, 0, 0, 16], dtype=torch.int32, device="cuda")
    # Fresh empty, resumed empty, two null empty routes, then a nonempty resumed route.
    routes = torch.tensor([1, 2, 0, -1, 3], dtype=torch.int32, device="cuda")
    seeds = torch.tensor([0, 1, 0, 0, 1], dtype=torch.uint8, device="cuda")
    items = torch.full((10, 10), -99, dtype=torch.int32, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    compile_paged_order()(cu, routes, seeds, items, count)
    assert count.item() == 4
    assert sorted(items[:4, 0].tolist()) == [0, 0, 4, 4]
