"""TMA layout limits and pointer fallback in the paged indexer's CUDA dispatch."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym._backends.triton.utils import can_use_tensor_descriptor, can_use_tma
from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import assert_paged_indexer_selection, make_paged_indexer_test_inputs

requires_tma = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability()[0] < 9,
    reason="requires NVIDIA hardware TMA",
)


@requires_tma
@pytest.mark.parametrize(
    "layout, eligible",
    [
        ("dlpack", False),
        ("stride-limit", False),
        ("last-valid-stride", True),
        ("broadcast", True),
        ("last-i32-extent", True),
        ("oversized-extent", False),
    ],
)
def test_paged_indexer_tma_layout(layout, eligible):
    args = list(make_paged_indexer_test_inputs((2,), (128,), page_size=32, head_dim=16))
    q, k = args[:2]
    match layout:
        case "dlpack":
            misaligned = q.new_empty(q.numel() + 1)[1:].view_as(q)
            args[0] = torch.from_dlpack(misaligned.copy_(q))
            assert args[0].storage_offset() == 0 and args[0].data_ptr() % 16 == 2
        case "stride-limit" | "last-valid-stride":
            # A singleton page needs no extra storage, but its stride is encoded verbatim.
            byte_stride = 2**40 if layout == "stride-limit" else 2**40 - 16
            args[1] = k.as_strided(k.shape, (byte_stride // k.element_size(), *k.stride()[1:]))
        case "broadcast":
            args[0] = q[:, :1].expand_as(q)
        case "last-i32-extent" | "oversized-extent":
            pages = 2**31 - 1 if layout == "last-i32-extent" else 2**32 + 1
            args[1] = k.expand(pages, -1, -1)

    # Exercise the real-pointer dispatch inside an already captured custom op.
    indexer = (
        torch.compile(paged_lightning_indexer, fullgraph=True, dynamic=True, backend="eager")
        if layout == "dlpack"
        else paged_lightning_indexer
    )
    actual = indexer(*args, topk=3, kernel_options={"backend": "triton"})
    assert_paged_indexer_selection(args, actual, 3)
    assert all(map(can_use_tma, args[:2])) == eligible


def test_tensor_descriptor_fake_metadata():
    with FakeTensorMode():
        base = torch.empty(16, device="cuda", dtype=torch.float16)
        assert can_use_tensor_descriptor(base)
        assert not can_use_tensor_descriptor(base[1:])
        assert not can_use_tma(base)  # A fake has no verifiable address for hardware dispatch.
