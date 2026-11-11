"""``autotune`` is a kernel option of every tuned operation, not a keyword argument."""

import pytest
import torch

from attn_gym import linear, mla, sparse
from attn_gym._backends.autotune import resolve_tuning_options
from attn_gym.linear import gdn, kda
from attn_gym.linear._delta_rule.validation import resolve_recurrent_kernel_options
from attn_gym.linear.gdn.validation import resolve_kernel_options as resolve_gdn_options
from attn_gym.linear.kda.validation import resolve_kernel_options as resolve_kda_options
from attn_gym.mla import paged_mla_decode

RESOLVERS = {
    "chunk_kda": resolve_kda_options,
    "chunk_gdn": resolve_gdn_options,
    "recurrent": lambda options: resolve_recurrent_kernel_options(options, op_name="recurrent"),
    "tuning_only": lambda options: resolve_tuning_options(options, op_name="tuning_only"),
}
# Module and name, because the staged and context-parallel operations load lazily.
TUNED_OPERATIONS = [
    (mla, "paged_mla_decode"),
    (sparse, "gather_attn"),
    (linear, "chunk_kda"),
    (linear, "paged_chunk_kda"),
    (linear, "recurrent_kda"),
    (linear, "chunk_gdn"),
    (linear, "paged_chunk_gdn"),
    (linear, "recurrent_gdn"),
    (linear, "l2norm"),
    (kda, "chunk_kda_prepare"),
    (kda, "chunk_kda_prepare_backward"),
    (kda, "context_parallel_kda"),
    (gdn, "chunk_gdn_prepare"),
    (gdn, "chunk_gdn_prepare_backward"),
    (gdn, "context_parallel_gdn"),
]


def delta_rule_inputs() -> tuple[torch.Tensor, ...]:
    q = torch.randn(1, 4, 1, 8)
    return q, q, q, -torch.rand(1, 4, 1), torch.rand(1, 4, 1)


@pytest.mark.parametrize("resolve", RESOLVERS.values(), ids=RESOLVERS)
def test_resolvers_validate_the_autotune_option(resolve):
    assert resolve(None) == resolve({}) == resolve({"autotune": True})
    assert resolve({"autotune": False}) != resolve(None)
    for value in (0, 1, "False", None):
        with pytest.raises(TypeError, match=r"kernel_options\['autotune'\] must be a bool"):
            resolve({"autotune": value})
    with pytest.raises(ValueError, match="unsupported .* kernel options: tune"):
        resolve({"tune": False})


def test_cudnn_accepts_the_autotune_option():
    for resolve in (resolve_kda_options, resolve_gdn_options):
        options = resolve({"backend": "cudnn", "autotune": False})
        assert (options.backend, options.autotune) == ("cudnn", False)


@pytest.mark.parametrize("operation", [linear.chunk_gdn, linear.recurrent_gdn])
def test_operations_reject_a_non_bool_autotune_option(operation):
    with pytest.raises(TypeError, match="must be a bool"):
        operation(*delta_rule_inputs(), kernel_options={"autotune": 1})
    with pytest.raises(TypeError, match="must be a bool"):
        operation(*delta_rule_inputs(), impl="reference", kernel_options={"autotune": 1})


def kda_inputs() -> tuple[torch.Tensor, ...]:
    q, k, v, _gate, beta = delta_rule_inputs()
    return q, k, v, -torch.rand(1, 4, 1, 8), beta


REFERENCE_OPTIONS = {
    "chunk_kda": (linear.chunk_kda, kda_inputs, {"autotune": False, "backend": "cudnn"}),
    "recurrent_kda": (
        linear.recurrent_kda,
        kda_inputs,
        {"autotune": False, "batch_invariant": True},
    ),
    "chunk_gdn": (linear.chunk_gdn, delta_rule_inputs, {"autotune": False, "backend": "cudnn"}),
    "recurrent_gdn": (
        linear.recurrent_gdn,
        delta_rule_inputs,
        {"autotune": False, "batch_invariant": True},
    ),
}


@pytest.mark.parametrize(
    "operation,make_inputs,options", REFERENCE_OPTIONS.values(), ids=REFERENCE_OPTIONS
)
def test_reference_validates_and_ignores_kernel_options(operation, make_inputs, options):
    """Kernel options choose how fused kernels run, so one dict serves both implementations."""
    inputs = make_inputs()
    expected, _ = operation(*inputs, impl="reference")
    actual, _ = operation(*inputs, impl="reference", kernel_options=options)
    assert torch.equal(actual, expected)


def test_paged_mla_decode_validates_its_kernel_options():
    tensors = [torch.empty(0)] * 7
    for kernel_options, error in (({"autotune": 1}, TypeError), ({"tune": True}, ValueError)):
        with pytest.raises(error, match="kernel_options|kernel options"):
            paged_mla_decode(
                *tensors,
                max_seqlen_q=1,
                scale=1.0,
                impl="reference",
                kernel_options=kernel_options,
            )


@pytest.mark.parametrize(
    ("module", "name"), TUNED_OPERATIONS, ids=[name for _, name in TUNED_OPERATIONS]
)
def test_autotune_keyword_argument_is_gone(module, name):
    pytest.importorskip("triton")
    pytest.importorskip("cutlass")
    with pytest.raises(TypeError, match="unexpected keyword argument 'autotune'"):
        getattr(module, name)(autotune=False)
