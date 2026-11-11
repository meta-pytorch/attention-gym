"""Operations share option validation; ``autotune`` belongs in ``kernel_options``."""

from typing import Literal, get_args, get_origin, get_type_hints

import pytest
import torch

from attn_gym import linear, mla, sparse
from attn_gym.linear import gdn, kda
from attn_gym.linear import types as linear_types
from attn_gym.mla import api as mla_api
from attn_gym.mla import paged_mla_decode
from attn_gym.sparse.gather_attn import api as gather_attn_api
from attn_gym.sparse.indexer import api as indexer_api
from attn_gym.types import TUNING_OPTION_SPEC, TuningOptions, validate_kernel_options


def short_conv_specs(decode: bool) -> tuple[dict, type]:
    cute = pytest.importorskip("attn_gym.linear.short_conv.cute")
    if decode:
        return cute.DECODE_KERNEL_OPTION_SPEC, cute.ShortConvDecodeKernelOptions
    return cute.KERNEL_OPTION_SPEC, cute.ShortConvKernelOptions


# Each specification with the TypedDict that documents it.
OPTION_SPECS = {
    "tuning": lambda: (TUNING_OPTION_SPEC, TuningOptions),
    "chunk_gdn": lambda: (linear_types.SPLIT_OPTION_SPEC, linear_types.SplitOptions),
    "chunk_kda": lambda: (linear_types.KERNEL_OPTION_SPEC, linear_types.KernelOptions),
    "recurrent": lambda: (
        linear_types.RECURRENT_OPTION_SPEC,
        linear_types.RecurrentKernelOptions,
    ),
    "gather_attn": lambda: (
        gather_attn_api.KERNEL_OPTION_SPEC,
        gather_attn_api.GatherAttnKernelOptions,
    ),
    "lightning_indexer": lambda: (
        indexer_api.KERNEL_OPTION_SPEC,
        indexer_api.IndexerKernelOptions,
    ),
    "paged_mla_decode": lambda: (mla_api.KERNEL_OPTION_SPEC, TuningOptions),
    "causal_conv1d": lambda: short_conv_specs(decode=False),
    "causal_conv1d_decode": lambda: short_conv_specs(decode=True),
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


@pytest.mark.parametrize("make_spec", OPTION_SPECS.values(), ids=OPTION_SPECS)
def test_kernel_options_follow_one_contract(make_spec):
    spec, typed_dict = make_spec()

    def resolve(options):
        return validate_kernel_options(options, spec, op_name="some_op")

    defaults = {name: default for name, (_, default) in spec.items()}
    assert resolve(None) == resolve({}) == defaults
    with pytest.raises(ValueError, match="unsupported some_op kernel options: alpha, zeta$"):
        resolve({"zeta": True, "alpha": True})
    for name, (accepted, _) in spec.items():
        if accepted is bool:
            valid, wrong, error, expected = (True, False), (0, 1, "False", None), TypeError, "bool"
        elif isinstance(accepted, tuple):
            valid, wrong, error, expected = accepted, ("other", None), ValueError, accepted[-1]
        else:  # ShortConvConfig is the only class-typed option.
            valid, wrong, error = (accepted(64, 2, 8),), (None, (64, 2, 8)), TypeError
            expected = accepted.__name__
        for value in valid:
            assert resolve({name: value}) == {**defaults, name: value}
        for value in wrong:
            with pytest.raises(error, match=rf"kernel_options\['{name}'\] must be .*{expected}"):
                resolve({name: value})

    documented = {
        name: get_args(hint) if get_origin(hint) is Literal else hint
        for name, hint in get_type_hints(typed_dict).items()
    }
    assert documented == {name: accepted for name, (accepted, _) in spec.items()}


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
