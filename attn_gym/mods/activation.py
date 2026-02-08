"""Non-softmax activation attention via FlexAttention's log-space wrapper trick.

FlexAttention computes O = softmax(score_mod(S)) @ V where S = QK^T / sqrt(d).
To replace softmax with any activation f, define:

    score_mod(s) = log(f(s) + N)

where N is an additive offset that keeps the log argument positive (f(s) + N > 0
for all s). This makes softmax's numerator become f(s) + N, so FlexAttention
returns O' = (f(S) + N)V / ell where ell = sum_j(f(s_j) + N). Using
return_aux=AuxRequest(lse=True) to retrieve log(ell), we can recover:

    ell = exp(lse)
    f(S) @ V = O' * ell - N * sum_j(V_j)
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import AuxRequest, _score_mod_signature
from collections.abc import Callable

import torch.nn.functional as F


def generate_activation_score_mod(
    activation: Callable[[Tensor], Tensor] = F.gelu,
    offset: float = 1.0,
) -> _score_mod_signature:
    """Returns a score_mod that replaces softmax with an arbitrary activation.

    Wraps the activation as log(activation(s) + offset) so that FlexAttention's
    internal softmax reduces to (activation(s) + offset) / ell. Use
    undo_softmax() on the output to recover activation(S) @ V.

    Args:
        activation: Pointwise activation function. Must satisfy
            f(s) + offset > 0 for all s (e.g. gelu, relu, sigmoid).
        offset: Additive constant to keep log argument positive. Larger values
            increase numerical stability but also increase the bias correction.
    """

    def activation_score_mod(score, b, h, q_idx, kv_idx):
        return torch.log(activation(score) + offset)

    return activation_score_mod


def undo_softmax(
    out: Tensor,
    lse: Tensor,
    v_sum: Tensor,
    offset: float = 1.0,
) -> Tensor:
    """Recover activation(S) @ V from FlexAttention's softmax-normalized output.

    Args:
        out: FlexAttention output [B, H, Q_LEN, HEAD_DIM].
        lse: Log-sum-exp from AuxOutput.lse [B, H, Q_LEN].
        v_sum: Bias term sum_j(V_j) for each query's attended keys.
            For full (unmasked) attention: V.sum(dim=-2, keepdim=True).
            For causal attention: V.cumsum(dim=-2).
        offset: Must match the offset used in generate_activation_score_mod.
    """
    ell = lse.exp().unsqueeze(-1)
    return out * ell - offset * v_sum


def main(device: str = "cuda", compile: bool = False):
    """Demonstrate non-softmax attention via FlexAttention with correctness check."""
    import math
    from torch.nn.attention.flex_attention import flex_attention

    flex_fn = torch.compile(flex_attention) if compile else flex_attention

    torch.manual_seed(0)

    B, H, SEQ_LEN, HEAD_DIM = 1, 4, 128, 64
    Q = torch.randn(B, H, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float32)
    K = torch.randn(B, H, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float32)
    V = torch.randn(B, H, SEQ_LEN, HEAD_DIM, device=device, dtype=torch.float32)

    for name, activation in [("GELU", F.gelu), ("ReLU", F.relu)]:
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        naive_out = activation(scores) @ V

        score_mod = generate_activation_score_mod(activation)
        flex_out, aux = flex_fn(
            Q,
            K,
            V,
            score_mod=score_mod,
            return_aux=AuxRequest(lse=True),
        )
        result = undo_softmax(flex_out, aux.lse, V.sum(dim=-2, keepdim=True))

        max_err = (result - naive_out).abs().max().item()
        print(f"{name} attention max error: {max_err:.2e}")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
