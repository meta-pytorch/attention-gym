import torch
from torch.autograd import grad
from torch.nn.attention.flex_attention import flex_attention
import pytest
from functools import partial
from attn_gym.mods import generate_tanh_softcap


def test_tanh_approx():
    torch.compiler.reset()
    softcap_mod = generate_tanh_softcap(30, approx=False)
    softcap_mod_approx = generate_tanh_softcap(30, approx=True)
    make_tensor = partial(
        torch.randn, 1, 1, 128, 64, dtype=torch.float16, device="cuda", requires_grad=True
    )

    query, key, value = make_tensor(), make_tensor(), make_tensor()

    flex_attention_compile = torch.compile(flex_attention)
    out = flex_attention_compile(query, key, value, score_mod=softcap_mod)

    grad_q, grad_k, grad_v = grad(out.sum(), (query, key, value))

    out_approx = flex_attention_compile(query, key, value, score_mod=softcap_mod_approx)
    grad_q_approx, grad_k_approx, grad_v_approx = grad(out_approx.sum(), (query, key, value))

    for tensor_softcap, tensor_softcap_approx in zip(
        [out, grad_q, grad_k, grad_v], [out_approx, grad_q_approx, grad_k_approx, grad_v_approx]
    ):
        torch.testing.assert_close(tensor_softcap, tensor_softcap_approx, atol=5e-4, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
