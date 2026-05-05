import pytest
import torch
import torch.nn.functional as F

from attn_gym.mods.exclusive_sa import XSAMultiheadAttention, exclusive_output_mod


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def standard_input(device):
    torch.manual_seed(42)
    B, H, T, d = 2, 4, 32, 16
    Y = torch.randn(B, H, T, d, device=device)
    V = torch.randn(B, H, T, d, device=device)
    return Y, V


class TestExclusiveOutputMod:
    def test_output_shape_preserved(self, standard_input):
        Y, V = standard_input
        Z = exclusive_output_mod(Y, V)
        assert Z.shape == Y.shape

    def test_orthogonality_guarantee(self, standard_input):
        Y, V = standard_input
        Z = exclusive_output_mod(Y, V)
        cos = F.cosine_similarity(Z, V, dim=-1).abs().max().item()
        assert cos < 1e-5

    def test_orthogonality_with_zero_v(self, device):
        Y = torch.randn(1, 1, 4, 8, device=device)
        V = torch.zeros(1, 1, 4, 8, device=device)
        Z = exclusive_output_mod(Y, V)
        assert torch.allclose(Z, Y, atol=1e-6)

    def test_gradient_flows(self, standard_input):
        Y, V = standard_input
        Y = Y.requires_grad_(True)
        V = V.requires_grad_(True)

        loss = exclusive_output_mod(Y, V).sum()
        loss.backward()

        assert Y.grad is not None
        assert V.grad is not None
        assert not torch.isnan(Y.grad).any()
        assert not torch.isnan(V.grad).any()

    def test_idempotent(self, standard_input):
        Y, V = standard_input
        Z1 = exclusive_output_mod(Y, V)
        Z2 = exclusive_output_mod(Z1, V)
        assert torch.allclose(Z1, Z2, atol=1e-5)

    @pytest.mark.parametrize(
        "shape",
        [
            (1, 1, 1, 64),
            (4, 1, 16, 64),
            (2, 8, 64, 256),
            (1, 16, 512, 32),
        ],
    )
    def test_various_shapes(self, shape, device):
        torch.manual_seed(0)
        Y = torch.randn(*shape, device=device)
        V = torch.randn(*shape, device=device)
        Z = exclusive_output_mod(Y, V)

        cos = F.cosine_similarity(Z, V, dim=-1).abs().max().item()
        assert Z.shape == Y.shape
        assert cos < 1e-5


class TestXSAMultiheadAttention:
    @pytest.fixture
    def module(self, device):
        return XSAMultiheadAttention(d_model=64, num_heads=4).to(device).eval()

    @pytest.fixture
    def x(self, device):
        torch.manual_seed(0)
        return torch.randn(2, 32, 64, device=device)

    def test_output_shape(self, module, x):
        out = module(x)
        assert out.shape == x.shape

    def test_no_nan_in_output(self, module, x):
        out = module(x)
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_module(self, device):
        module = XSAMultiheadAttention(d_model=32, num_heads=4).to(device).train()
        x = torch.randn(2, 16, 32, device=device, requires_grad=True)

        loss = module(x).sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_output_orthogonal_to_self_value(self, module, x):
        with torch.no_grad():
            Q = module._split_heads(module.W_q(x))
            K = module._split_heads(module.W_k(x))
            V = module._split_heads(module.W_v(x))
            Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            Z = exclusive_output_mod(Y, V)

        cos = F.cosine_similarity(Z, V, dim=-1).abs().max()
        assert cos.item() < 1e-5

    @pytest.mark.parametrize(
        "d_model,num_heads",
        [
            (32, 1),
            (64, 4),
            (128, 8),
            (256, 16),
        ],
    )
    def test_various_configurations(self, device, d_model, num_heads):
        module = XSAMultiheadAttention(d_model=d_model, num_heads=num_heads).to(device).eval()
        x = torch.randn(2, 16, d_model, device=device)
        out = module(x)
        assert out.shape == (2, 16, d_model)

    def test_invalid_d_model_raises(self, device):
        with pytest.raises(ValueError, match="divisible"):
            XSAMultiheadAttention(d_model=65, num_heads=4).to(device)

    def test_train_eval_dropout_difference(self, device):
        module = XSAMultiheadAttention(d_model=64, num_heads=4, dropout=0.5).to(device)
        x = torch.randn(2, 32, 64, device=device)

        module.train()
        torch.manual_seed(0)
        out_train = module(x)

        module.eval()
        torch.manual_seed(0)
        out_eval = module(x)

        module.eval()
        torch.manual_seed(999)
        out_eval2 = module(x)

        assert not torch.allclose(out_train, out_eval)
        assert torch.allclose(out_eval, out_eval2)
