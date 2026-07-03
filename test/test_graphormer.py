import pytest
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.attention.flex_attention import flex_attention

from attn_gym.mods import (
    generate_graphormer_edge_bias,
    generate_graphormer_spatial_bias,
    shortest_path_distances,
    shortest_path_edge_types,
)


def random_graph(B: int, N: int, edge_prob: float, device: str) -> torch.Tensor:
    adjacency = torch.rand(B, N, N, device=device) < edge_prob
    adjacency = adjacency | adjacency.transpose(-1, -2)
    adjacency.diagonal(dim1=-2, dim2=-1).zero_()
    return adjacency


def path_graph(N: int) -> torch.Tensor:
    adjacency = torch.zeros(1, N, N, dtype=torch.bool)
    idx = torch.arange(N - 1)
    adjacency[0, idx, idx + 1] = True
    adjacency[0, idx + 1, idx] = True
    return adjacency


def dense_spatial_bias(spatial_bias, distances):
    return spatial_bias[:, distances].permute(1, 0, 2, 3)


def dense_edge_bias(edge_bias, path_types, path_lengths):
    """Dense (B, H, N, N) reference for the edge encoding built with plain indexing."""
    B, N = path_lengths.shape[0], path_lengths.shape[1]
    H, K, _ = edge_bias.shape
    total = torch.zeros(B, H, N, N, device=edge_bias.device, dtype=edge_bias.dtype)
    for n in range(K):
        contrib = edge_bias[:, n, path_types[..., n]].permute(1, 0, 2, 3)
        total = total + contrib * (path_lengths > n).unsqueeze(1)
    return total / path_lengths.clamp(min=1).unsqueeze(1)


def test_shortest_path_distances_path_graph():
    N, max_distance = 8, 3
    expected = (torch.arange(N)[:, None] - torch.arange(N)[None, :]).abs()
    torch.testing.assert_close(
        shortest_path_distances(path_graph(N), max_distance)[0],
        expected.clamp_max(max_distance + 1).to(torch.int32),
    )


def test_shortest_path_distances_disconnected():
    adjacency = torch.zeros(1, 4, 4, dtype=torch.bool)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
    dist = shortest_path_distances(adjacency, max_distance=5)[0]
    assert dist[0, 1] == 1
    assert dist[0, 2] == 6
    assert dist[2, 3] == 6


def test_shortest_path_edge_types_path_graph():
    N, K = 6, 3
    adjacency = path_graph(N)
    edge_types = torch.zeros(1, N, N, dtype=torch.int64)
    idx = torch.arange(N - 1)
    edge_types[0, idx, idx + 1] = idx + 1
    edge_types[0, idx + 1, idx] = idx + 1

    path_types, path_lengths = shortest_path_edge_types(adjacency, edge_types, K)

    assert path_lengths[0, 0, 0] == 0
    assert path_lengths[0, 0, 1] == 1
    assert path_lengths[0, 0, N - 1] == K
    torch.testing.assert_close(path_types[0, 0, 3], torch.tensor([1, 2, 3], dtype=torch.int32))
    torch.testing.assert_close(path_types[0, 4, 1], torch.tensor([4, 3, 2], dtype=torch.int32))
    torch.testing.assert_close(path_types[0, 0, 2], torch.tensor([1, 2, 0], dtype=torch.int32))


def test_shortest_path_edge_types_unreachable():
    adjacency = torch.zeros(1, 4, 4, dtype=torch.bool)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = True
    edge_types = torch.ones(1, 4, 4, dtype=torch.int64)
    _, path_lengths = shortest_path_edge_types(adjacency, edge_types, max_path_len=3)
    assert path_lengths[0, 0, 2] == 0
    assert path_lengths[0, 2, 3] == 0


@pytest.mark.parametrize("compile_flex", [False, True], ids=["eager", "compiled"])
def test_graphormer_flex_matches_sdpa_with_learnable_bias_grads(compile_flex):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    B, H, N, D, max_distance = 2, 4, 128, 64, 5
    device = "cuda"

    adjacency = random_graph(B, N, edge_prob=0.03, device=device)
    distances = shortest_path_distances(adjacency, max_distance)
    spatial_bias = torch.randn(H, max_distance + 2, device=device, requires_grad=True)
    q, k, v = (torch.randn(B, H, N, D, device=device, requires_grad=True) for _ in range(3))

    flex = torch.compile(flex_attention) if compile_flex else flex_attention
    out = flex(q, k, v, score_mod=generate_graphormer_spatial_bias(spatial_bias, distances))
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense_spatial_bias(spatial_bias, distances)
    )
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    grad_out = torch.randn_like(out)
    grads = grad(out, (q, k, v, spatial_bias), grad_out)
    ref_grads = grad(ref, (q, k, v, spatial_bias), grad_out)
    for g, rg, name in zip(grads, ref_grads, ("q", "k", "v", "spatial_bias")):
        assert g.abs().sum() > 0, f"grad_{name} is all zeros"
        torch.testing.assert_close(g, rg, rtol=2e-2, atol=2e-2, msg=lambda m: f"grad_{name}: {m}")


@pytest.mark.parametrize("compile_flex", [False, True], ids=["eager", "compiled"])
def test_graphormer_edge_bias_flex_matches_sdpa_with_grads(compile_flex):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    B, H, N, D, num_edge_types, max_path_len = 2, 4, 128, 64, 3, 4
    device = "cuda"

    adjacency = random_graph(B, N, edge_prob=0.03, device=device)
    edge_types = torch.randint(0, num_edge_types, (B, N, N), device=device)
    path_types, path_lengths = shortest_path_edge_types(adjacency, edge_types, max_path_len)
    edge_bias = torch.randn(H, max_path_len, num_edge_types, device=device, requires_grad=True)
    q, k, v = (torch.randn(B, H, N, D, device=device, requires_grad=True) for _ in range(3))

    flex = torch.compile(flex_attention) if compile_flex else flex_attention
    out = flex(
        q, k, v, score_mod=generate_graphormer_edge_bias(edge_bias, path_types, path_lengths)
    )
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=dense_edge_bias(edge_bias, path_types, path_lengths)
    )
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    grad_out = torch.randn_like(out)
    grads = grad(out, (q, k, v, edge_bias), grad_out)
    ref_grads = grad(ref, (q, k, v, edge_bias), grad_out)
    for g, rg, name in zip(grads, ref_grads, ("q", "k", "v", "edge_bias")):
        assert g.abs().sum() > 0, f"grad_{name} is all zeros"
        torch.testing.assert_close(g, rg, rtol=2e-2, atol=2e-2, msg=lambda m: f"grad_{name}: {m}")


def test_graphormer_combined_spatial_and_edge_bias():
    """Full Graphormer attention bias: spatial + edge encodings composed in one score_mod."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    B, H, N, D, max_distance, num_edge_types, max_path_len = 2, 4, 128, 64, 5, 3, 4
    device = "cuda"

    adjacency = random_graph(B, N, edge_prob=0.03, device=device)
    edge_types = torch.randint(0, num_edge_types, (B, N, N), device=device)
    distances = shortest_path_distances(adjacency, max_distance)
    path_types, path_lengths = shortest_path_edge_types(adjacency, edge_types, max_path_len)

    spatial_bias = torch.randn(H, max_distance + 2, device=device, requires_grad=True)
    edge_bias = torch.randn(H, max_path_len, num_edge_types, device=device, requires_grad=True)
    q, k, v = (torch.randn(B, H, N, D, device=device, requires_grad=True) for _ in range(3))

    spatial_mod = generate_graphormer_spatial_bias(spatial_bias, distances)
    edge_mod = generate_graphormer_edge_bias(edge_bias, path_types, path_lengths)

    def graphormer_mod(score, b, h, q_idx, kv_idx):
        return edge_mod(spatial_mod(score, b, h, q_idx, kv_idx), b, h, q_idx, kv_idx)

    out = torch.compile(flex_attention)(q, k, v, score_mod=graphormer_mod)
    dense = dense_spatial_bias(spatial_bias, distances) + dense_edge_bias(
        edge_bias, path_types, path_lengths
    )
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=dense)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)

    grad_out = torch.randn_like(out)
    grads = grad(out, (spatial_bias, edge_bias), grad_out)
    ref_grads = grad(ref, (spatial_bias, edge_bias), grad_out)
    for g, rg, name in zip(grads, ref_grads, ("spatial_bias", "edge_bias")):
        assert g.abs().sum() > 0, f"grad_{name} is all zeros"
        torch.testing.assert_close(g, rg, rtol=2e-2, atol=2e-2, msg=lambda m: f"grad_{name}: {m}")
