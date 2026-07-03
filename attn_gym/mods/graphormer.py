"""Implementation of the Graphormer attention-bias score mods from the paper
Do Transformers Really Perform Badly for Graph Representation?: https://arxiv.org/abs/2106.05234

Graphormer biases attention scores between node pairs with two learnable terms:
a spatial encoding indexed by shortest-path distance, and an edge encoding that
averages learnable per-position weights over the edge types along the shortest path.
(The paper's third term, centrality encoding, is added to node features rather than
attention scores, so it is out of scope for a score_mod.)
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature


def shortest_path_distances(adjacency: Tensor, max_distance: int) -> Tensor:
    """Computes batched all-pairs shortest-path distances via Floyd-Warshall.

    Distances beyond ``max_distance`` and unreachable pairs are bucketed together at
    ``max_distance + 1``, matching Graphormer's treatment of far/disconnected nodes.

    Args:
        adjacency: (B, N, N) boolean or 0/1 adjacency matrices (unweighted edges).
        max_distance: largest distance that gets its own bias bucket.

    Returns:
        (B, N, N) int32 distances with values in [0, max_distance + 1]. int32 is the
        smallest integer dtype flex_attention can gather with, and the matrix is the
        only O(N^2) state Graphormer needs: compute it once per graph in preprocessing
        and share it across every layer and head.
    """
    _, N, _ = adjacency.shape
    unreachable = max(N, max_distance + 1)
    dist = torch.where(adjacency.bool(), 1, unreachable)
    dist.diagonal(dim1=-2, dim2=-1).zero_()
    for k in range(N):
        dist = torch.minimum(dist, dist[:, :, k].unsqueeze(-1) + dist[:, k].unsqueeze(1))
    return dist.clamp_max(max_distance + 1).to(torch.int32)


def generate_graphormer_spatial_bias(
    spatial_bias: Tensor, distances: Tensor
) -> _score_mod_signature:
    """Returns a Graphormer spatial-encoding score_mod.

    Args:
        spatial_bias: (H, num_buckets) bias table, one learnable scalar per head and
            shortest-path distance. Pass an ``nn.Parameter`` (or a tensor with
            ``requires_grad=True``) to train it. ``num_buckets`` must cover every value
            in ``distances``: ``max_distance + 2`` when produced by
            ``shortest_path_distances``.
        distances: (B, N, N) integer shortest-path distances between node pairs.

    Returns:
        graphormer_spatial_bias: Graphormer spatial bias score_mod
    """

    def graphormer_spatial_bias(score, b, h, q_idx, kv_idx):
        return score + spatial_bias[h, distances[b, q_idx, kv_idx]]

    return graphormer_spatial_bias


def shortest_path_edge_types(
    adjacency: Tensor, edge_types: Tensor, max_path_len: int
) -> tuple[Tensor, Tensor]:
    """Reconstructs the edge types along each pair's shortest path for edge encoding.

    Runs Floyd-Warshall with next-hop tracking, then walks each (i, j) path for up to
    ``max_path_len`` hops, recording the type of every traversed edge. Like the paper's
    ``multi_hop_max_dist``, longer paths are truncated to their first ``max_path_len``
    edges. Precompute this once per graph alongside ``shortest_path_distances``.

    Args:
        adjacency: (B, N, N) boolean or 0/1 adjacency matrices (unweighted edges).
        edge_types: (B, N, N) integer edge-type ids, read only where an edge exists.
        max_path_len: number of leading path positions to record (K).

    Returns:
        path_edge_types: (B, N, N, K) int32 edge types; positions past the path end are
            0 and must be masked with ``path_lengths`` (as the returned score_mod does).
        path_lengths: (B, N, N) int32 count of valid positions per pair:
            min(distance, K), with 0 for unreachable pairs and the diagonal.
    """
    B, N, _ = adjacency.shape
    device = adjacency.device
    unreachable = N + 1
    dist = torch.where(adjacency.bool(), 1, unreachable)
    dist.diagonal(dim1=-2, dim2=-1).zero_()
    nxt = torch.where(adjacency.bool(), torch.arange(N, device=device).expand(B, N, N), -1)
    for k in range(N):
        alt = dist[:, :, k].unsqueeze(-1) + dist[:, k].unsqueeze(1)
        improved = alt < dist
        dist = torch.where(improved, alt, dist)
        nxt = torch.where(improved, nxt[:, :, k].unsqueeze(-1), nxt)

    path_lengths = torch.where(dist > N, 0, dist).clamp_max(max_path_len)
    batch = torch.arange(B, device=device).view(B, 1, 1)
    dest = torch.arange(N, device=device).view(1, 1, N)
    cur = torch.arange(N, device=device).view(1, N, 1).expand(B, N, N)
    path_types = torch.zeros(B, N, N, max_path_len, dtype=torch.int32, device=device)
    for n in range(max_path_len):
        valid = path_lengths > n
        step = nxt[batch, cur, dest]
        path_types[..., n] = torch.where(valid, edge_types[batch, cur, step], 0)
        cur = torch.where(valid, step, cur)
    return path_types, path_lengths.to(torch.int32)


def generate_graphormer_edge_bias(
    edge_bias: Tensor, path_edge_types: Tensor, path_lengths: Tensor
) -> _score_mod_signature:
    """Returns a Graphormer edge-encoding score_mod.

    Implements the paper's c_ij = mean_n(x_{e_n} . w_n^E) for categorical edge types,
    where the feature/weight dot product collapses into one learnable scalar per
    (head, path position, edge type). The sum over path positions is unrolled at trace
    time, so keep ``max_path_len`` small (the paper truncates multi-hop paths too).

    The table is sliced per path position because compiled flex_attention only supports
    one gather per gradient-requiring captured tensor; gradients flow from each slice
    back to ``edge_bias`` through regular autograd. Because those slices join the
    autograd graph when this generator runs, call it once per forward pass when
    training (it is cheap).

    Args:
        edge_bias: (H, K, num_edge_types) learnable bias table. Pass an
            ``nn.Parameter`` to train it.
        path_edge_types: (B, N, N, K) edge types from ``shortest_path_edge_types``.
        path_lengths: (B, N, N) valid position counts from ``shortest_path_edge_types``.

    Returns:
        graphormer_edge_bias: Graphormer edge encoding score_mod
    """
    max_path_len = path_edge_types.shape[-1]
    position_tables = [edge_bias[:, n] for n in range(max_path_len)]

    def graphormer_edge_bias(score, b, h, q_idx, kv_idx):
        length = path_lengths[b, q_idx, kv_idx]
        total = 0.0
        for n, table in enumerate(position_tables):
            contrib = table[h, path_edge_types[b, q_idx, kv_idx, n]]
            total = total + torch.where(length > n, contrib, 0.0)
        return score + total / torch.clamp(length, min=1)

    return graphormer_edge_bias


SMILEY_GRAPH = [
    "................",
    "................",
    "....##....##....",
    "....##....##....",
    "................",
    "................",
    "................",
    "................",
    "..#..........#..",
    "...#........#...",
    "....##....##....",
    "......####......",
    "................",
    "................",
    "................",
    "................",
]


def main(device: str = "cpu", graph: str = "ring"):
    """Visualize the attention bias of the Graphormer spatial encoding.

    Args:
        device (str): Device to use for computation.
        graph (str): "ring", or "smiley" to hide a friendly graph in the adjacency
            matrix — edges (distance 1) draw the face, and the fainter distance-2/3
            glow around it is the shortest-path computation at work.
    """
    from attn_gym import visualize_attention_scores

    B, H, HEAD_DIM = 1, 1, 8

    if graph == "smiley":
        num_nodes, max_distance = len(SMILEY_GRAPH), 3
        adjacency = torch.tensor(
            [[c == "#" for c in row] for row in SMILEY_GRAPH], device=device
        ).unsqueeze(0)
        spatial_bias = torch.tensor([[0.0, 6.0, 2.0, 1.0, 0.0]], device=device)
    else:
        num_nodes, max_distance = 12, 4
        nodes = torch.arange(num_nodes, device=device)
        adjacency = torch.zeros(B, num_nodes, num_nodes, dtype=torch.bool, device=device)
        adjacency[:, nodes, (nodes + 1) % num_nodes] = True
        adjacency[:, (nodes + 1) % num_nodes, nodes] = True
        spatial_bias = -torch.arange(max_distance + 2, dtype=torch.float32, device=device).repeat(
            H, 1
        )

    distances = shortest_path_distances(adjacency, max_distance)

    def make_tensor():
        return torch.ones(B, H, num_nodes, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(
        query,
        key,
        score_mod=generate_graphormer_spatial_bias(spatial_bias, distances),
        device=device,
        name=f"graphormer_score_mod_{graph}",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
