import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import pytest
from attn_gym.masks import generate_natten, generate_tiled_natten, generate_morton_natten
from attn_gym.masks.natten import morton_decode, morton_encode


def run_natten(
    mask=None,
    encoder=None,
    decoder=None,
    query=None,
    key=None,
    value=None,
    gradOut=None,
    print_mask=True,
):
    B, H, W, _, D = query.shape
    if decoder:
        permuter_x, permuter_y = decoder(torch.arange(W * W))
        q = (
            query[:, :, permuter_x, permuter_y, :]
            .clone()
            .detach()
            .requires_grad_(query.requires_grad)
        )
        k = key[:, :, permuter_x, permuter_y, :].clone().detach().requires_grad_(key.requires_grad)
        v = (
            value[:, :, permuter_x, permuter_y, :]
            .clone()
            .detach()
            .requires_grad_(value.requires_grad)
        )
        dO = gradOut[:, :, permuter_x, permuter_y, :]
    else:
        q = query.flatten(2, 3).clone().detach().requires_grad_(query.requires_grad)
        k = key.flatten(2, 3).clone().detach().requires_grad_(key.requires_grad)
        v = value.flatten(2, 3).clone().detach().requires_grad_(value.requires_grad)
        dO = gradOut.flatten(2, 3)
    block_mask = create_block_mask(mask, 1, 1, W * W, W * W, device=query.device)
    if print_mask:
        print(f"\nBlock Mask:\n{block_mask}")

    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)
    out = flex_attention_compiled(q, k, v, block_mask=block_mask)

    out.backward(dO)

    if encoder:
        i_x = torch.arange(W)[:, None].broadcast_to(W, W).flatten()
        i_y = torch.arange(W)[None, :].broadcast_to(W, W).flatten()
        depermuter = encoder(i_x, i_y)
        out = out[:, :, depermuter, :].reshape(B, H, W, W, D)
        q_grad = q.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        k_grad = k.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        v_grad = v.grad[:, :, depermuter, :].reshape(B, H, W, W, D)
        results = [out, q_grad, k_grad, v_grad]
    else:
        out = out.reshape(B, H, W, W, D)
        q_grad = q.grad.reshape(B, H, W, W, D)
        k_grad = k.grad.reshape(B, H, W, W, D)
        v_grad = v.grad.reshape(B, H, W, W, D)
        results = [out, q_grad, k_grad, v_grad]

    del q, k, v, dO

    return results


def test_natten_masks(
    B=16,
    H=16,
    W=128,
    D=64,
    K_W=13,
    T_W=8,
    print_mask=True,
):
    torch.compiler.reset()
    query = torch.randn(B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn(B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.randn(B, H, W, W, D, device="cuda", dtype=torch.float16, requires_grad=True)
    gradOut = torch.randn(B, H, W, W, D, device="cuda", dtype=torch.float16)

    # Run naive NA
    naive_mask = generate_natten(W, W, K_W, K_W)
    naive_results = run_natten(
        mask=naive_mask, query=query, key=key, value=value, gradOut=gradOut, print_mask=print_mask
    )

    # Run tiled NA
    T_H = T_W

    def tiled_encoder(x, y):
        """
        Map 2-D coordinates to 1-D index for static tiles of T_H x T_W.
        """
        t_x, t_y = x // T_W, y // T_H
        t_id = t_x * (W // T_W) + t_y
        i_x, i_y = x % T_W, y % T_H
        t_offset = i_x * T_W + i_y
        return t_id * (T_H * T_W) + t_offset

    def tiled_decoder(idx):
        """
        Map 1-D index to 2-D coordinates for static tiles of T_H x T_W.
        """
        t_id = idx // (T_H * T_W)
        t_x, t_y = t_id // (W // T_W), t_id % (W // T_W)
        t_offset = idx % (T_H * T_W)
        i_x, i_y = t_offset // T_W, t_offset % T_W
        return t_x * T_W + i_x, t_y * T_H + i_y

    tiled_mask = generate_tiled_natten(W, W, K_W, K_W, T_W, T_H)
    tiled_results = run_natten(
        mask=tiled_mask,
        encoder=tiled_encoder,
        decoder=tiled_decoder,
        query=query,
        key=key,
        value=value,
        gradOut=gradOut,
        print_mask=print_mask,
    )

    # Run morton NA
    morton_mask = generate_morton_natten(W, W, K_W, K_W)
    morton_results = run_natten(
        mask=morton_mask,
        encoder=morton_encode,
        decoder=morton_decode,
        query=query,
        key=key,
        value=value,
        gradOut=gradOut,
        print_mask=print_mask,
    )

    for naive, tiled, morton in zip(naive_results, tiled_results, morton_results):
        torch.testing.assert_close(naive, tiled, atol=1e-1, rtol=1e-2)
        print("Tiled NATTEN: Correctness check passed ✅")
        torch.testing.assert_close(naive, morton, atol=1e-1, rtol=1e-2)
        print("Morton NATTEN: Correctness check passed ✅")

    # Clean up to save memory
    del query, key, value, gradOut, naive_results, tiled_results
    torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__])
