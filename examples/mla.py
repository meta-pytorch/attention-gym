import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import flex_attention

from attn_gym.mods.latent_attention import generate_mla_rope_score_mod

torch._inductor.config.unroll_reductions_threshold = 65

# H100 config
kernel_options = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "num_stages": 2,
    "FORCE_USE_FLEX_ATTENTION": True,  # TODO inspect flex_decode
}


def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DeepseekV2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class DeepseekV2AttentionVanilla(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_size = 5120
        self.num_heads = 128

        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.q_head_dim = 192  # 192 = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.rope_theta = 10000

        # W^DQ ~ [5120, 1536]
        self.q_a_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
        )
        torch.nn.init.normal_(self.q_a_proj.weight)

        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        # W^UQ & W^QR = [1536, 128*(128+64)]
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        torch.nn.init.normal_(self.q_b_proj.weight)

        # We don't need these modules, since we already have cached k_pe and compressed_kv tensor.
        # self.kv_a_proj_with_mqa = nn.Linear( # [,5120]-->[, 512+64] W^DKV & W^KR = [5120, 512+64]
        #     self.hidden_size,
        #     self.kv_lora_rank + self.qk_rope_head_dim,
        #     bias=False,
        # )
        # self.kv_a_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)

        # W^UK & W^UV ~ [512, 128*(128+128)]
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )
        torch.nn.init.normal_(self.kv_b_proj.weight)

        # W^O ~ [128*128, 5120]
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
        )
        torch.nn.init.normal_(self.o_proj.weight)

        self.softmax_scale = self.q_head_dim ** (-0.5)

    def run_decode(
        self,
        hidden_states: torch.Tensor,
        compressed_kv_normed_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = hidden_states.size()
        if q_len != 1:
            raise ValueError(f"Only support decode, but got hidden_states[{hidden_states.size()}]")

        ckv_bsz, kv_len, ckv_dim = compressed_kv_normed_cache.size()
        if ckv_bsz != bsz or ckv_dim != self.kv_lora_rank:
            raise ValueError(
                f"Unexpected shape: compressed_kv_normed_cache[{compressed_kv_normed_cache.size()}]"
            )

        kpe_bsz, kpe_len, kpe_dim = k_pe_cache.size()
        if kpe_bsz != bsz or kpe_dim != self.qk_rope_head_dim or kv_len != kpe_len:
            raise ValueError(f"Unexpected shape: k_pe_cache[{k_pe_cache.size()}]")

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        # q_nope ~ [bsz, q_len, 128]   q_pe ~ [bsz, q_len, 64]
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        k_pe = k_pe_cache.view(bsz, kv_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(compressed_kv_normed_cache)
            .view(bsz, kv_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        # k_nope ~ [bsz, num_heads, kv_len, 128]  value_states ~ [bsz, num_heads, kv_len, 128]
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        if k_nope.size() != (bsz, self.num_heads, kv_len, self.qk_nope_head_dim):
            raise ValueError(f"k_nope[{k_nope.size()}]")
        if value_states.size() != (bsz, self.num_heads, kv_len, self.v_head_dim):
            raise ValueError(f"value_states[{value_states.size()}]")

        freqs_cis = precompute_freqs_cis(
            self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
        ).to(q_pe.device)
        q_pe, k_pe = apply_rotary_emb(
            q_pe.transpose(1, 2).repeat(1, kv_len, 1, 1),
            k_pe.transpose(1, 2),
            freqs_cis,
        )
        q_pe = q_pe[:, -1:, :, :].transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)

        # Concat q_nope and q_pe to produce a new Q tensor with head_dim = 192
        query_states = q.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        # Concat k_nope and k_pe to produce a new K tensor with head_dim = 192
        key_states = k_pe.new_empty(bsz, self.num_heads, kv_len, self.q_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, self.num_heads * self.v_head_dim
        )

        output = self.o_proj(attn_output)

        return output


class DeepseekV2AttentionMatAbsorbDecode(nn.Module):
    def __init__(self, mla_vanilla: DeepseekV2AttentionVanilla):
        super().__init__()

        self.hidden_size = mla_vanilla.hidden_size  # 5120
        self.num_heads = mla_vanilla.num_heads  # 128

        self.q_lora_rank = mla_vanilla.q_lora_rank  # 1536
        self.qk_rope_head_dim = mla_vanilla.qk_rope_head_dim  # 64
        self.kv_lora_rank = mla_vanilla.kv_lora_rank  # 512
        self.v_head_dim = mla_vanilla.v_head_dim  # 128
        self.qk_nope_head_dim = mla_vanilla.qk_nope_head_dim  # 128
        self.q_head_dim = (
            mla_vanilla.q_head_dim
        )  # qk_nope_head_dim + qk_rope_head_dim # 128+64=192

        self.softmax_scale = mla_vanilla.softmax_scale

        self.rope_theta = mla_vanilla.rope_theta
        # self.rotary_emb =  mla_vanilla.rotary_emb

        # W^DQ ~ [5120, 1536]
        self.W_DQ = mla_vanilla.q_a_proj.weight.transpose(0, 1)

        self.q_a_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)

        # W_UQ ~ [1536, 128, 128]
        W_UQ, W_QR = torch.split(
            mla_vanilla.q_b_proj.weight.t().view(
                self.q_lora_rank, self.num_heads, self.q_head_dim
            ),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            -1,
        )
        # W_UQ ~ [1536, 128*64]
        self.W_QR = W_QR.reshape(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim)

        # W_UK ~ [512, 128, 128]   W_UV ~ [512, 128, 128]
        W_UK, W_UV = torch.split(
            mla_vanilla.kv_b_proj.weight.t().view(
                self.kv_lora_rank,
                self.num_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ),
            [self.qk_nope_head_dim, self.v_head_dim],
            -1,
        )

        # Now we merge W_UQ and W_UK (absorb W_UK into W_UQ)
        # q~q_lora_rank  n~num_heads  d~qk_nope_head_dim  l~kv_lora_rank
        self.W_UQ_UK = torch.einsum("q n d, l n d -> q n l", W_UQ, W_UK).flatten(
            start_dim=1
        )  # [1536, 65536]

        W_O = mla_vanilla.o_proj.weight.view(self.hidden_size, self.num_heads, self.v_head_dim)

        # Merge W_UV and W_O (absorb W_UV into W_O)
        # l~kv_lora_rank  n~num_heads  d~v_head_dim  h~hidden_size
        self.W_UV_O = torch.einsum("l n d, h n d -> n l h", W_UV, W_O).flatten(
            start_dim=0, end_dim=1
        )  # [65536, 5120]

    def run_proof_of_concept(
        self,
        hidden_states: torch.Tensor,
        compressed_kv_normed_cache: torch.Tensor,
        k_pe_cache: torch.Tensor,
        use_flex: bool,
        compile: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, _ = hidden_states.size()

        c_Q = torch.matmul(hidden_states, self.W_DQ)
        # c_Q ~ [bsz, q_lora_rank:1536]
        c_Q = self.q_a_layernorm(c_Q)

        q_pe = torch.matmul(
            c_Q,
            self.W_QR,  # c_Q ~ [bsz, q_lora_rank~1536]
        )  # W_QR ~ [1536, 128*64]

        # q_pe ~ [bsz, seq_len(1),  128, 64]
        # There are 128 heads, each head has 64 dimensions that will be rotated.
        q_pe = q_pe.reshape(bsz, self.num_heads, self.qk_rope_head_dim)

        # q_nope these are the folded query inputs that have no rope_applied
        q_nope = torch.matmul(c_Q, self.W_UQ_UK)  # W_UQ_UK~[1536, 128*512]

        # q_nope ~ [bsz, seq_len(1), 128, 512 (DC)]
        q_nope = q_nope.reshape(bsz, self.num_heads, self.kv_lora_rank)

        kv_len = compressed_kv_normed_cache.size(1)

        if not use_flex:
            # For now lets compute but normally we would fuse in rope
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.device)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(1).repeat(1, kv_len, 1, 1),
                k_pe_cache.unsqueeze(2),
                freqs_cis,
            )
            q_pe = q_pe[:, -1:, :, :].squeeze(1)
            k_pe_cache = k_pe_cache.squeeze(2)

            # attn_weights_pe ~ [bsz, 128, kv_len]
            attn_weights_pe = torch.matmul(
                q_pe,  # [bsz, num_heads, qk_rope_head_dim]
                k_pe_cache.transpose(
                    1, 2
                ),  # [bsz, kv_len, 64] view(bsz, kv_len, self.qk_rope_head_dim)
            )
            # attn_weights_nope ~ [bsz, 128, kv_len]
            attn_weights_nope = torch.matmul(
                q_nope,  # [bsz, 128, 512]
                compressed_kv_normed_cache.transpose(1, 2),  # view(bsz, kv_len, 512)
            )

            attn_weights = (attn_weights_pe + attn_weights_nope) * self.softmax_scale

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                q_nope.dtype
            )

            # attn_output ~ {attn_output.shape}") # [bsz, 128, 512]
            attn_output = torch.matmul(
                attn_weights,  # [bsz, 128, kv_len]
                compressed_kv_normed_cache,  # [bsz, kv_len, 512]
            )

        else:
            # q_pe ~ [bsz, num_q_heads(128), qk_rope_head_dim(64)]
            # q_nope ~ [bsz, num_q_heads(128), kv_lora_rank(512)]
            freqs_cis = precompute_freqs_cis(
                self.qk_rope_head_dim, kv_len, self.rope_theta, use_scaled=False
            ).to(k_pe_cache.device)
            q_pe, k_pe_cache = apply_rotary_emb(
                q_pe.unsqueeze(1).repeat(1, kv_len, 1, 1),
                k_pe_cache.unsqueeze(2),
                freqs_cis,
            )
            # We should fuse in rope but leave like this for now
            q_pe = q_pe[:, -1:, :, :].transpose(1, 2)
            k_pe_cache = k_pe_cache.transpose(1, 2)
            sm = generate_mla_rope_score_mod(
                q_pe, k_pe_cache, num_heads=128, scale=self.softmax_scale
            )
            q_nope = q_nope.unsqueeze(-2)

            unsqueezed_cache = compressed_kv_normed_cache.unsqueeze(1)
            if compile:
                flex = torch.compile(
                    flex_attention, fullgraph=True, mode="max-autotune-no-cudagraphs"
                )
            else:
                flex = flex_attention
            attn_output = flex(
                q_nope,
                unsqueezed_cache,
                unsqueezed_cache,
                score_mod=sm,
                enable_gqa=True,
                scale=self.softmax_scale,
                kernel_options=kernel_options,
            )

        # output ~ [bsz, 5120]
        output = torch.matmul(
            attn_output.reshape(bsz, self.num_heads * self.kv_lora_rank), self.W_UV_O
        )  # W_UV_O ~ [65536, 5120]

        return output


def run_accuracy_test(
    dev_id: int,
    compile: bool,
    mla_vanilla: DeepseekV2AttentionVanilla,
    hidden_states: torch.Tensor,
    compressed_kv_normed_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
) -> None:
    """
    Run accuracy comparison tests between vanilla and absorbed attention implementations.

    Args:
        dev_id: CUDA device ID
        batch_size: Batch size for testing
        kv_length: KV length for testing
        page_size: Page size parameter
        seed: Random seed for reproducibility
        compile: Whether to compile the FlexAttention call
        mla_vanilla: Instance of vanilla attention model
        hidden_states: Input hidden states tensor
        compressed_kv_normed_cache: Compressed KV cache tensor
        k_pe_cache: Key position encoding cache tensor
    """
    output_vanilla = mla_vanilla.run_decode(hidden_states, compressed_kv_normed_cache, k_pe_cache)

    mla_mat_absorb = DeepseekV2AttentionMatAbsorbDecode(mla_vanilla).cuda(device=dev_id)

    output_mat_absorbed_use_torch = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flex=False,
        compile=compile,
    )

    output_mat_absorbed_use_flex = mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flex=True,
        compile=compile,
    )

    cos_sim_use_torch = F.cosine_similarity(
        output_vanilla.reshape(-1), output_mat_absorbed_use_torch.reshape(-1), dim=0
    )
    print(f"cos_sim_use_torch={cos_sim_use_torch}")
    assert cos_sim_use_torch > 0.99

    cos_sim_use_flex = F.cosine_similarity(
        output_vanilla.reshape(-1),
        output_mat_absorbed_use_flex.reshape(-1),
        dim=0,
    )
    print(f"cos_sim_use_flex={cos_sim_use_flex}\n")
    assert cos_sim_use_flex > 0.99


def run_performance_test(
    dev_id: int,
    compile: bool,
    mla_vanilla: DeepseekV2AttentionVanilla,
    hidden_states: torch.Tensor,
    compressed_kv_normed_cache: torch.Tensor,
    k_pe_cache: torch.Tensor,
):
    from pathlib import Path

    from transformer_nuggets.utils.benchmark import (
        benchmark_cuda_function_in_microseconds,
        profiler,
    )

    mla_mat_absorb = DeepseekV2AttentionMatAbsorbDecode(mla_vanilla).cuda(device=dev_id)

    no_ops_eager = lambda: mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flex=False,
        compile=compile,
    )
    no_ops_compile = lambda: mla_mat_absorb.run_proof_of_concept(
        hidden_states.squeeze(1),
        compressed_kv_normed_cache,
        k_pe_cache,
        use_flex=True,
        compile=compile,
    )

    base_time = benchmark_cuda_function_in_microseconds(no_ops_eager)
    compile_time = benchmark_cuda_function_in_microseconds(no_ops_compile)

    print(f"base_time={base_time} us")
    print(f"compile_time={compile_time} us")

    with profiler(Path("eager")):
        no_ops_eager()

    with profiler(Path("compile")):
        no_ops_compile()

    print("Done")


def main(
    dev_id: int = 0,
    batch_size: int = 6,
    kv_length: int = 640,
    page_size: int = 16,
    seed: int = 666,
    compile: bool = False,
    mode: str = "acc",
) -> None:
    """
    Main function to run attention model comparisons.

    Args:
        dev_id: CUDA device ID
        batch_size: Batch size for testing
        kv_length: KV length for testing
        page_size: Page size parameter
        seed: Random seed for reproducibility
        compile: Whether to compile the FlexAttention call
        mode: Test mode - "acc" for accuracy or "perf" for performance testing
    """
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    mla_vanilla = DeepseekV2AttentionVanilla().cuda(device=dev_id)

    hidden_states = torch.randn([batch_size, 1, mla_vanilla.hidden_size]).to(dev_id)
    compressed_kv_normed_cache = torch.randn([batch_size, kv_length, mla_vanilla.kv_lora_rank]).to(
        dev_id
    )
    k_pe_cache = torch.randn([batch_size, kv_length, mla_vanilla.qk_rope_head_dim]).to(dev_id)

    if mode == "acc":
        run_accuracy_test(
            dev_id,
            compile,
            mla_vanilla,
            hidden_states,
            compressed_kv_normed_cache,
            k_pe_cache,
        )
    elif mode == "perf":
        run_performance_test(
            dev_id,
            compile,
            mla_vanilla,
            hidden_states,
            compressed_kv_normed_cache,
            k_pe_cache,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be either 'acc' or 'perf'")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
