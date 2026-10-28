# Utilities

Helper functions for visualization, benchmarking, and profiling.

## Visualization

::: attn_gym.utils.visualize_attention_scores

::: attn_gym.utils.create_score_mod

## Benchmarking

::: attn_gym.utils.benchmark_cuda_function_in_microseconds

::: attn_gym.utils.calculate_tflops

## Profiling

::: attn_gym.utils.cuda_kernel_profiler

## Distributed

Copy-engine transfers into symmetric memory, for groups inside one NVLink domain (one node or
an NVL72 rack). Payload copies do not occupy SMs, so they overlap attention kernels well; they
still share memory and NVLink bandwidth.

::: attn_gym.distributed.CopyEngineAllGather

::: attn_gym.distributed.CopyEngineChannel

::: attn_gym.distributed.same_nvlink_domain

::: attn_gym.distributed.nvlink_domain

### Selective block exchange

Context-parallel attention often needs only some remote K/V blocks, for example under a
sliding window. Each rank lists the blocks of its assembled K/V, the K/V that attention reads.
`BlockExchangePlan.from_assembled_blocks` turns those lists into which blocks move between
ranks. `CopyEngineBlockExchange` runs that plan with copy-engine pushes into symmetric memory.

```python
from attn_gym.distributed.selective_gather import (
    BlockExchangePlan, BlockLocation, CopyEngineBlockExchange,
)

# The same lists on every rank: assembled_blocks[r][i] is where block i of rank r's
# assembled K/V is stored.
assembled_blocks = [[BlockLocation(owner_rank, block_number), ...], ...]
plan = BlockExchangePlan.from_assembled_blocks(
    assembled_blocks, rank=rank, blocks_per_rank=blocks_per_rank, block_tokens=128,
)
exchange = CopyEngineBlockExchange(
    group, plan, k_token_shape=(heads, k_dim), v_token_shape=(heads, v_dim),
    dtype=torch.bfloat16, device=k.device,
)
assembled_k, assembled_v = exchange.exchange_and_assemble(k, v)
```

With `exchange_and_assemble`, attention saves the full assembled K/V for backward. Under
activation checkpointing, call `exchange.exchange` outside the checkpoint and
`exchange.assemble` inside it, so only the received blocks are saved.

::: attn_gym.distributed.selective_gather.BlockLocation

::: attn_gym.distributed.selective_gather.BlockExchangePlan

::: attn_gym.distributed.selective_gather.CopyEngineBlockExchange

## Misc

::: attn_gym.utils.get_flash_block_size
