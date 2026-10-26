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
an NVL72 rack). They use no SMs, so they overlap compute without slowing it.

::: attn_gym.distributed.CopyEngineAllGather

::: attn_gym.distributed.CopyEngineChannel

::: attn_gym.distributed.single_node

## Misc

::: attn_gym.utils.get_flash_block_size
