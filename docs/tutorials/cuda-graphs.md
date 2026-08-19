# So you wanna CUDA Graph

GPUs continue to get faster and their hunger for kernel launches driven by the CPU is seemingly insatiable. A natural solution to this problem is `cuda graphs`. The post will walk through some common footguns both in kernel design and user invocation for ragged kernels, how to alleviate them and where there is still room to
improve today - especially at the framework level.

## TLDR:
This tutorial will describe some quarks that arise when using cuda-graphs with ragged attention kernels. It proposes some solutions and fun patterns for dealing with the slowdowns that can arise, while ultimately suggesting codesigning a solution based off of your training dataset.

### What are ragged kernels?

<div class="with-sidenote" markdown="1">
<div markdown="1">

Most sequence data comes from an underlying distribution and does not have one exact length - unless the data is boring. A batch of documents, conversations, or videos therefore has a different number of tokens per sample. "But Tensors are regular! What am I to do" You say, the normal solution truncates or pads every sample to a fixed length.

A <span class="sidenote-ref" tabindex="0" aria-describedby="ragged-terminology-note">ragged</span> representation instead packs only the real tokens into one contiguous buffer:

<div class="ragged-viz" role="img" aria-label="A padded PyTorch tensor with shape batch three by maximum sequence length four by feature dimension contains nine real token vectors and three padding slots. Packing removes the padding to produce a nine by feature-dimension tensor. Its sequence metadata is cumulative sequence lengths zero, four, six, and nine with shape batch plus one, or four.">
<div class="ragged-viz__stage">
<div class="ragged-viz__meta">
<span>before · x_padded</span>
<code>[B, max_T, D] = [3, 4, D]</code>
</div>
<div class="ragged-viz__padded">
<span class="ragged-viz__cell ragged-viz__a">A</span>
<span class="ragged-viz__cell ragged-viz__a">A</span>
<span class="ragged-viz__cell ragged-viz__a">A</span>
<span class="ragged-viz__cell ragged-viz__a">A</span>
<span class="ragged-viz__cell ragged-viz__b">B</span>
<span class="ragged-viz__cell ragged-viz__b">B</span>
<span class="ragged-viz__cell ragged-viz__pad">∅</span>
<span class="ragged-viz__cell ragged-viz__pad">∅</span>
<span class="ragged-viz__cell ragged-viz__c">C</span>
<span class="ragged-viz__cell ragged-viz__c">C</span>
<span class="ragged-viz__cell ragged-viz__c">C</span>
<span class="ragged-viz__cell ragged-viz__pad">∅</span>
</div>
</div>
<div class="ragged-viz__action"><strong>9 / 12 slots are real</strong> · remove padding and pack <span aria-hidden="true">↓</span></div>
<div class="ragged-viz__stage">
<div class="ragged-viz__meta">
<span>after · x_packed</span>
<code>[L, D] = [9, D]</code>
<span>metadata · cu_seqlens</span>
<code>[len(docs) + 1] = [B + 1] = [4]</code>
</div>
<div class="ragged-viz__packed-wrap">
<div class="ragged-viz__packed">
<span class="ragged-viz__segment ragged-viz__a" style="grid-column: 1 / 5" data-end="4">A A A A</span>
<span class="ragged-viz__segment ragged-viz__b" style="grid-column: 5 / 7" data-end="6">B B</span>
<span class="ragged-viz__segment ragged-viz__c ragged-viz__segment--last" style="grid-column: 7 / 10" data-end="9">C C C</span>
<span class="ragged-viz__start">0</span>
</div>
<code class="ragged-viz__tensor">cu_seqlens = tensor([0, 4, 6, 9], dtype=torch.int32)</code>
</div>
</div>
</div>

The metadata preserves the logical boundaries: sequence `i` occupies `tokens[cu_seqlens[i]:cu_seqlens[i + 1]]`. Operations like attention use this information to prevent tokens from interacting when they don't belong to the same document. Token-parallel operations like projections do not care about these boundaries, and operate directly on the packed `[L, D]` tensor. Everybody wins!

</div>

<aside id="ragged-terminology-note" class="sidenote" markdown="1">

There are a million different names around this idea: ragged, varlen, packed, jagged, nested tensors, etc. I will use ragged and varlen interchangeably in this doc.

</aside>
</div>

### What is a CUDA Graph

{% call sidenote(
    "PyTorch usage and definition",
    "see the [og doc](https://docs.pytorch.org/docs/2.13/notes/cuda.html#cuda-graph-semantics) for a better and more complete description than this",
) %}
Its a graph. Duh, but really - the common PyTorch usage and definition is to `record` all gpu work issued to a capture stream and its dependent streams, then `replay` that captured dependency graph later. Why would you want to do this? Because if you know exactly what you want to launch you dont need to redo the Python and dispatcher work that invoked those kernels 1 by 1. Instead you can launch the full `Graph` in one go and remove a majority if not all the CPU overhead.
{% endcall %}

#### Hello World

Lets setup a small test program, since this is attn_gym we will focus on a simple proxy attention module. However, the learnings will apply to both varlen attention and the new linear-attention variants we are adding to the gym. I will start with varlen for now in the examples.

```python
--8<-- "examples/cuda_graphs.py:hello-world"
```

1. simple wrapper around the standard pytorch profiler

run it and see how where the very expensive gpu is spending its time:

{{ perfetto_trace(
    "hello_world_no_cuda_graphs",
    title="Hello World without CUDA Graphs",
    alt="Annotated Perfetto crop showing per-operator CPU dispatch and launch gaps on the GPU stream",
) }}


{% call sidenote(
    "We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU",
    "source: [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)",
) %}
This is somewhat a contrived example since we are using small model dim and token counts, regardless "We Bought the Whole GPU, So We're Damn Well Going to Use the Whole GPU" - if we can.
{% endcall %}

#### Graph Time

PyTorch's api pretty closely mirror the lower level gpu apis for CG (CUDA Graphs, im done writing those two words).

```python
--8<-- "examples/cuda_graphs.py:capture-graph"
```

1. We use a side stream to isolate the exact work we want to record.
2. We warmup to initialize any lazy CUDA state such as library handles and then wait on it before the real capture.

```python
--8<-- "examples/cuda_graphs.py:hello-world-graph"
```

1. its replay time!

{{ perfetto_trace(
    "hello_world_with_cuda_graphs",
    title="Hello World with CUDA Graphs",
    alt="Annotated Perfetto crop showing one CUDA Graph launch and captured kernels replaying back-to-back",
) }}

We can see from the trace that the launch gaps are dramatically smaller and the GPU stream is much denser. Note that the 46us `cudaGraphLaunch` shown here is inflated by profiling. In general, when I `look` for CPU overhead I use the PyTorch profiler, but when I want to `measure` it I use Python's built-in timer.


#### Terminology time

<div class="capacity-glossary" markdown="1">

<div class="capacity-definition capacity-definition--tokens" markdown="1">

<span class="capacity-term capacity-term--lead">Max Tokens {{ capacity_symbol("T") }}</span>

The physical token capacity of the static input buffers. The captured graph always sees tensors with {{ capacity_symbol("T") }} token rows. When we capture the worst case, {{ capacity_symbol("T") }} = {{ capacity_symbol("T_max") }}.

<div class="with-sidenote capacity-sidenote" markdown="1">
<div markdown="1">

On replay, <span class="capacity-term">Active Tokens {{ capacity_symbol("L") }}</span> is the amount of real token data. It is available on device as `cu_seqlens[-1]`, and must satisfy <span class="sidenote-ref" tabindex="0" aria-describedby="active-token-capacity-note">{{ capacity_symbol("L") }} &lt;= {{ capacity_symbol("T") }}</span>.

</div>

<aside id="active-token-capacity-note" class="sidenote" markdown="1">

If {{ capacity_symbol("L") }} was greater than {{ capacity_symbol("T") }} where would the `L-T` tokens go?

</aside>
</div>

</div>

<div class="capacity-definition capacity-definition--sequences" markdown="1">

<span class="capacity-term capacity-term--lead">Max Sequences {{ capacity_symbol("N") }}</span>

The number of sequence slots supported by the captured graph. This fixes the shape of `cu_seqlens` to `[N + 1]`.

<div class="with-sidenote capacity-sidenote capacity-sidenote--sequences" markdown="1">
<div markdown="1">

On replay, <span class="capacity-term">Active Sequences {{ capacity_symbol("M") }}</span> may be smaller than {{ capacity_symbol("N") }}. The <span class="sidenote-ref" tabindex="0" aria-describedby="unused-sequence-capacity-note">unused tail</span> is represented by repeating the active token endpoint: `[0, ..., L, L, L]`.

</div>

<aside id="unused-sequence-capacity-note" class="sidenote" markdown="1">

The unused sequence slots have 0 length, since length ==> `cu_seqlen[i+1] - cu_seqlen[i] = L - L = 0`

</aside>
</div>

</div>

</div>

---

The graph therefore keeps the physical [{{ capacity_symbol("T") }}, D] and [{{ capacity_symbol("N") }} + 1] shapes fixed. Each replay changes only the logical workload: {{ capacity_symbol("L") }} &lt;= {{ capacity_symbol("T") }} and {{ capacity_symbol("M") }} &lt;= {{ capacity_symbol("N") }}.

### A More Realistic Training Loop

<details class="code-disclosure" markdown="1">
<summary>Packed batch loader</summary>

```python
--8<-- "examples/cuda_graphs.py:training-batches"
```

</details>

```python
--8<-- "examples/cuda_graphs.py:realistic-training-loop"
```

1. `token_capacity` is a policy typically chosen by the data pipeline and whatever your global batchsize you found acceptable for your model arch. Every replay must satisfy {{ capacity_symbol("L") }} &lt;= {{ capacity_symbol("T") }}.
2. `sequence_capacity` is another data-pipeline quantity. If the only guarantee is that a nonempty sequence contains at least one token, then the active count satisfies {{ capacity_symbol("M") }} &lt;= {{ capacity_symbol("T") }} and choosing {{ capacity_symbol("N") }} `=` {{ capacity_symbol("T") }} covers the absolute worst case - a batch of single token docs. Obviously thats a shitty dataset so this should be gleaned from real data.
3. Varlen attention also requires a static upper bound on the maximum sequence length across every replay. If one sequence can consume the full token budget, the safe bound is `max_seqlen = T`. I will argue later on that the `max_seqlen` arg is bad, and that there are established patterns to avoid paying for it with minimal perf hit.
4. This synthetic loader yields the max capacity-sized batch first. Capture therefore sees the largest physical token buffer and all {{ capacity_symbol("N") }} sequence slots; later batches only change the values copied into those same static tensors.
5. checkout our fancy new use of `mark_kernels`, more in the next section
6. `torch.autograd.grad` makes each parameter gradient an explicit output of the fwd+bwd graph. Just like `loss`, these tensors keep the same graph-pool addresses across replay.
7. The first replay produces the graph outputs for the first batch before we hand its gradients to eager Adam.
8. `parameter.grad = grad` attaches those graph outputs to the ordinary optimizer interface
9. The fused Adam step is updating inplace
10. `set_to_none=True` only removes the `parameter.grad` references. The `graph_grads` tuple still owns the graph outputs, replay writes the next gradients into the same storage and we attach them again before `optimizer.step()`.

Parameters already have stable storage, above we allocate static input buffers once, copy each new batch into them and keep the graph outputs alive so the allocator cannot reuse their storage. TorchTitan provides [utilities](https://github.com/pytorch/torchtitan/blob/2807d3f550fe27db18bd9395ba63176364eaed6d/torchtitan/distributed/cudagraph.py#L189) that hide the input copies; this example performs them directly.

{% call sidenote("more advanced storage handoffs", "What happens when the optimizer and fwd_bwd graph use different allocator pools. With fsdp, a fully unsharded gradient may be produced as a leaf output of the fwd_bwd graph but its stuck in the private pool. You could reconstruct a tensor alias for that known address with `torch._C._construct_storage_from_data_pointer(...)` and `torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(...)`. [PyTorch PR 178215](https://github.com/pytorch/pytorch/pull/178215) adds `storage._resize_with_addr_(size_bytes, address)` a more allocator-aware way to recover storage at a known address. Lots of fun stuff - keeping it simple here.") %}
CG memory management is a big topic. A little to big for this tutorial. For a taste, larger distributed systems can use more advanced storage handoffs instead of retaining and directly consuming the original graph-output tensors.
{% endcall %}

What about intermediaries? This module produce intermediate in projections prior to calling attention. How do we ensure that this output always lands in exactly the right the slot? Even if we could ensure that how do we keep this tensor alive between graph replays??

#### What about intermediates?

The graph does not keep every intermediate Python Tensor alive. During capture the caching allocator gives intermediates addresses from a graph-private pool and the kernels record those exact addresses. On replay Python and the allocator do not recreate the intermediates; CUDA just launches the recorded kernels reading and writing the same addresses. The private pool remains alive until every graph using it and every live tensor created during capture is gone.

#### Small Digression into profiling

<div class="with-sidenote" markdown="1">
<div markdown="1">

Eager code is great. It works seamlessly with the stock pytorch profiler - providing plenty of info for an intrepid user. One very handy feature is `record_function` that lets users annotate regions of your code and have these <span class="sidenote-ref" tabindex="0" aria-describedby="profiler-label-meme-note">labels land on the perfetto trace</span>. If you have gigs to spare you might even use `with_stacks=True` and get a microscopic view of the world. This is not the same for cuda-graphs, historically. Below is an example of what the stock profiler might produce.

</div>

<aside id="profiler-label-meme-note" class="sidenote sidenote--hover-media">
<a href="https://tenor.com/view/nice-smack-delicious-meme-gif-8375212" aria-label="View the Nice Smack meme on Tenor">
<img src="../../assets/memes/nice-smack.webp" alt="Chef approvingly kissing food and saying nice">
</a>
</aside>
</div>

In the stock-profiler half of the merged trace below you can still see which graph launch owns what kernels, their graph and node ids, launch metadata and dependency arrows. This is already useful, but the graph still looks like a wall of kernel names. We can do better!

At PyTorch we believe observability is only becoming more important. What used to take ages for an individual developer to digest can take an agent seconds, provided we give it structured traces instead of screenshots and a wall of kernel names;

{% call sidenote("graph is replayed", "thats the point!") %}
**1.** There are new CUDA Graph annotation APIs. Python does not run the individual operators again when a graph is replayed, so `record_function` cannot recover the internal regions after capture. [`mark_kernels`](https://docs.pytorch.org/docs/main/generated/torch.cuda.graph_annotations.mark_kernels.html) records metadata while the graph is being captured, and [`enable_annotations=True`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.graph.html) keeps the mapping from graph nodes back to those labels. With this info stored on the graph nodes we can write postprocessors like: [transformer-nuggets post processing](https://github.com/drisspg/transformer_nuggets/blob/742466ecdfe9e616210cb32f415f95b3c213cc53/transformer_nuggets/utils/perfetto.py#L98-L167) which joins that metadata to the replayed kernels and reconstructs our labels -> `embedding`, `attention`, `loss` and `backward`.
{% endcall %}

{% call sidenote("Hardware Event System", "Useful for reducing collection overhead, but not required for the annotation flow shown below.") %}
**2.** CUPTI's [Hardware Event System (HES)](https://docs.nvidia.com/cupti/main/main.html#hardware-event-system-hes) is a separate Blackwell feature for collecting kernel timestamps with lower per-node overhead. The comparison below intentionally uses the regular PyTorch profiler on both sides; the only difference is whether graph annotations are enabled.
{% endcall %}

{{ perfetto_trace(
    "hello_world_training_loop_comparison",
    title="Stock versus annotated CUDA Graph training trace",
    alt="Side-by-side Perfetto comparison of a stock CUDA Graph trace and the same replay with reconstructed graph annotation tracks",
) }}

If you think wow these are some well annotated cuda-graph traces now you know why - sidequest done!

### The anatomy of a ragged kernel

A ragged kernel needs to map a regular GPU grid onto sequences with different lengths. Turns out there are a few ways to do this.

Assume that for every token we have some parallel work to do. Attention forward is one example: every input token needs an output. We could launch one CTA per token, no that is dumb and not how gpus work. Instead we tile the tokens into **chunks**. For the rest of this section assume one CTA handles one chunk of \(C\) tokens. Real kernels may divide the work further, but this gives us something concrete to schedule.

Suppose a packed tensor contains {{ capacity_symbol("N") }} sequences and sequence \(i\) contains \(s_i\) tokens. A simple mapping uses one grid axis for the sequence and another for its chunk:

```python
grid(sequence_i, chunk_j)

start = chunk_j * C
if start >= seqlen[sequence_i]:
    # wrong sequence bro
    return
process(sequence_i, start, C)
```
{% call sidenote("\\(\\left\\lceil S_{\\max} / C \\right\\rceil\\)", "this fancy looking upside down hockey stick is ceil division") %}
The chunk axis must be large enough for the longest sequence. Since this is baked into the grid it also must be static, which means we need to launch \(\left\lceil S_{\max} / C \right\rceil\) chunks for each of the {{ capacity_symbol("N") }} sequences:
{% endcall %}

\[
G_{\mathrm{rect}}
= N \left\lceil \frac{S_{\max}}{C} \right\rceil,
\qquad
S_{\max} = \max_i s_i,
\]

This covers every sequence, but it overcounts. The minimal covering is:

\[
G_{\mathrm{active}}
= \sum_{i=0}^{N-1} \left\lceil \frac{s_i}{C} \right\rceil
\]

The problem is that the longest sequence stretches out our required grid. CTAs assigned to chunks outside shorter sequences immediately return, which sounds cheap but it can add up.

[FlashAttention 2](https://github.com/Dao-AILab/flash-attention/blob/v2.5.9/csrc/flash_attn/src/flash_fwd_launch_template.h#L60-L62) used this same rectangular grid launch, requiring the host to keep track of `max_seqlen_q`.

### Perhaps there is another way

<div class="with-sidenote" markdown="1">
<div markdown="1">

We are adding <span class="sidenote-ref" tabindex="0" aria-describedby="new-attention-variants-meme-note">new attention variants to the gym</span> and will use KDA as our first case study.

</div>

<aside id="new-attention-variants-meme-note" class="sidenote sidenote--hover-media">
<a href="https://giphy.com/gifs/OnceInHollywood-leonardo-dicaprio-leo-kd9BlRovbPOykLBMqX" aria-label="View the Leonardo DiCaprio pointing GIF on GIPHY">
<img src="../../assets/memes/pointing-leo.webp" alt="Leonardo DiCaprio pointing excitedly at a television">
</a>
</aside>
</div>

#### 1. Scheduling without `max_seqlen`

If you look at our [`chunk_kda`](https://github.com/meta-pytorch/attention-gym/blob/b286b35ffab089bc24f6513e6effb9d29ece89e7/attn_gym/linear/kda/api.py#L29-L42) function you will notice it does not accept `max_seqlen` at all. But how!?

The rectangular scheduler uses `max_seqlen` to decide how many chunk slots every sequence should receive. What if we stop treating our grid as rectangular but instead flatten all of the sequence-local chunks into one minimal logical work list. A CUDA Graph-safe kernel can cover that list with either a static capacity grid or bounded persistent workers, then use offsets to recover which sequence owns each chunk. This sounds kinda similar to what we originally did for ragged tensors in memory!

For sequence lengths \(s_i\), we build `chunk_offsets`:

\[
\mathtt{chunk\_offsets}[0] = 0,
\qquad
\mathtt{chunk\_offsets}[i+1]
= \mathtt{chunk\_offsets}[i] + \left\lceil \frac{s_i}{C} \right\rceil.
\]

This is just another prefix sum over sequences saying how many chunks are needed to cover each. `chunk_offsets[i]` is where sequence \(i\)'s chunks begin in the flat list, and

<div class="with-sidenote with-sidenote--equation" markdown="1">
<div markdown="1">
<div class="sidenote-ref sidenote-ref--equation" tabindex="0" aria-describedby="chunk-offsets-active-note" markdown="1">

\[
\mathtt{chunk\_offsets[-1]} = G_{\mathrm{active}}
\]

</div>
</div>

<aside id="chunk-offsets-active-note" class="sidenote" markdown="1">

KDA runs a small metadata kernel to build this transformed prefix once, instead of each sub kernel recomputing this.

</aside>
</div>

##### Dont we already have a prefix sum -> `cu_seqlens`

The ceiling must be applied separately to each sequence because chunking resets at every sequence boundary. For \(C = 64\):

<div class="chunk-offset-viz" role="figure" aria-label="Build chunk offsets by taking adjacent differences of cumulative sequence lengths, ceiling each sequence length by the chunk size, and taking another prefix sum. Directly ceiling cumulative sequence lengths produces the wrong answer.">
<div class="chunk-offset-viz__steps">
<div class="chunk-offset-viz__step chunk-offset-viz__step--input">
<span>cu_seqlens</span>
<code>[0, 65, 128]</code>
</div>
<div class="chunk-offset-viz__connector"><span>difference</span><b aria-hidden="true">→</b></div>
<div class="chunk-offset-viz__step chunk-offset-viz__step--lengths">
<span>lengths</span>
<code>[65, 63]</code>
</div>
<div class="chunk-offset-viz__connector"><span>ceil each / 64</span><b aria-hidden="true">→</b></div>
<div class="chunk-offset-viz__step chunk-offset-viz__step--chunks">
<span>chunks</span>
<code>[2, 1]</code>
</div>
<div class="chunk-offset-viz__connector"><span>prefix sum</span><b aria-hidden="true">→</b></div>
<div class="chunk-offset-viz__step chunk-offset-viz__step--output">
<span>chunk_offsets</span>
<code>[0, 2, 3]</code>
</div>
</div>
<div class="chunk-offset-viz__wrong">
<code>ceil_div(cu_seqlens, 64)</code>
<b aria-hidden="true">→</b>
<code>[0, 2, 2]</code>
<strong>wrong boundary</strong>
</div>
</div>

The flat grid gives each CTA one integer, `flat_chunk`, but the CTA still needs to know which sequence it belongs to and where that chunk begins in the packed tensor. With our handy new prefix sums we can do the following:

```python
sequence = search_first_greater(chunk_offsets, flat_chunk) - 1
local_chunk = flat_chunk - chunk_offsets[sequence]
token_start = cu_seqlens[sequence] + local_chunk * C
```

For the example above, `chunk_offsets = [0, 2, 3]`. Flat chunks `0` and `1` belong to sequence `0`; flat chunk `2` belongs to sequence `1` and is its local chunk `0`, beginning at packed-token offset `65`. Since our sums are monotonic we can binary_search our way to the right location.

There is 1 more missing piece to the puzzle though; what is the host launch size? Instead of getting it from `max_seqlen`, we can derive an upper bound from the static token capacity {{ capacity_symbol("T") }}, sequence capacity {{ capacity_symbol("N") }} and chunk size \(C\):

Let {{ capacity_symbol("M") }} be the number of nonempty sequences in this replay. We know \(M \le N\) because there are only {{ capacity_symbol("N") }} sequence slots, and \(M \le T\) because every nonempty sequence needs at least one token. Therefore \(M \le \min(T,N)\).

{% call sidenote("\\(\\lceil s_i/C \\rceil = 1 + \\lfloor (s_i-1)/C \\rfloor\\)", "exercise for the reader") %}
For each nonempty sequence, \(\lceil s_i/C \rceil = 1 + \lfloor (s_i-1)/C \rfloor\). Starting from the real chunk count:
{% endcall %}

<div class="with-sidenote with-sidenote--equation" markdown="1">
<div markdown="1">
<div class="sidenote-ref sidenote-ref--equation" tabindex="0" aria-describedby="floor-division-note" markdown="1">

\[
\begin{aligned}
G_{\mathrm{active}}
&= \sum_{i=0}^{M-1} \left\lceil \frac{s_i}{C} \right\rceil \\
&= M + \sum_{i=0}^{M-1} \left\lfloor \frac{s_i-1}{C} \right\rfloor \\
&\le M + \left\lfloor \frac{\sum_{i=0}^{M-1} s_i-M}{C} \right\rfloor \\
&\le M + \left\lfloor \frac{T-M}{C} \right\rfloor.
\end{aligned}
\]

</div>
</div>

<aside id="floor-division-note" class="sidenote" markdown="1">

regular hockey stick = floor division

</aside>
</div>

Using the largest possible active sequence count, \(M_{\max} = \min(T,N)\), gives the host capacity:

\[
G_{\mathrm{cap}}(T,N,C)
= M_{\max} + \left\lfloor \frac{T-M_{\max}}{C} \right\rfloor.
\]

The per-sequence tile-prefix and flat-decoding pattern is not a new idea, see:

- **FA3 forward:** [`prepare_varlen_num_blocks_kernel`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/hopper/flash_prepare_scheduler.cu#L43-L211) reads `cu_seqlens` and writes each sequence's tile count. [`tile_idx_to_work_tile`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/hopper/tile_scheduler.hpp#L596-L690) has the persistent grid claim a flat tile id and map to the sequence that owns it.
- **FA4:** for large batches and the applicable static/CLC scheduler paths, [`_compute_tile_cumsum`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/interface.py#L411-L485) builds `cu_total_m_blocks`, and [`VarlenDecoder.decode`](https://github.com/Dao-AILab/flash-attention/blob/0251105a2fb19d2957484b7f023cd8c115286ced/flash_attn/cute/tile_scheduler.py#L1011-L1110) maps it back to its sequence.

#### 2. Graph overcapture: bound capacity-sized launches

Time to bring CUDA Graphs back into this party. We just saw how to not pay worst case sequence peformance with an updated grid. What about worst case num_tokens? During capture we have to use {{ capacity_symbol("T_max") }} tokens - for the worst case.

Although we have tightend the bounds - a graph captured for {{ capacity_symbol("T_max") }} keeps launching `chunk_capacity(T_max, N, C)` even when a replay contains only {{ capacity_symbol("L") }} `<<` {{ capacity_symbol("T_max") }} active tokens.

<div class="with-sidenote" markdown="1">
<div markdown="1">

What we need is a yet another type of Grid Schedule. Building off of the previous lets add a <code class="sidenote-ref" tabindex="0" aria-describedby="persistent-scheduler-note">PERSISTENT</code> schedule.

</div>

<aside id="persistent-scheduler-note" class="sidenote" markdown="1">

A pretty common strategy on modern GPUs. It does have some gotchas though, see → [A Tale of Two Schedulers](https://drisspg.github.io/nuggets/A-Tale-of-Two-Schedulers).

</aside>
</div>

Here is the flat-task Triton form. CuTeDSL K3/K4 keep their pair/head grid axes and stride only the chunk axis, but the idea is the same:

```Python
capacity_tasks = chunk_capacity(T_max, N, chunk) * subtasks
num_workers = min(capacity_tasks, num_sms * ctas_per_sm)
launch grid(num_workers)

kernel(worker):
    active_tasks = chunk_offsets[-1] * subtasks
    for task in range(worker, active_tasks, num_workers):
        process(task)
```

Each worker reads the active task count from the device and strides over the real task list. The launch stays fixed but is capped at the resident-worker count; the task-loop work scales with the active list instead of worst case tokens. This does not mean persistence always wins—the static early-exit grid is still better when captured and active work are close.

#### Show me the numbers!

There are really two separate ways our captured graph can guess too high:

1. **Too many sequences.** Keep the real token count fixed at {{ capacity_symbol("L") }} `=8192`, capture room for {{ capacity_symbol("N") }} `=256` sequences and change how many real sequences {{ capacity_symbol("M") }} make up those tokens.
2. **Too many tokens.** Capture {{ capacity_symbol("T_max") }} `=16384` tokens, keep {{ capacity_symbol("M") }} `=64` and replay progressively fewer real tokens {{ capacity_symbol("L") }}.

For every complete implementation I also capture an `ideal` graph for the actual `(L, M, max_seqlen)` shape with dashed lines. Comparing the worst-case graph against the `ideal` graph shows how much forward-plus-first-order-backward replay time we lose to extra scheduling capacity. The FA4 persistent-forward series uses exact points for its slowdown denominator but does not draw a separate dashed trace.

{{ plotly_chart(
    "ragged_attention_overcapture",
    title="Captured versus exact CUDA Graphs under fragmentation and token overcapture",
    height=880,
) }}

Measured on B200 as CUDA Graph replay only: pooled p50 of 30 samples with p05-p95 error bars. Capture, allocation, input updates, correctness checks and host dispatch are excluded. The FA2/FA4 `L < T` cases are implementation probes outside the public exact-packed varlen contract; KDA has an explicit fixed-capacity contract.

### Does this hold E2E

I have been cheating a little. Every operation above is a ragged kernel that natively supports `cu_seqlens`. A real attention module also has token paralell QKV and output projections, normalization, masks and a bunch of ordinary PyTorch operations.

Most of those operations only see a physical [{{ capacity_symbol("T_max") }}, D] tensor. Therefore they processes {{ capacity_symbol("T_max") }} rows even when only {{ capacity_symbol("L") }} are real. At {{ capacity_symbol("T_max") }} `=8192` and {{ capacity_symbol("L") }} `=512`, that is 16 times as many rows!

#### Correctness tangent

Another sidequest but this one is important because I have seen many an issue that boils down to not masking the padding tokens correclty. The inactive padding token suffix of your ragged tensor is not harmless. Its easy to remeber these tokens in the fwd; i.e. loss mask, you cant forget about the grad. A op may leave inactive outputs or input gradients undefined. That is kind of the point from the above secitons; dont do wasted work and only zero out the padded tokens when you need to. Ohh the joys of floating points; `NaN * 0 = NaN`.

The training example therefore uses two different masks:

```python
--8<-- "examples/kda_training.py:kda-fixed-capacity-masking"
```

1. Build one reusable device mask from `cu_seqlens[-1]`. Because this happens inside capture, replay rebuilds it from the current device endpoint without a host read.
2. Value masking writes the inactive rows to zero before an ordinary operation can read or save them.
3. Leave the forward tensor unchanged, but zero its inactive-row gradients during backward. Placing the barrier after `qkv_projection` prevents undefined inactive gradients from the short convolution from contaminating the projection's weight-gradient reduction.

The mental model is: mask where inactive rows can participate in a mix or reduction, or where inactive values or gradients become undefined.

A projection `Y = XW` is token-parallel in the forward, but its weight gradient reduces over every physical row:

\[
\nabla W = X^\mathsf{T}\nabla Y = \sum_t X_t^\mathsf{T}\nabla Y_t.
\]

Zero out the padding before an operation where inactive rows can mix, contribute to a token-global statistic, or be saved for a backward reduction over rows. If a ragged op may produce undefined gradients for padded rows, zero them before they reach the previous layer. Here, `mask_inactive_token_gradients` stops the short convolution's invalid padded-row `dInput` before `qkv_projection` uses it to compute `dW`.

mini-TLDR: pure tokenwise operations with defined padded outputs and gradients do not need a mask at every boundary. The masks belong where rows first mix, reduce, or become undefined.

#### A real KDA module

Putting whole `KDAAttention` module back together: QKV and output projections, normalization, short convolution, masking, the KDA core and backward. This Kimi-style shape uses {{ capacity_symbol("T_max") }} `=4096`, {{ capacity_symbol("N") }} `=32`, hidden size 2304 and 32 KDA heads.

{{ plotly_chart(
    "kda_module_amdahl_scaling",
    title="KDAAttention scaling and its fixed-shape module floor",
) }}

At full capacity `chunk_kda` core is 1.23 ms versus 3.43 ms for the complete module. At half tokens kda falls to 0.77 ms -> a 38% reduction. However, the complete module only falls to 2.91 ms, or 15%. Amdahl's law strikes again! The padding-aware part scales while most surrounding work still processes {{ capacity_symbol("T_max") }} rows :(

Below is a merged trace of full and half tokens. Feel free to inspect for yourself.

{{ perfetto_trace(
    "kda_module_full_vs_half_replay",
    title="Full versus half-token KDAAttention CUDA Graph replay",
    alt="Annotated merged Perfetto crop comparing full and half-token replay of the same fixed-capacity KDAAttention graph",
) }}

### Could we use more than one graph?

This MM tax seems harder to fight without rewriting all of PyTorch. What if we just captured a few physical token capacities and selected the smallest one that fits {{ capacity_symbol("L") }}?

```python
capacity = min(candidate for candidate in capacities if L_host <= candidate)
graphs[capacity].replay()
```

Here `L_host` is the token count the data loader already knows; reading `cu_seqlens[-1].item()` only to dispatch would introduce a device-to-host sync. Compute-wise this is great, but we have to be careful about memory.

#### Pools and Pools of Memory

As we know, replay needs the same exact virtual addresses it saw during capture. PyTorch protects those addresses by retaining graph allocations in a [private allocator pool](https://docs.pytorch.org/docs/2.13/notes/cuda.html#graph-memory-management). This however causes memory fragmentation: the default allocator cannot borrow an inactive private block while the graph still owns it.

If one Naively captures every sub graph in its own pool then we get no memory resuse. But if we capture them on the same side stream and ensure their GPU executions never overlap, we can give them one shared pool. Our static inputs and outputs can all be slices from max-sized backing buffers!

```python
static_input = torch.empty((T_max, D), device="cuda")
static_output = torch.empty_like(static_input)
pool = torch.cuda.graph_pool_handle()

for capacity in capacities:
    with torch.cuda.graph(graphs[capacity], pool=pool):
        static_output[:capacity].copy_(model(static_input[:capacity]))
```

All input prefix views share the input base address, and all output prefix views share a separate output base address. Graph-local intermediates can reuse the same memory because only one bucket executes at a time.

{{ html_widget(
    "cuda-graph-memory/pool-strategies.html",
    title="One maximum graph versus multiple graph memory pools",
    classes="memory-pools-frame",
) }}

<p class="memory-pools-caption" markdown="1">Measured on B200 from fresh-process allocator snapshots of the same two-projection BF16 core, {{ capacity_symbol("T_max") }} `=1024` and `D=4096`; the shared-pool case also copies into an external output buffer.</p>

{{ html_widget(
    "cuda-graph-memory/max-vs-four-shared-buckets.html",
    title="Allocator traces for one maximum graph and four shared-pool capacity graphs",
    classes="memory-viz-frame",
) }}

#### Crossing graph boundaries

Sharing a pool also shares lifetimes. A pool-backed output is temporary and can be snatched from under you: the next graph using that pool may overwrite it. This can be tricky in real programs, so Inductor has some great functionality to help here: [CUDA Graph Trees](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) manage compatible path generations and pending forward/backward lifetimes in one pool.

<div class="with-sidenote" markdown="1">
<div markdown="1">

A common boundary break in CG for training is `fwd_bwd -> optimizer`. The optimizer graph expects every gradient at the address it captured. Copying graph-owned grads into persistent eager buffers works, but now some of your largest tensors exist twice!  <span class="sidenote-ref" tabindex="0" aria-describedby="optimizer-address-handoff-note">There are tricks that</span> can release the storage between steps and recover it before optimizer replay but suffice it to say be careful and use the memory profiler to help you track all your allocations.

</div>

<aside id="optimizer-address-handoff-note" class="sidenote sidenote--hover-media" markdown="1">

New Api alert!: [`claim`](https://github.com/pytorch/pytorch/pull/178215) is `_resize_with_addr_`, `release` is `resize_(0)`. Use one stream or explicit events.

```python
# capture
A, n = save(grad)
opt_g = capture(
    optimizer
)
release(grad)

# replay
bwd_g.replay()
claim(pool, A, n)
opt_g.replay()
release(grad)
```

</aside>
</div>

We have a nice little system setup now: a small set of CUDA Graphs handles the coarse changes in physical token count, while our device-driven ragged kernels handle the sequence count, lengths and remaining underfill within each graph. Together the demonstrated KDA/FA schedulers can scale across many input sizes without needing one graph for every exact shape!

### Show me the data!

{% call sidenote("like all great ML problems", "all problems in CS can be solved with pipelining and buffering") %}
Hopefully you are convinced from the above that padding tokens are expensive, and there are lower level ways to mitigate this cost. HOWEVER, like all great ML problems we have try to engineer our way out of this problem -> when perhaps we should have just looked at the data..
{% endcall %}

I sampled 20,000 source rows from [Dolma 3's 5.93T-token mix](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B), the OLMo 3 7B stage-1 pretraining mix. Using the [OLMo 3 tokenizer](https://huggingface.co/allenai/Olmo-3-1025-7B), I split them into 8192 token capped segments. A trick to get much better packing is to let your dataloader look ahead and try to pack samples from this `buffer of candidates`. Thus a candidate buffer of 32 means each batch may inspect and select from at most 32 entries.e

The figure below sweeps local token budgets from 8K to 32K and shows how much of each budget this packing policy actually fills.

{{ plotly_chart(
    "packing_fill_by_dataset",
    title="Packing efficiency by dataset and candidate-buffer size",
    height=610,
) }}


For candidate-buffer size 32:

| local {{ capacity_symbol("T") }} | median {{ capacity_symbol("M") }} | p99 {{ capacity_symbol("M") }} | mean fill | mean padding |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1 | 13 | 99.80% | 16 tokens |
| 16384 | 4 | 32 | 99.75% | 40 tokens |

For this Dolma segmentation and packing policy we probably dont even need multiple token-capacity graphs. {{ capacity_symbol("M") }} `<=32` is guaranteed by the candidate policy, so {{ capacity_symbol("N") }} `=32` is sufficient by construction; the loader also fills the token budget almost perfectly. The raw row overhead of one graph is about 0.2% at 8K and 0.25% at 16K.

Is this trend universal? No. An illustrative, not tokenizer-controlled comparison, look at [FineWeb `sample-10BT`](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/9bb295ddab0e05d785b879661af7260fed5140fc) sample using `tiktoken:o200k_base`. At a 16K budget its mean underfill falls from 8.8% with candidate buffer 32 to 0.7% with buffer 64.

### Conclusion

This lil tutorial ended up more meandering then expected, but, in the end we have found 3 complimentary techincques for handiling ragedness in training:

1. **Measure the data, then pack it away in the dataloader.** Choose {{ capacity_symbol("T") }}, {{ capacity_symbol("N") }} and the candidate-buffer policy from the real sequence distribution instead of assuming one packing policy works everywhere.
2. **Make the sequence-aware kernels read the real work dynamically on the GPU.** The demonstrated KDA and FA schedulers can support multiple schedules. A runtime work-aware persistent grid can substantially reduce padding work when overcapture is large, while the static early-exit grid can still win near full capacity.
3. **Add graph buckets in a way that limits memory fragmentation.** A core [vLLM idea](https://github.com/vllm-project/vllm/blob/9bf0d4717ecd5ad9e6c5fe3eb933cf654a256321/docs/design/torch_compile.md#L242-L257) applied to training: capture a finite list of physical capacities, select the smallest `T_bucket >= L`, and share one [global graph pool](https://github.com/vllm-project/vllm/blob/9bf0d4717ecd5ad9e6c5fe3eb933cf654a256321/vllm/platforms/interface.py#L1152-L1159).

The final recipe is: measure the dataset, pick {{ capacity_symbol("T") }}, {{ capacity_symbol("N") }} and engineer prolbems out in the dataloader when you can, start with one graph, add capacity buckets only if the measured padding justifies them, let sequence-aware kernels read the actual work from the GPU and mask whatever padding is left.
