import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Debugging Score Mods:

    This notebook demonstrates how to debug attention score modifications using FlexAttention's
    new debugging capabilities. We'll implement a "anti sliding window" attention pattern where:
    - positions within some window we decrease the score
    - positions outside of this window we will boost by their distance


    Idk why but seems interesting, maybe you think local information hurts your sequence modeling 🤷

    **The Problem**: FlexAttention was hard to introspect because we always enbaled dynamo compilation, which doesn't let you use breakpoints or print statments.

    **The Solution**: FlexAttention's new debug flag lets us set breakpoints and inspect
    actual tensor values!
    """
    )


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn.attention.flex_attention as fa

    flex_attention = fa.flex_attention
    return fa, flex_attention, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 1: Implementing Checker Attention (Broken Version)

    Let's start with a "broken" implementation that's hard to debug:
    """
    )


@app.cell
def _(torch):
    def create_broken_mod(window_size: int):
        def broken_anti_window_mod(score, batch, head, q_idx, kv_idx):
            # You probably already see the problem
            score_distance = q_idx - kv_idx
            score_boost = score + score_distance
            window_penalty = score - 1.0
            return torch.where(score_distance <= window_size, window_penalty, score_boost)

        return broken_anti_window_mod

    return (create_broken_mod,)


@app.cell
def _(create_broken_mod, flex_attention, mo, torch):
    mo.md("### Running the broken version - let's see what happens:")

    # Setup
    B, H, S, D = 1, 1, 16, 16
    torch.manual_seed(42)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    broken_score_mod = create_broken_mod(4)
    # Run broken version
    with torch.no_grad():
        broken_output = flex_attention(q, k, v, score_mod=broken_score_mod)
    print(broken_output[0, 0, 1, :])

    return broken_score_mod, k, q, v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The old way

    It outputs a tensor, so probaly fine. But we can't see the actual attention scores or how they change with different inputs. Lets do the old way and look at the attention scores graph
    """
    )


@app.cell
def _(broken_score_mod, k, q):
    from attn_gym.utils import plot_attention_scores

    graph = plot_attention_scores(
        query=q, key=k, score_mod=broken_score_mod, device="cpu", figsize=(8, 8)
    )

    graph  # noqa: B018
    return (plot_attention_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Old Way - Manual Debugging (Painful!)

    This matrix forsure doesn't look right, I would expand some band of blue around the diagonal, but why is hte lower left seem to be so hot?!

    Before the debug flag, we'd have to:

    1. Add print statements everywhere
    2. Create separate test functions
    3. Guess what the compiled code was doing
    4. Manually compute expected values

    This was time-consuming and error-prone!
    """
    )


@app.cell
def _(flex_attention, k, q, torch, v):
    # To prove to you, lets try printing in a new function (new because marimo makes it so)

    def create_broken_mod_2(window_size: int):
        def broken_anti_window_mod(score, batch, head, q_idx, kv_idx):
            # You probably already see the problem
            score_distance = q_idx - kv_idx
            score_boost = score + score_distance
            window_penalty = score - 1.0
            print(f"The new score is: {score_distance}")
            return torch.where(score_distance <= window_size, window_penalty, score_boost)

        return broken_anti_window_mod

    with torch.no_grad():
        flex_attention(q, k, v, score_mod=create_broken_mod_2(4))

    # The dreaded : It looks like you're calling .item() on a Tensor!!


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Step 3: The New Way - Debug Flag Magic! ✨

    Now we can use FlexAttention's debug flag to set breakpoints and inspect values:
    """
    )


@app.cell
def _(fa, flex_attention, k, q, torch, v):
    unwrap = torch._C._functorch.get_unwrapped

    # ⬇️⬇️⬇️ NEW FLAG ALERT ⬇️⬇️⬇️
    # The NEW Flag, but be careful w/ it and read the doc block next to it please :) Yes I wrote this
    fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
    # ⬆️⬆️⬆️ HANDLE WITH CARE ⬆️⬆️⬆️

    def create_broken_mod_3(window_size: int):
        def broken_anti_window_mod(score, batch, head, q_idx, kv_idx):
            # You probably already see the problem
            score_distance = q_idx - kv_idx
            score_boost = score + score_distance
            window_penalty = score - 1.0
            # Not as pretty as printing but we need to do some magic to the the inner vmap tensor
            new_score = torch.where(score_distance <= window_size, window_penalty, score_boost)
            print(f"The new score is: {unwrap(score_distance)}")
            return new_score

        return broken_anti_window_mod

    create_broken_mod_3 = create_broken_mod_3(4)

    with torch.no_grad():
        flex_attention(q, k, v, score_mod=create_broken_mod_3)


@app.cell
def _(flex_attention, k, plot_attention_scores, q, torch, v):
    # Ahhh that looks wrong, also I wrote the bad thing... we should be using abs to get the real score distance
    def create_broken_mod_4(window_size: int):
        def broken_anti_window_mod(score, batch, head, q_idx, kv_idx):
            # You probably already see the problem
            score_distance = torch.abs(q_idx - kv_idx)
            score_boost = score + score_distance
            window_penalty = score - 1.0
            new_score = torch.where(score_distance <= window_size, window_penalty, score_boost)
            return new_score

        return broken_anti_window_mod

    create_broken_mod_4 = create_broken_mod_4(4)

    with torch.no_grad():
        flex_attention(q, k, v, score_mod=create_broken_mod_4)

    flex_viz = plot_attention_scores(
        query=q, key=k, score_mod=create_broken_mod_4, device="cpu", figsize=(8, 8)
    )
    flex_viz  # noqa: B018


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why the unwraps?

    In eager execution, FlexAttention uses PyTorch's `vmap` to apply your score_mod function across all batch, head, query, and key dimensions. This creates nested "wrapper" tensors that need to be unwrapped to access the actual values.

    **The Unwrapping Rules:**

    - **`q_idx`, `kv_idx`, `batch`, `head`**: Need **1 layer** of unwrapping
      ```python
      actual_q_idx = unwrap(q_idx)
      actual_kv_idx = unwrap(kv_idx)
      actual_batch = unwrap(batch)
      actual_head = unwrap(head)
      ```

    - **`score`**: Needs **4 layers** of unwrapping
      ```python
      actual_score = unwrap(unwrap(unwrap(unwrap(score))))
      ```

    **Why the difference?**

    The score tensor gets wrapped 4 times because vmap is applied hierarchically:
    1. **Batch dimension** vmap wrapper
    2. **Head dimension** vmap wrapper
    3. **Query dimension** vmap wrapper
    4. **Key dimension** vmap wrapper


    So if you if you need to debug an expression involving score you will need 4 unwraps to get to the innner `tensor` and 1 for expressions on the indices.
    """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Key Takeaways
    🔧 **Debug Flag for FlexAttention**:

    - `fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True`
    - Allows breakpoints and prints in score_mod functions
    - Use `unwrap()` to inspect tensor values during debugging
    - **Requires PyTorch 2.9 nightly or later**
    """
    )


if __name__ == "__main__":
    app.run()
