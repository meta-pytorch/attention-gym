# Examples

Examples follow the library's high-level concepts, with **at most one directory level** below
`examples/`. Run commands from the repository root after installing the package and the optional
dependencies required by the example.

| Directory | Contents |
| --- | --- |
| `flex_attention/` | FlexAttention masks/score mods, walkthroughs, MLA, paged and ring attention, compilation, CUDA Graphs, and tuning |
| `linear/` | KDA/GDN training, context parallelism, and recurrent decoding |
| `sparse/` | Compressed sparse attention, VSA, and FastWan integration |

Start with:

```bash
jupyter notebook examples/flex_attention/flex_attn.ipynb
python examples/linear/delta_rule_training.py --backend=reference
python examples/sparse/compressed_sparse_attention.py
```

The paged-attention recipe stays flat: `flex_attention/paged_attention.py` contains the cache,
`paged_attention_model.py` and `paged_attention_utils.py` support it, and
`paged_attention_latency.py` / `paged_attention_throughput.py` are the runnable benchmarks.

Keep variant names in filenames instead of adding nested directories. Examples remain editable
teaching recipes, not another public API layer. See `docs/examples.md` for the full guide.
