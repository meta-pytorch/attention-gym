from typing import Literal

from . import reference


Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]


def selected_attention(
    Q,
    KV,
    index_kv,
    indices,
    attention_sink,
    doc_ids,
    sliding_window_size: int,
    share_kv: bool = True,
    backend: Backend = "eager",
    mode: Mode = "auto"
):
    return reference.selected_attention(
        Q,
        KV,
        index_kv,
        indices,
        attention_sink,
        doc_ids,
        sliding_window_size,
        share_kv,
    )
