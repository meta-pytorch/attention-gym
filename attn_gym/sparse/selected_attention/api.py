from . import reference


def selected_attention(
    Q,
    KV,
    index_kv,
    indices, 
    attention_sink,
    doc_ids,
    sliding_window_size: int,
    share_kv: bool = True,
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