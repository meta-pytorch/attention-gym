# SPDX-License-Identifier: BSD-3-Clause

"""The layout of the FP32 piece entry states an uncut KDA forward hands a chain backward (NOTE
[Piece entry states] in ``drivers.kda``). It imports nothing, so the custom ops' fake
implementations size the buffer without CuTeDSL installed."""


def entry_state_shape(rows, heads_out: int, dim_v: int, dim_k: int) -> tuple:
    """The FP32 piece entry states' shape (NOTE [Entry state layout] in ``kda_prefill``)."""
    return (rows, heads_out, dim_k // 4, dim_v, 4)
