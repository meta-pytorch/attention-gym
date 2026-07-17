import torch
import reference


class compressed_sparse_attention(torch.autograd.Function):
    '''

    '''
    def forward(Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I, K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, attention_sink, compression_rate: int, num_topk_blocks: int, sliding_window_size:int, share_kv: bool):
        