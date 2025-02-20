import torch

ROPE_GROUP_SIZE = 4

def rope_embedding(Q, cos, sin, seqlen, head_dim, n_heads, backward_pass=False):
    ROPE_GROUP_SIZE = 4
    row_position = torch.arange(0, Q.size(0))
    group_head_position = torch.arange(0, (n_heads + ROPE_GROUP_SIZE - 1) // ROPE_GROUP_SIZE)

    half_head_dim = head_dim // 2
    cos1 = cos[row_position % seqlen, :half_head_dim]
    sin1 = sin[row_position % seqlen, :half_head_dim]

    if backward_pass:
        sin1 = -sin1

    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min(head_start + ROPE_GROUP_SIZE, n_heads)

    for k in range(head_start, head_end):
        offs_q1 = k * head_dim
        offs_q2 = k * head_dim + half_head_dim

        Q1 = Q[:, offs_q1:offs_q1 + half_head_dim].float()
        Q2 = Q[:, offs_q2:offs_q2 + half_head_dim].float()

        Q[:, offs_q1:offs_q1 + half_head_dim] = Q1 * cos1 - Q2 * sin1
        Q[:, offs_q2:offs_q2 + half_head_dim] = Q2 * cos1 + Q1 * sin1

    return Q

class Fast_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos = cos.squeeze()
        sin = sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        Q = Q.view(batch * seq_len, n_heads * head_dim)
        n_rows, n_cols = Q.shape
        assert(seq_len <= cos.shape[0])

        BLOCK_SIZE = head_dim // 2
        div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
        n_groups = div + (mod != 0)

        Q = rope_embedding(Q, cos, sin, seq_len, head_dim, n_heads)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.n_groups = n_groups
        ctx.cos = cos
        ctx.sin = sin
        return Q.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.view(batch * seq_len, n_heads * head_dim)
        cos = ctx.cos
        sin = ctx.sin

        dY = rope_embedding(dY, cos, sin, seq_len, head_dim, n_heads, backward_pass=True)
        return dY.view(batch, seq_len, n_heads, head_dim), None, None

def fast_rope_embedding(Q, K, cos, sin):
    Q = Fast_RoPE_Embedding.apply(Q.transpose(1, 2), cos, sin).transpose(1, 2)
    K = Fast_RoPE_Embedding.apply(K.transpose(1, 2), cos, sin).transpose(1, 2)
    return Q, K

class Slow_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids):
        if position_ids is not None:
            cos = cos.squeeze(1).squeeze(0)
            sin = sin.squeeze(1).squeeze(0)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)

        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        Q *= cos
        Q.addcmul_(RH_Q, sin)
        ctx.save_for_backward(cos, sin)
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        half = dY.shape[-1] // 2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim=-1)
        dY *= cos
        dY.addcmul_(RH_dY, sin)
        return dY, None, None, None

def inplace_rope_embedding(Q, K, cos, sin, position_ids):
    Q = Slow_RoPE_Embedding.apply(Q, cos, sin, position_ids)
    K = Slow_RoPE_Embedding.apply(K, cos, sin, position_ids)
    return Q, K
