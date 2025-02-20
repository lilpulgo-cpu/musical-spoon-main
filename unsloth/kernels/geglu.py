import torch

def geglu_exact_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    e_row = gate.view(-1).float()
    g_row = up.view(-1)
    
    f_row = 0.5 * e_row * (torch.erf(e_row / torch.sqrt(torch.tensor(2.0))) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row
    
    out = h_row.view(batch, seq_len, hd)
    return out

def geglu_exact_backward_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    e_row = e.view(-1).float()
    g_row = g.view(-1)
    DW_row = DW.view(-1)

    f_partial_row = 0.5 * (torch.erf(e_row / torch.sqrt(torch.tensor(2.0))) + 1.0)
    f_row = f_partial_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row

    t = 0.3989422804014327
    df_de = f_partial_row + t * e_row * torch.exp(-0.5 * e_row * e_row)
    
    de_row = dg_row.to(torch.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    DW = h_row.view(batch_seq_len, hd)
    e = df_row.view(batch_seq_len, hd)
    g = de_row.view(batch_seq_len, hd)
    
    return DW, e, g

def geglu_approx_forward_kernel(gate, up):
    batch, seq_len, hd = gate.shape
    e_row = gate.view(-1).float()
    g_row = up.view(-1)
    
    s = 0.7978845608028654
    f_row = 0.5 * e_row * (
        torch.tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0
    )
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row
    
    out = h_row.view(batch, seq_len, hd)
    return out

def geglu_approx_backward_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    e_row = e.view(-1).float()
    g_row = g.view(-1)
    DW_row = DW.view(-1)

    s = 0.7978845608028654
    a = s * e_row
    b = a * 0.044715 * e_row * e_row
    T = 1.0 + torch.tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2
    
    f_row = T2 * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row

    de_row = dg_row.to(torch.float32) * df_de
    de_row = de_row.to(DW_row.dtype)

    DW = h_row.view(batch_seq_len, hd)
    e = df_row.view(batch_seq_len, hd)
    g = de_row.view(batch_seq_len, hd)
    
    return DW, e, g
