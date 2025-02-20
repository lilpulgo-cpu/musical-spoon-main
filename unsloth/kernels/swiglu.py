import torch

def swiglu_fg_kernel(e, g):
    batch, seq_len, hd = e.shape
    n_elements = e.numel()
    h = torch.empty((batch, seq_len, hd), dtype=e.dtype, device=e.device)
    
    e_row = e.view(-1)
    g_row = g.view(-1)
    
    f_row = e_row * torch.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row
    
    h = h_row.view(batch, seq_len, hd)
    return h

def swiglu_DWf_DW_dfg_kernel(DW, e, g):
    batch_seq_len, hd = e.shape
    n_elements = e.numel()

    DW_row = DW.view(-1)
    e_row = e.view(-1).float()
    g_row = g.view(-1)

    se_row = torch.sigmoid(e_row)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)
    h_row = f_row * g_row
    df_row = DW_row * f_row
    dg_row = DW_row * g_row
    de_row = dg_row.to(torch.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    DW = h_row.view(batch_seq_len, hd)
    e = df_row.view(batch_seq_len, hd)
    g = de_row.view(batch_seq_len, hd)
    
    return DW, e, g
