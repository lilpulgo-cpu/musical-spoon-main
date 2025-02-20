import torch
torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")



def calculate_settings(n):
    BLOCK_SIZE = next_power_of_2(n)
    num_warps = 4
    if BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

def QUANT_STATE(W):
    return getattr(W, "quant_state", None)

def get_lora_parameters(proj):
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None

    active_adapter = proj.active_adapters[0] if hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s

def get_lora_parameters_bias(proj):
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight
    bias = base_layer.bias

    if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
        return W, QUANT_STATE(W), None, None, None, bias

    active_adapter = proj.active_adapters[0] if hasattr(proj, "active_adapters") else proj.active_adapter
    A = proj.lora_A[active_adapter].weight
    B = proj.lora_B[active_adapter].weight
    s = proj.scaling[active_adapter]
    return W, QUANT_STATE(W), A, B, s, bias

def fast_dequantize(W, quant_state=None, out=None):
    if quant_state is None:
        return W
    if type(quant_state) is not list:
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2

    if out is None:
        out = torch.empty(shape, dtype=dtype, device="cpu")
    else:
        assert(out.shape == shape)
        assert(out.dtype == dtype)

    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(n_elements_absmax, dtype=torch.float32, device="cpu")

    # Assume appropriate `get_ptr` function is defined elsewhere
    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), get_ptr(absmax), get_ptr(out),
       ctypes.c_int(blocksize), ctypes.c_int(out.numel()))

    is_transposed = (True if W.shape[0] == 1 else False)
    return out.t() if is_transposed else out

def fast_gemv(X, W, quant_state, out=None):
    if quant_state is None:
        return torch.matmul(X, W, out=out)

    _, q_len, hd = X.shape

    if type(quant_state) is not list:
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        stats = quant_state.code
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2

    bout = shape[0]

    if out is None:
        out = torch.empty((1, 1, bout,), dtype=dtype, device="cpu")

    n = 1
    m = shape[0]
    k = shape[1]
    lda = shape[0]
    ldc = shape[0]
    ldb = (hd + 1) // 2

    df = torch.empty(absmax.shape, dtype=torch.float32, device="cpu")
    fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else cdequantize_blockwise_bf16_nf4
    fx(get_ptr(None), get_ptr(W), get_ptr(absmax), get_ptr(df),
       ctypes.c_int(blocksize2), ctypes.c_int(df.numel()))
    df += offset
    absmax = df

    fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else cgemm_4bit_inference_naive_bf16

    blocksize = ctypes.c_int32(blocksize)
    fx(m, n, k, get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(stats), get_ptr(out),
       lda, ldb, ldc, blocksize)

    return out

def fast_linear_forward(proj, X, temp_lora=None, out=None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch.matmul(X, W.t(), out=out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out=out)
    else:
        W = fast_dequantize(W.t(), W_quant)
        out = torch.matmul(X, W, out=out)

    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch.mv(lora_A._fast_lora, X.ravel(), out=temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha=lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch.mm(X.view(bsz, in_dim), lora_A._fast_lora.t(), out=temp_lora)
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha=lora_S)
        out = out.view(bsz, 1, out_dim)

    if bias is not None:
        out += bias

    return out

def matmul_lora(X, W, W_quant, A, B, s, out=None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    out = torch.matmul(X, W, out=out)
    if W_quant is not None:
        del W

    if A is not None:
        A, B = A.t(), B.t()
        out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out
