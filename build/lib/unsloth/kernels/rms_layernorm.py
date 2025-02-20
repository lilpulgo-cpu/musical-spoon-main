# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def rms_layernorm_forward(Y, X, W, r, n_cols, eps):
    """
    Fast RMS Layernorm kernel without Triton and CUDA.
    """
    X_row = X.to(torch.float32)
    W_row = W

    row_var = torch.sum(X_row * X_row) / n_cols
    inv_var = torch.rsqrt(row_var + eps)
    r.copy_(inv_var)
    normed = X_row * inv_var
    output = normed * W_row
    Y.copy_(output)
    return Y

def rms_layernorm_backward(dY, X, W, r, n_cols, eps, GEMMA):
    """
    Fast RMS Layernorm kernel for the backward pass without Triton and CUDA.
    """
    dY_row = dY.to(torch.float32)
    X_row = X.to(torch.float32)
    W_row = W.to(torch.float32)

    inv_var = r.to(torch.float32)
    normed = X_row * inv_var

    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row

    rowsum_dY_normed = torch.sum(dY_W * normed)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    dY.copy_(output)
    return dY

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        shape = X.shape
        dim = shape[-1]
        X = X.view(-1, dim)
        n_rows, n_cols = X.shape

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype)
        r = torch.empty(n_rows, dtype=torch.float32)

        rms_layernorm_forward(Y, X, W, r, n_cols, eps)
        ctx.eps = eps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        shape = dY.shape
        dim = shape[-1]
        dY = dY.view(-1, dim)
        X, W, r = ctx.saved_tensors
        n_rows, n_cols = dY.shape

        rms_layernorm_backward(dY, X, W, r, n_cols, ctx.eps, ctx.GEMMA)
        dX = dY.view(*shape)
        return dX, None, None, None

def fast_rms_layernorm(layernorm, X, gemma=False):
    W = layernorm.weight
    eps = layernorm.variance_epsilon
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out
