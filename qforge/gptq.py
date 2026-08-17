"""GPTQ layer-wise INT4 quantization (2nd-order error compensation).

Implements the classic GPTQ algorithm (Frantar et al., 2210.17323) for the
W4A16 weight-only setting, matching the qforge/qbin INT4 contract:

  * symmetric INT4, range [-7, 7], level = max|w_group| / 7;
  * group axis = the K (input) dimension, per-(output-row, input-group)
    scales — the `[N, K//group]` BF16 layout the executor's `_gemm_dequant`
    consumes (identical to `qforge.quant.quantize_weight_int4` output shape);
  * group sizes g in {64, 128} (the decision's g64/g128).

The difference vs plain RTN (`quantize_weight_int4`) is the 2nd-order
compensation: weights are quantized column-by-column along the input dim and
the quantization error of column i is propagated to the remaining columns via
the inverse Hessian `(X^T X + lambda I)^{-1}` of the calibration activations
X.  Runs on GPU (fp32 internally); the returned scales are fp32 and must be
rounded to BF16 by the caller before dequant.

Note: blocksize is an exact reformulation of the sequential algorithm (the
within-block update uses the same `H_inv[i1:i2, i1:i2]` entries, and the
cross-block update uses `H_inv[i1:i2, i2:]`), not an approximation.
"""
from __future__ import annotations

import torch

INT4_LEVEL = 7.0


def inverse_hessian(X: torch.Tensor, damp: float = 0.01) -> torch.Tensor:
    """`(X^T X + lambda * mean(diag) * I)^{-1}` via Cholesky.

    X: [n, K] fp32 calibration activations.  Returns [K, K] fp32 inverse.
    """
    n, K = X.shape
    H = X.t() @ X                       # [K, K]
    diag_mean = H.diag().mean()
    H = H + damp * diag_mean * torch.eye(K, device=X.device, dtype=torch.float32)
    L = torch.linalg.cholesky(H)
    return torch.cholesky_inverse(L)


def gptq_quantize(W: torch.Tensor, X: torch.Tensor, group: int,
                  blocksize: int = 128, damp: float = 0.01
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ-quantize W to symmetric INT4 with per-(N, K//group) scales.

    W: [N, K] fp32 weight.  X: [n, K] fp32 calibration activations.
    Returns (Q [N, K] fp32 dequantized weight, scales [N, K//group] fp32).
    The INT4 indices are `round(Q / scale)` clipped to [-7, 7].
    """
    N, K = W.shape
    assert K % group == 0, f"K={K} not divisible by group={group}"
    G = K // group
    dev = W.device
    H_inv = inverse_hessian(X, damp=damp)

    W = W.clone()
    Q = torch.zeros_like(W)
    scales = torch.zeros((N, G), device=dev, dtype=torch.float32)

    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Hinv1 = H_inv[i1:i2, i1:i2].contiguous()
        Err1 = torch.zeros_like(W1)
        scale = None
        for j in range(count):
            i_abs = i1 + j
            if j % group == 0:
                g_idx = i_abs // group
                jg = min(j + group, count)
                wg = W1[:, j:jg]
                scale = wg.abs().max(dim=1).values / INT4_LEVEL   # [N]
                scales[:, g_idx] = scale
            w = W1[:, j]
            q = (w / scale).round().clamp(-INT4_LEVEL, INT4_LEVEL) * scale
            Q[:, i_abs] = q
            e = (w - q) / Hinv1[j, j]                             # [N]
            Err1[:, j] = e
            if j + 1 < count:
                W1[:, j + 1:] -= torch.outer(e, Hinv1[j, j + 1:])
        if i2 < K:
            W[:, i2:] -= Err1 @ H_inv[i1:i2, i2:]
    return Q, scales

def dequant_to_bf16(Q: torch.Tensor, scales: torch.Tensor, group: int
                    ) -> torch.Tensor:
    """Round dequantized W4A16 weight to BF16 (the executor storage dtype).

    Q: [N, K] fp32 (int4 * fp32 scale).  scales: [N, K//group] fp32.
    Returns [N, K] bf16 = bf16(int4 * bf16(scale)), matching hardware dequant.
    """
    N, K = Q.shape
    G = K // group
    s_bf = scales.to(torch.bfloat16).to(torch.float32)            # BF16 scale
    Qr = Q.reshape(N, G, group)
    wqi = (Qr / s_bf[:, :, None]).round().clamp(-INT4_LEVEL, INT4_LEVEL)
    return (wqi * s_bf[:, :, None]).reshape(N, K).to(torch.bfloat16)

