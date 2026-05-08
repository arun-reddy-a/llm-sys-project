#!/usr/bin/env python3
"""
Compare Triton FP8 grouped GEMM kernel against a pure-PyTorch reference.
The deep_gemm reference is broken on CUDA 12.8/B200 (NVCC < 12.9 bug),
so we compute ground truth in FP32 PyTorch instead.
"""
import sys
import torch

sys.path.insert(0, ".")
from solution.triton.kernel import kernel as triton_kernel

NUM_EXPERTS   = 256
NUM_LOCAL_EXP = 32
HIDDEN        = 7168
INTERMEDIATE  = 2048
GEMM1_OUT     = 4096
TOP_K         = 8
BLOCK_SIZE    = 128
LOCAL_OFFSET  = 0
ROUTED_SCALE  = 2.5

ATOL = 1.0
RTOL = 0.3
MATCH_FRAC = 0.9


def make_inputs(T, device="cuda"):
    torch.manual_seed(42)
    rl  = torch.randn(T, NUM_EXPERTS, dtype=torch.float32, device=device)
    # Small weights so intermediate values stay in BF16 range
    w1  = (torch.randn(NUM_LOCAL_EXP, GEMM1_OUT, HIDDEN, dtype=torch.float32, device=device) * 0.01).to(torch.float8_e4m3fn)
    w1s = torch.ones(NUM_LOCAL_EXP, GEMM1_OUT // BLOCK_SIZE, HIDDEN // BLOCK_SIZE, dtype=torch.float32, device=device)
    w2  = (torch.randn(NUM_LOCAL_EXP, HIDDEN, INTERMEDIATE, dtype=torch.float32, device=device) * 0.01).to(torch.float8_e4m3fn)
    w2s = torch.ones(NUM_LOCAL_EXP, HIDDEN // BLOCK_SIZE, INTERMEDIATE // BLOCK_SIZE, dtype=torch.float32, device=device)
    hs  = (torch.randn(T, HIDDEN, dtype=torch.float32, device=device) * 0.1).to(torch.float8_e4m3fn)
    hss = torch.ones(HIDDEN // BLOCK_SIZE, T, dtype=torch.float32, device=device)
    return rl, w1, w1s, w2, w2s, hs, hss


def _dequant(fp8_tensor, scale, block_size=128):
    """Dequantize a block-scale FP8 tensor to float32.

    fp8_tensor: [E, N, K] or [M, K]  fp8_e4m3fn
    scale:      [E, N//128, K//128] or [M, K//128]  fp32
    Returns:    same shape as fp8_tensor, fp32
    """
    x = fp8_tensor.float()
    s = scale
    # expand each scale dim by block_size to match the tensor dims
    for d in range(s.dim()):
        if s.shape[d] != x.shape[d]:
            s = s.repeat_interleave(block_size, dim=d)
    return x * s


def reference_kernel(rl, w1, w1s, w2, w2s, hs, hss):
    """Pure-PyTorch reference: exact FP32 computation with FP8 dequantization."""
    T      = rl.shape[0]
    device = rl.device

    # Routing
    routing_weights            = torch.softmax(rl, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, k=TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) * ROUTED_SCALE

    local_ids = topk_indices - LOCAL_OFFSET
    local_ok  = (local_ids >= 0) & (local_ids < NUM_LOCAL_EXP)
    tok_idx, k_idx = local_ok.nonzero(as_tuple=True)
    if tok_idx.numel() == 0:
        return torch.zeros(T, HIDDEN, dtype=torch.bfloat16, device=device)

    exp_ids  = local_ids[tok_idx, k_idx]
    wts_flat = topk_weights[tok_idx, k_idx]
    order    = exp_ids.argsort(stable=True)
    exp_ids  = exp_ids[order]
    tok_sort = tok_idx[order]
    wts_sort = wts_flat[order]
    N_local  = tok_sort.shape[0]

    # Dequantize hidden states: hs [T,H] fp8, hss [H//128, T] → [T, H//128]
    hss_t = hss.T.contiguous()                # [T, 56]
    hs_dq = _dequant(hs, hss_t)               # [T, H] fp32
    g_hs  = hs_dq[tok_sort]                   # [N_local, H]

    # Dequantize w1: [32, 4096, 7168] fp8, w1s [32, 32, 56]
    w1_dq = _dequant(w1.view(NUM_LOCAL_EXP, GEMM1_OUT, HIDDEN),
                     w1s.view(NUM_LOCAL_EXP, GEMM1_OUT // BLOCK_SIZE, HIDDEN // BLOCK_SIZE))

    # GEMM1
    g1_out = torch.zeros(N_local, GEMM1_OUT, dtype=torch.float32, device=device)
    for i in range(N_local):
        e = exp_ids[i].item()
        g1_out[i] = g_hs[i] @ w1_dq[e].T   # [H] @ [4096, H].T → [4096]

    # SwiGLU
    gate, up = g1_out.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate) * up   # [N_local, 2048] fp32

    # Dequantize w2: [32, 7168, 2048] fp8, w2s [32, 56, 16]
    w2_dq = _dequant(w2.view(NUM_LOCAL_EXP, HIDDEN, INTERMEDIATE),
                     w2s.view(NUM_LOCAL_EXP, HIDDEN // BLOCK_SIZE, INTERMEDIATE // BLOCK_SIZE))

    # GEMM2
    g2_out = torch.zeros(N_local, HIDDEN, dtype=torch.float32, device=device)
    for i in range(N_local):
        e = exp_ids[i].item()
        g2_out[i] = act[i] @ w2_dq[e].T   # [2048] @ [7168, 2048].T → [7168]

    # Weighted scatter
    out = torch.zeros(T, HIDDEN, dtype=torch.float32, device=device)
    for i in range(N_local):
        out[tok_sort[i]] += g2_out[i] * wts_sort[i]

    return out.to(torch.bfloat16)


def check(T):
    device = "cuda"
    rl, w1, w1s, w2, w2s, hs, hss = make_inputs(T, device)

    # Triton kernel
    out_triton = torch.zeros(T, HIDDEN, dtype=torch.bfloat16, device=device)
    triton_kernel(rl, w1, w1s, w2, w2s, LOCAL_OFFSET, ROUTED_SCALE, out_triton,
                  hidden_states=hs, hidden_states_scale=hss)

    # PyTorch reference
    out_ref = reference_kernel(rl, w1, w1s, w2, w2s, hs, hss)

    diff    = (out_triton.float() - out_ref.float()).abs()
    ref_abs = out_ref.float().abs()
    max_abs = diff.max().item()
    max_rel = (diff / (ref_abs + 1e-6)).max().item()
    match   = ((diff <= ATOL + RTOL * ref_abs).float().mean()).item()

    status = "PASS" if match >= MATCH_FRAC else "FAIL"
    print(f"  T={T:<6}  max_abs={max_abs:.2e}  max_rel={max_rel:.2e}  "
          f"match={match*100:.1f}%  [{status}]")
    return status == "PASS"


def main():
    print(f"\n=== Triton vs PyTorch reference (atol={ATOL}, rtol={RTOL}) ===\n")
    seq_lens = [16, 52, 80, 128, 256, 512, 901, 2048, 4096]
    results  = [check(T) for T in seq_lens]
    all_pass = all(results)
    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
