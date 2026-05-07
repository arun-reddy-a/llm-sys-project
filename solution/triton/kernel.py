"""
FlashInfer MLSys 2026 Contest — fused_moe track
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Strategy:
  - DeepSeek-V3 routing (sigmoid + group-topk) in PyTorch
  - deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt for both GEMMs
  - Custom Triton kernel for block-scale FP8 quantization of the SwiGLU output
  - Weighted scatter directly into the BF16 output buffer

Kernel signature (DPS — output is pre-allocated):
  kernel(routing_logits, routing_bias,
         hidden_states, hidden_states_scale,
         gemm1_weights, gemm1_weights_scale,
         gemm2_weights, gemm2_weights_scale,
         local_expert_offset, routed_scaling_factor,
         output)
"""

import torch
import triton
import triton.language as tl

# ── Constants (fixed by the problem definition) ──────────────────────────────
NUM_EXPERTS     = 256
NUM_LOCAL_EXP   = 32
HIDDEN          = 7168
INTERMEDIATE    = 2048
GEMM1_OUT       = 4096   # 2 * INTERMEDIATE (gate + up)
TOP_K           = 8
N_GROUP         = 8
TOPK_GROUP      = 4
BLOCK_SIZE      = 128    # FP8 block-scale granularity
FP8_MAX         = 448.0  # float8_e4m3fn max representable value


# ── Triton: block-scale FP8 quantization ─────────────────────────────────────

@triton.jit
def _quantize_fp8_block_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,      # = BLOCK_SIZE = 128
    TILE_M: tl.constexpr,
):
    """Quantize [M, K] BF16 → FP8 E4M3 with per-(row, K-block) scales."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    row_start = pid_m * TILE_M
    k_start   = pid_k * BLOCK_K

    rows = row_start + tl.arange(0, TILE_M)
    cols = k_start   + tl.arange(0, BLOCK_K)

    mask = (rows[:, None] < M) & (cols[None, :] < K)
    x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)          # [TILE_M]
    scale = tl.where(amax > 0, amax / 448.0, tl.full([TILE_M], 1e-12, tl.float32))

    x_scaled = x / scale[:, None]
    x_fp8 = x_scaled.to(tl.float8e4nv)        # float8_e4m3fn

    tl.store(out_ptr   + rows[:, None] * K + cols[None, :], x_fp8,  mask=mask)
    tl.store(scale_ptr + rows * NUM_K_BLOCKS + pid_k, scale, mask=(rows < M))


def quantize_fp8_block(x: torch.Tensor) -> tuple:
    """[M, K] bf16 → ([M, K] fp8_e4m3fn, [M, K//128] fp32 scales)."""
    M, K = x.shape
    K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    K_pad    = K_blocks * BLOCK_SIZE

    if K_pad != K:
        x = torch.nn.functional.pad(x, (0, K_pad - K))

    out   = torch.empty(M, K_pad, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, K_blocks, dtype=torch.float32,    device=x.device)

    TILE_M = 4
    grid = (triton.cdiv(M, TILE_M), K_blocks)
    _quantize_fp8_block_kernel[grid](
        x, out, scale,
        M, K_pad,
        NUM_K_BLOCKS=K_blocks,
        BLOCK_K=BLOCK_SIZE,
        TILE_M=TILE_M,
    )
    return out[:, :K].contiguous(), scale


# ── Triton: weighted scatter-add into output ──────────────────────────────────

@triton.jit
def _weighted_scatter_kernel(
    src_ptr,        # [total_local, H] bf16
    token_ids_ptr,  # [total_local] int32
    weights_ptr,    # [total_local] fp32
    out_ptr,        # [T, H] bf16  (pre-zeroed by caller)
    total_local, H,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)   # one block per assigned token-expert pair
    if pid >= total_local:
        return

    token_id = tl.load(token_ids_ptr + pid)
    weight   = tl.load(weights_ptr   + pid).to(tl.float32)

    for h_start in range(0, H, BLOCK_H):
        cols = h_start + tl.arange(0, BLOCK_H)
        mask = cols < H
        val  = tl.load(src_ptr + pid * H + cols, mask=mask).to(tl.float32)
        tl.atomic_add(out_ptr + token_id * H + cols, val * weight, mask=mask)


# ── Main competition kernel ───────────────────────────────────────────────────

def kernel(
    routing_logits:       torch.Tensor,   # [T, 256]  fp32
    routing_bias:         torch.Tensor,   # [256]     fp32
    hidden_states:        torch.Tensor,   # [T, H]    fp8_e4m3fn
    hidden_states_scale:  torch.Tensor,   # [H//128, T]  fp32  (note: transposed!)
    gemm1_weights:        torch.Tensor,   # [E_local, 4096, H]  fp8_e4m3fn
    gemm1_weights_scale:  torch.Tensor,   # [E_local, 32, 56]   fp32
    gemm2_weights:        torch.Tensor,   # [E_local, H, 2048]  fp8_e4m3fn
    gemm2_weights_scale:  torch.Tensor,   # [E_local, 56, 16]   fp32
    local_expert_offset:  int,
    routed_scaling_factor: float,
    output:               torch.Tensor,   # [T, H]    bf16  (preallocated)
):
    import deep_gemm

    T = routing_logits.shape[0]
    device = routing_logits.device

    # ── 1. DeepSeek routing ────────────────────────────────────────────────────
    scores = torch.sigmoid(routing_logits + routing_bias)  # [T, 256]

    # Group pruning: select top-TOPK_GROUP groups of N_GROUP each
    scores_grouped = scores.view(T, N_GROUP, NUM_EXPERTS // N_GROUP)      # [T, 8, 32]
    group_scores   = scores_grouped.topk(2, dim=-1).values.sum(dim=-1)    # [T, 8]  (top-2 sum)
    group_sel      = group_scores.topk(TOPK_GROUP, dim=-1).indices        # [T, 4]

    group_mask = torch.zeros(T, N_GROUP, dtype=torch.bool, device=device)
    group_mask.scatter_(1, group_sel, True)                               # [T, 8]
    expert_mask = group_mask.unsqueeze(-1).expand_as(scores_grouped).reshape(T, NUM_EXPERTS)

    masked_scores = scores.masked_fill(~expert_mask, float('-inf'))
    topk_weights, topk_indices = masked_scores.topk(TOP_K, dim=-1)       # [T, K] each

    # Normalize & scale
    topk_weights = torch.softmax(topk_weights, dim=-1) * routed_scaling_factor  # [T, K] fp32

    # ── 2. Filter to local experts only ───────────────────────────────────────
    local_ids = topk_indices - local_expert_offset                        # [T, K]
    local_ok  = (local_ids >= 0) & (local_ids < NUM_LOCAL_EXP)           # [T, K]

    tok_idx, k_idx   = local_ok.nonzero(as_tuple=True)
    exp_ids_sorted_i = local_ids[tok_idx, k_idx]
    weights_flat     = topk_weights[tok_idx, k_idx]

    if tok_idx.numel() == 0:
        output.zero_()
        return

    # Sort by expert id (required by deep_gemm m_grouped API)
    order           = exp_ids_sorted_i.argsort(stable=True)
    exp_ids_sorted  = exp_ids_sorted_i[order].to(torch.int32)
    token_ids_sorted = tok_idx[order].to(torch.int32)
    weights_sorted  = weights_flat[order]

    total_local = token_ids_sorted.shape[0]

    # ── 3. Gather token inputs ────────────────────────────────────────────────
    # hidden_states_scale: [H//128, T] → deep_gemm wants [total_local, H//128]
    hs_scale_t = hidden_states_scale.T.contiguous()      # [T, H//128]
    grouped_hs       = hidden_states[token_ids_sorted]   # [total_local, H]  fp8
    grouped_hs_scale = hs_scale_t[token_ids_sorted]      # [total_local, 56] fp32

    # ── 4. GEMM1: [total_local, H] × [E_local, 4096, H]^T → [total_local, 4096] bf16
    gemm1_out = torch.empty(total_local, GEMM1_OUT, dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
        (grouped_hs, grouped_hs_scale),
        (gemm1_weights, gemm1_weights_scale),
        gemm1_out,
        exp_ids_sorted,
    )

    # ── 5. SwiGLU activation ──────────────────────────────────────────────────
    gate, up = gemm1_out.chunk(2, dim=-1)                    # each [total_local, 2048]
    activated = torch.nn.functional.silu(gate.float()) * up.float()  # [total_local, 2048] fp32

    # ── 6. Quantize activated BF16 → FP8 for GEMM2 ───────────────────────────
    act_fp8, act_scale = quantize_fp8_block(activated.to(torch.bfloat16))
    # act_fp8:   [total_local, 2048] fp8_e4m3fn
    # act_scale: [total_local, 16]   fp32

    # ── 7. GEMM2: [total_local, 2048] × [E_local, H, 2048]^T → [total_local, H] bf16
    gemm2_out = torch.empty(total_local, HIDDEN, dtype=torch.bfloat16, device=device)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
        (act_fp8, act_scale),
        (gemm2_weights, gemm2_weights_scale),
        gemm2_out,
        exp_ids_sorted,
    )

    # ── 8. Weighted scatter into output ───────────────────────────────────────
    output.zero_()
    BLOCK_H = 256
    _weighted_scatter_kernel[(total_local,)](
        gemm2_out, token_ids_sorted, weights_sorted.to(torch.float32), output,
        total_local, HIDDEN,
        BLOCK_H=BLOCK_H,
    )
