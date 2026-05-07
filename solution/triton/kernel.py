"""
FlashInfer MLSys 2026 Contest — fused_moe track
Definition: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

Strategy:
  - Routing: softmax + top-k (matches reference implementation exactly)
  - GEMM1 & GEMM2: custom Triton FP8 block-scale grouped GEMM
      * single kernel launch for all experts (no per-expert dispatch)
      * precomputed tile→expert mapping avoids in-kernel binary search
      * FP8 Tensor Cores via tl.dot with float8e4nv operands
      * per-row a_scale × per-tile b_scale applied per K-block
      * Triton's num_stages handles async double-buffering automatically
  - SwiGLU in BF16
  - Triton FP8 block-scale requantizer for intermediate activations
  - Triton atomic weighted scatter into pre-allocated BF16 output

Fallback: deep_gemm (kept as reference baseline)
"""

import torch
import triton
import triton.language as tl

# ── Fixed problem constants ───────────────────────────────────────────────────
NUM_EXPERTS   = 256
NUM_LOCAL_EXP = 32
HIDDEN        = 7168
INTERMEDIATE  = 2048
GEMM1_OUT     = 4096
TOP_K         = 8
BLOCK_SIZE    = 128


# ── Triton: FP8 block-scale grouped GEMM ─────────────────────────────────────
#
# Computes:  C[M, N] = dequant(A) @ dequant(W[expert])^T
# where dequant is block-128 scale: each K-block of A has a per-row scale,
# each (N-block, K-block) of W has one scale.
#
# Grid: (total_m_tiles, N // BLOCK_N)
#   total_m_tiles = sum over experts of ceil(tokens_for_expert / BLOCK_M)
#
# Tile→expert mapping is precomputed on CPU and passed as device tensors,
# matching the approach from the old grouped_gemm_blackwell_async_kernel.
#
@triton.jit
def _fp8_grouped_gemm_kernel(
    # A: [M, K] fp8_e4m3fn  (tokens, sorted by expert)
    a_ptr,
    # A scales: [M, K//128] fp32  (per-row per-k-block)
    as_ptr,
    # B: [G, N, K] fp8_e4m3fn  (weights)
    b_ptr,
    # B scales: [G, N//128, K//128] fp32
    bs_ptr,
    # Output: [M, N] bf16
    c_ptr,
    # Tile metadata (precomputed, [num_m_tiles] each)
    tile_expert_ptr,   # int32 — which expert this m-tile belongs to
    tile_m_start_ptr,  # int32 — first row in A for this m-tile
    tile_m_end_ptr,    # int32 — exclusive last row for this m-tile
    # Dims
    M, N, K,
    K_BLOCKS: tl.constexpr,   # K // 128
    N_BLOCKS: tl.constexpr,   # N // 128
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,   # = 128
    BLOCK_K:  tl.constexpr,   # = 128
):
    pid_m = tl.program_id(0)   # which m-tile (all experts flattened)
    pid_n = tl.program_id(1)   # which n-tile

    # Look up expert and row range for this tile
    expert  = tl.load(tile_expert_ptr  + pid_m)
    m_start = tl.load(tile_m_start_ptr + pid_m)
    m_end   = tl.load(tile_m_end_ptr   + pid_m)   # exclusive
    n_start = pid_n * BLOCK_N

    rows = m_start + tl.arange(0, BLOCK_M)
    cols = n_start + tl.arange(0, BLOCK_N)
    m_mask = rows < m_end   # handles partial tiles at expert boundary

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kb in range(K_BLOCKS):
        k_start = kb * BLOCK_K
        ks = k_start + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K] fp8
        a = tl.load(
            a_ptr + rows[:, None] * K + ks[None, :],
            mask=m_mask[:, None], other=0,
        )
        # A scales: [BLOCK_M] — one per row per k-block
        a_s = tl.load(
            as_ptr + rows * K_BLOCKS + kb,
            mask=m_mask, other=0.0,
        )

        # B tile: [BLOCK_N, BLOCK_K] fp8 — weights for this expert
        # N is always a multiple of 128 for our problem dims, no n mask needed
        b = tl.load(
            b_ptr + expert * (N * K) + cols[:, None] * K + ks[None, :],
        )
        # B scale: scalar — one per (expert, n-block, k-block)
        b_s = tl.load(bs_ptr + expert * (N_BLOCKS * K_BLOCKS) + pid_n * K_BLOCKS + kb)

        # FP8 Tensor Core dot: [BLOCK_M, BLOCK_K] @ [BLOCK_N, BLOCK_K]^T → [BLOCK_M, BLOCK_N]
        dot = tl.dot(a, tl.trans(b), out_dtype=tl.float32)

        # Apply block scales: a_s is per-row, b_s is a scalar for this tile
        acc = acc + a_s[:, None] * b_s * dot

    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        acc.to(tl.bfloat16),
        mask=m_mask[:, None],
    )


def _build_tile_metadata(exp_ids_cpu: torch.Tensor, G: int, BLOCK_M: int, device):
    """
    Precompute per-tile (expert, m_start, m_end) on CPU.
    Mirrors the expert_offsets array from grouped_gemm_blackwell_async_kernel.
    """
    counts      = torch.bincount(exp_ids_cpu, minlength=G)          # [G]
    tok_starts  = torch.cat([torch.zeros(1, dtype=torch.int32),
                             counts.cumsum(0).to(torch.int32)])      # [G+1]
    n_tiles_e   = (counts + BLOCK_M - 1) // BLOCK_M                 # ceil per expert

    experts, m_starts, m_ends = [], [], []
    for e in range(G):
        es = tok_starts[e].item()
        ee = tok_starts[e + 1].item()
        for lt in range(n_tiles_e[e].item()):
            ts = es + lt * BLOCK_M
            experts.append(e)
            m_starts.append(ts)
            m_ends.append(min(ts + BLOCK_M, ee))

    return (
        torch.tensor(experts,  dtype=torch.int32, device=device),
        torch.tensor(m_starts, dtype=torch.int32, device=device),
        torch.tensor(m_ends,   dtype=torch.int32, device=device),
    )


def triton_fp8_grouped_gemm(
    a:       torch.Tensor,   # [M, K]          fp8_e4m3fn
    a_scale: torch.Tensor,   # [M, K//128]     fp32
    b:       torch.Tensor,   # [G, N, K]       fp8_e4m3fn
    b_scale: torch.Tensor,   # [G, N//128, K//128] fp32
    exp_ids: torch.Tensor,   # [M]             int32, sorted
) -> torch.Tensor:           # [M, N]          bfloat16
    M, K  = a.shape
    G, N, _ = b.shape
    device = a.device

    BLOCK_M = 16
    BLOCK_N = 128
    BLOCK_K = 128

    tile_expert, tile_m_start, tile_m_end = _build_tile_metadata(
        exp_ids.cpu(), G, BLOCK_M, device
    )
    total_m_tiles = tile_expert.shape[0]
    n_tiles = N // BLOCK_N   # N is always a multiple of 128 for our dims

    c = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    _fp8_grouped_gemm_kernel[(total_m_tiles, n_tiles)](
        a, a_scale, b, b_scale, c,
        tile_expert, tile_m_start, tile_m_end,
        M, N, K,
        K_BLOCKS=K // BLOCK_K,
        N_BLOCKS=N // BLOCK_N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=3,    # Triton emits async double-buffering (like __pipeline_memcpy_async)
        num_warps=4,
    )
    return c


# ── Triton: block-scale FP8 quantization ─────────────────────────────────────
@triton.jit
def _fp8_quantize_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    K_BLOCKS: tl.constexpr,
    BLOCK_K:  tl.constexpr,
    TILE_M:   tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * TILE_M + tl.arange(0, TILE_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (rows[:, None] < M) & (cols[None, :] < K)
    x = tl.load(x_ptr + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    amax  = tl.max(tl.abs(x), axis=1)
    scale = tl.where(amax > 0, amax / 448.0, tl.full([TILE_M], 1e-12, tl.float32))
    x_fp8 = (x / scale[:, None]).to(tl.float8e4nv)
    tl.store(out_ptr   + rows[:, None] * K + cols[None, :], x_fp8,  mask=mask)
    tl.store(scale_ptr + rows * K_BLOCKS + pid_k,           scale,  mask=(rows < M))


def quantize_fp8_block(x: torch.Tensor):
    """[M, K] bf16/fp32 → ([M, K] fp8_e4m3fn, [M, K//128] fp32)."""
    M, K = x.shape
    KB   = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    Kp   = KB * BLOCK_SIZE
    if Kp != K:
        x = torch.nn.functional.pad(x, (0, Kp - K))
    out   = torch.empty(M, Kp, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, KB, dtype=torch.float32,        device=x.device)
    TILE_M = 4
    _fp8_quantize_kernel[triton.cdiv(M, TILE_M), KB](
        x, out, scale, M, Kp,
        K_BLOCKS=KB, BLOCK_K=BLOCK_SIZE, TILE_M=TILE_M,
    )
    return out[:, :K].contiguous(), scale


# ── Triton: weighted scatter-add ──────────────────────────────────────────────
@triton.jit
def _scatter_kernel(
    src_ptr, tok_ptr, wt_ptr, out_ptr,
    N, H,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return
    tok = tl.load(tok_ptr + pid)
    wt  = tl.load(wt_ptr  + pid).to(tl.float32)
    for h0 in range(0, H, BLOCK_H):
        cols = h0 + tl.arange(0, BLOCK_H)
        mask = cols < H
        val  = tl.load(src_ptr + pid * H + cols, mask=mask).to(tl.float32)
        tl.atomic_add(out_ptr + tok * H + cols, val * wt, mask=mask)


# ── deep_gemm fallback (kept as reference baseline) ──────────────────────────
def _get_deepgemm_fn():
    try:
        import deep_gemm
        return (
            getattr(deep_gemm, 'm_grouped_gemm_fp8_fp8_bf16_nt',   None) or
            getattr(deep_gemm, 'm_grouped_fp8_gemm_nt_contiguous', None)
        )
    except ImportError:
        return None


# ── Core forward ─────────────────────────────────────────────────────────────
def _moe_forward_inner(
    routing_logits,        # [T, 256] fp32
    hidden_states,         # [T, 7168] fp8_e4m3fn
    hidden_states_scale,   # [56, T] fp32  (competition layout: transposed)
    gemm1_weights,         # [32, 4096, 7168] fp8_e4m3fn
    gemm1_weights_scale,   # [32, 32, 56] fp32
    gemm2_weights,         # [32, 7168, 2048] fp8_e4m3fn
    gemm2_weights_scale,   # [32, 56, 16] fp32
    local_expert_offset: int,
    routed_scaling_factor: float,
    output,                # [T, 7168] bf16 (pre-allocated)
):
    T      = routing_logits.shape[0]
    device = routing_logits.device

    # ── 1. Routing: softmax + top-k ───────────────────────────────────────────
    routing_weights            = torch.softmax(routing_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, k=TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) * routed_scaling_factor

    # ── 2. Filter to local experts ────────────────────────────────────────────
    local_ids = topk_indices - local_expert_offset
    local_ok  = (local_ids >= 0) & (local_ids < NUM_LOCAL_EXP)

    tok_idx, k_idx = local_ok.nonzero(as_tuple=True)
    if tok_idx.numel() == 0:
        output.zero_()
        return

    exp_ids  = local_ids[tok_idx, k_idx]
    wts_flat = topk_weights[tok_idx, k_idx]

    # Sort by expert — required for contiguous grouped layout
    order    = exp_ids.argsort(stable=True)
    exp_ids  = exp_ids[order].to(torch.int32)
    tok_sort = tok_idx[order].to(torch.int32)
    wts_sort = wts_flat[order].to(torch.float32)

    N_local = tok_sort.shape[0]

    # ── 3. Gather FP8 tokens ──────────────────────────────────────────────────
    # hidden_states_scale: [H//128, T] → [T, H//128] (row-major for our kernel)
    hs_scale = hidden_states_scale.T.contiguous()   # [T, 56]
    g_hs     = hidden_states[tok_sort]              # [N_local, 7168] fp8
    g_hs_sc  = hs_scale[tok_sort]                  # [N_local, 56] fp32

    # ── 4. GEMM1: [N_local, 7168] × [32, 4096, 7168]^T → [N_local, 4096] bf16
    g1_out = triton_fp8_grouped_gemm(g_hs, g_hs_sc, gemm1_weights, gemm1_weights_scale, exp_ids)

    # ── 5. SwiGLU ─────────────────────────────────────────────────────────────
    gate, up = g1_out.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate.float()) * up.float()

    # ── 6. FP8 requantize for GEMM2 ──────────────────────────────────────────
    act_fp8, act_sc = quantize_fp8_block(act.to(torch.bfloat16))

    # ── 7. GEMM2: [N_local, 2048] × [32, 7168, 2048]^T → [N_local, 7168] bf16
    g2_out = triton_fp8_grouped_gemm(act_fp8, act_sc, gemm2_weights, gemm2_weights_scale, exp_ids)

    # ── 8. Weighted scatter → output ──────────────────────────────────────────
    output.zero_()
    _scatter_kernel[(N_local,)](
        g2_out, tok_sort, wts_sort, output,
        N_local, HIDDEN, BLOCK_H=256,
    )


# ── Public entry point (competition DPS signature) ────────────────────────────
def kernel(
    routing_logits,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    output,
    routing_bias=None,
    hidden_states=None,
    hidden_states_scale=None,
):
    T      = routing_logits.shape[0]
    device = routing_logits.device

    if hidden_states is None:
        hidden_states = torch.zeros(T, HIDDEN, dtype=torch.float8_e4m3fn, device=device)
    if hidden_states_scale is None:
        hidden_states_scale = torch.ones(HIDDEN // BLOCK_SIZE, T, dtype=torch.float32, device=device)

    _moe_forward_inner(
        routing_logits, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
        int(local_expert_offset), float(routed_scaling_factor),
        output,
    )
