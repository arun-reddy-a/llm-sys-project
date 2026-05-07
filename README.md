# FlashInfer MLSys 2026 — fused_moe Submission

Competition: [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)
Track: **fused_moe** — `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

## Problem

FP8 block-scale MoE with DeepSeek-V3 routing. Two grouped GEMMs (gate+up, then down), SwiGLU, weighted scatter into BF16 output.

| Parameter | Value |
|-----------|-------|
| Total experts | 256 |
| Local experts | 32 |
| Top-K | 8 |
| Groups / selected | 8 / 4 |
| Hidden dim | 7168 |
| Intermediate dim | 2048 |
| Block scale size | 128 |
| Output dtype | bfloat16 |

## Approach

1. **Routing** — DeepSeek sigmoid + group top-4 + top-8 selection
2. **GEMM1** — `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt`
3. **SwiGLU** — fused in FP32
4. **FP8 requant** — custom Triton block-scale quantizer
5. **GEMM2** — `deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt`
6. **Scatter** — Triton atomic weighted scatter into BF16 output

## Run

```bash
python scripts/run_local.py --compare-baseline   # local GPU
python scripts/run_modal.py                       # B200 via Modal
```
