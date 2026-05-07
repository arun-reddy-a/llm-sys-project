<!-- Paste the FlashInfer benchmark page content here -->
FlashInfer-Bench Kernel Details
Kernel Name: moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048

---
### Axes
- seq_len: var
- num_experts: 256
- num_local_experts: 32
- hidden_size: 7168
- intermediate_size: 2048
- gemm1_out_size: 4096
- num_hidden_blocks: 56
- num_intermediate_blocks: 16
- num_gemm1_out_blocks: 32

---
### Signature
#### Inputs
- routing_logits: float32 [seq_len, num_experts]
- gemm1_weights: float8_e4m3fn [num_local_experts, gemm1_out_size, hidden_size]
- gemm1_weights_scale: float32 [num_local_experts, num_gemm1_out_blocks, num_hidden_blocks]
- gemm2_weights: float8_e4m3fn [num_local_experts, hidden_size, intermediate_size]
- gemm2_weights_scale: float32 [num_local_experts, num_hidden_blocks, num_intermediate_blocks]
- local_expert_offset: int32 Scalar
- routed_scaling_factor: float32 Scalar

---
### Reference Implementation Logic
```python
import torch

def reference_logic(
    routing_logits,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    seq_len, num_experts = routing_logits.shape
    num_local_experts, gemm1_out_size, hidden_size = gemm1_weights.shape
    
    # Routing logic (top-k=8)
    routing_weights = torch.softmax(routing_logits, dim=-1)
    topk_weights, topk_indices = torch.topk(routing_weights, k=8, dim=-1)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    topk_weights *= routed_scaling_factor

    output = torch.zeros((seq_len, hidden_size), dtype=torch.bfloat16, device=routing_logits.device)
    
    for le in range(num_local_experts):
        expert_idx = le + local_expert_offset
        # Find which tokens are routed to this expert
        mask = (topk_indices == expert_idx)
        if not mask.any():
            continue
        
        token_indices, k_indices = torch.where(mask)
        expert_inputs = hidden_states[token_indices] # hidden_states was the input, typically
        
        # GEMM 1 with block scaling
        # ... (logical flow of the block-scaled FP8 GEMMs)
    
    return output
```

---
### Performance Results (gpt-5-2025-08-07_cuda_a2d8ca)
| Workload (seq_len) | Baseline Perf (ms) | This Solution (ms) | Max Abs Err | Max Rel Err | Status |
|---|---|---|---|---|---|
| 901 | 0.699 | 16.223 | 6.10e-05 | 1.90e-01 | PASSED |
| 16 | 0.138 | 4.994 | 1.91e-06 | 6.71e-03 | PASSED |
| 15 | 0.091 | 2.361 | 7.63e-06 | 7.30e-03 | PASSED |
| 14 | 0.132 | 4.579 | 7.63e-06 | 7.25e-03 | PASSED |
| 14107 | 6.359 | 40.263 | 1.22e-04 | 3.51e-02 | PASSED |
| 11948 | 4.985 | 31.187 | 6.10e-05 | 6.90e-02 | PASSED |
| 62 | 0.200 | 7.240 | 1.53e-05 | 1.33e-02 | PASSED |
| 59 | 0.202 | 7.271 | 7.63e-06 | 1.26e-02 | PASSED |
| 58 | 0.242 | 10.710 | 1.53e-05 | 1.15e-02 | PASSED |
| 57 | 0.237 | 9.720 | 7.63e-06 | 1.55e-02 | PASSED |
| 56 | 0.240 | 10.362 | 1.53e-05 | 1.12e-02 | PASSED |
| 55 | 0.233 | 9.497 | 1.91e-06 | 7.59e-03 | PASSED |
| 54 | 0.218 | 8.797 | 1.53e-05 | 1.15e-02 | PASSED |
| 53 | 0.230 | 9.286 | 3.81e-06 | 7.44e-03 | PASSED |
| 52 | 0.179 | 6.398 | 1.91e-06 | 7.67e-03 | PASSED |
| 7 | 0.095 | 3.086 | ... | ... | PASSED |
| 1 | 0.063 | 1.506 | ... | ... | PASSED |
| 7 | 0.095 | 3.086 | 7.63e-06 | 1.04e-02 | PASSED |
| 1 | 0.063 | 1.506 | 0.00e+00 | 0.00e+00 | PASSED |
| 32 | 0.201 | 8.269 | 1.53e-05 | 1.15e-02 | PASSED |
| 80 | 0.269 | 11.988 | 1.53e-05 | 1.45e-02 | PASSED |

---
*End of Extraction*

| 32 | 0.201 | 8.269 | ... | ... | PASSED |
| 80 | 0.269 | 11.988 | ... | ... | PASSED |

#### Outputs
- output: bfloat16 [seq_len, hidden_size]


