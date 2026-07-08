# QAQ Router Design

This document describes the intended QAQ path in this repository. Items marked
`UNVALIDATED` exist in code or design but still need real checkpoint/model
validation before they can support paper claims.

## Entry Points

- Router module: `any_precision/modules/QAQRouter.py`
- Runtime linear: `any_precision/modules/QAQDPLLM_Linear.py`
- Model wrapper: `any_precision/modules/QAQDPLLMForCausalLM.py`
- Training script: `scripts/train_qaq_router.py`
- Inference sanity script: `scripts/run_qaq_inference.py`

## Router Inputs

For each routed linear row, `QAQRouter.forward` consumes:

- `x`: captured or runtime linear input activation, flattened over batch/token rows.
- `route_id`: embedding id for the decoder-layer and linear-module pair.
- `log1p(||x||)`: optional norm scalar when `use_norm_feature=True`.
- `estimated_error`: optional DP-LLM-style scalar when `use_estimated_error=True`.

Training uses a single `input_feature_dim` equal to the largest routed linear
input width and pads smaller linear inputs with zeros. `UNVALIDATED`: the effect
of this padding on router quality has not been measured.

## Label Definition

Labels must come from real captured activations, not random or synthetic labels.
For each candidate bit `b` and input row `x`, training compares dequantized
Any-Precision output against a reference bit:

```text
rel_error_b = ||W_ref x - W_b x|| / (||W_ref x|| + eps)
```

In `multibit` mode, the label is the smallest candidate bit whose relative error
is at or below `error_threshold`. If no lower candidate is safe, the label falls
back to the reference bit. In `binary` mode, exactly two bits are allowed; the
low bit is selected when safe, otherwise the high/reference bit is selected.

The intended reference bit is the highest candidate bit. `UNVALIDATED`: non-max
`--reference_bit` behavior is not suitable for paper claims until tested.

## Checkpoint Schema

`save_qaq_router_checkpoint` writes:

- `format`: expected to be `qaq_router_v1`
- `router_config`: hidden size, input feature dim, route count, bits, MLP shape,
  layer embedding dim, norm feature flag, estimated-error flag, dropout
- `router_state_dict`
- `candidate_bits`
- `hidden_size`
- `input_feature_dim`
- `num_layers`
- `training_config`
- `label_mode`
- `error_threshold`
- `target_bits`
- `route_map`
- `stats`

`UNVALIDATED`: load-time route-map identity checking is not currently proven.
Paper runs must verify the saved route map exactly matches the inference model's
layer/module route order.

## Runtime Modes

Implemented runtime modes:

- `fixed_low`: always use the lowest valid precision.
- `fixed_high`: always use the highest valid precision.
- `mlp_binary`: use router logits over two candidate bits.
- `mlp_multibit`: use router logits over all candidate bits.

Runtime behavior:

- Prefill uses max valid precision by default unless `prefill_by_router=True`.
- Decode routing flattens rows, predicts bits, and groups rows by selected bit.
- `confidence_threshold` can bump low-confidence predictions upward by
  `fallback_bits`.
- `max_mem_dict` can clamp valid bits per linear.

`UNVALIDATED`: generation statistics must be collected without extra forward
passes contaminating router counters.

## DP-LLM-Inspired Pieces

This QAQ path can reuse DP-LLM artifacts:

- `linear_reg_d.pt`: norm-based estimated-error parameters.
- `jl_d.pt`: Johnson-Lindenstrauss estimated-error projection matrices.
- `max_mem_dict.pt`: per-linear maximum valid precision.
- `T_d.pt`: DP-LLM threshold tuples for the DP baseline.

Implemented QAQ integration:

- Optional estimated-error feature via `--include_estimated_error`.
- Optional estimator loading in `QAQDPLLMForCausalLM.from_quantized`.
- DP-threshold baseline through `DPLLMForCausalLM` in
  `scripts/run_qaq_inference.py`.

`UNVALIDATED / NOT IMPLEMENTED`: a true `mlp_multibit_dp_guard` mode that applies
`T_d.pt` as a guard on QAQ predictions and reports separate guard counts.

## Baselines

Minimum baselines for QAQ claims:

- Static `fixed_low`.
- Static `fixed_high`.
- Static fixed candidate bits, if evaluating quality/bit tradeoffs.
- DP-LLM `dp_threshold` using `T_d.pt`.
- QAQ `mlp_multibit`.
- QAQ with estimated-error feature, only after validation.
- QAQ plus DP guard, only after implementation and validation.
