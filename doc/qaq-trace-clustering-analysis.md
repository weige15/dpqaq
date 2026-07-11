# QAQ Trace Clustering Analysis

## Scope

This note prepares the real trace collection workflow for `scripts/collect_qaq_profile_traces.py` and analyzes existing real GPU trace records for prompt clustering by scalar bit budget and coarse block precision profile.

This is not a dynamic batching speedup claim. Simulation outputs are not used as evidence of speedup here.

## Real Trace Collection Workflow

Use the lab GPU server, not the local RTX 4050. Check GPU availability manually before launching the job, then run from the repository root:

```bash
cd /nfs/home/s314511048/dpqaq
OUT=artifacts/qaq_mixed_trace_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
cp artifacts/qaq_mixed_trace_20260709_050923/prompts.jsonl "$OUT/prompts.jsonl"
export CUDA_VISIBLE_DEVICES=0
export AP_MODEL_PATH='/nfs/home/s314511048/dpqaq/cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512'
export ROUTER_CHECKPOINT='/nfs/home/s314511048/dpqaq/checkpoints/qaq_router_llama31_8b_th005.pt'
export ESTIMATOR_RESULTS='/nfs/home/s314511048/dpqaq/estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam'
python scripts/collect_qaq_profile_traces.py \
  --ap_model_path "$AP_MODEL_PATH" \
  --router_checkpoint "$ROUTER_CHECKPOINT" \
  --estimator_results "$ESTIMATOR_RESULTS" \
  --bits 3 4 5 6 \
  --router_mode mlp_multibit_dp_guard \
  --prompt_file "$OUT/prompts.jsonl" \
  --max_requests 200 \
  --workload_type mixed \
  --max_new_tokens 8 \
  --device cuda \
  --output_jsonl "$OUT/qaq_trace.jsonl" \
  --summary_json "$OUT/qaq_trace_summary.json" \
  2>&1 | tee "$OUT/run.log"
```

The existing completed real trace used for this analysis is:

- `artifacts/qaq_mixed_trace_20260709_050923/qaq_trace.jsonl`
- `artifacts/qaq_mixed_trace_20260709_050923/qaq_trace_summary.json`
- `artifacts/qaq_mixed_trace_20260709_050923/qaq_trace_cluster_analysis.json`

## Trace Inputs

The analyzed trace has 200 real single-request QAQ generation records collected on an NVIDIA GeForce RTX 3090 with `router_mode=mlp_multibit_dp_guard`, candidate bits `3 4 5 6`, and `max_new_tokens=8`.

Workload counts:

| Workload | Requests |
| --- | ---: |
| chat | 50 |
| code | 50 |
| math | 50 |
| summarization | 50 |

Key caveats:

- `finite_logits`, quality metrics, under-precision labels, over-precision labels, transfer bytes, HBM bytes, and kernel-switch metrics remain `UNVALIDATED` in this trace.
- Summarization completions split between 1-token and 8-token outputs, so latency comparisons should condition on output length.
- This analysis describes precision-profile clustering only.

## Profile Definitions

Scalar bit budget means `average_selected_bit` from each real trace record.

Scalar buckets use floor bucketing at 0.25-bit or 0.10-bit width.

Coarse block precision profile is computed from `per_layer_bit_counts` by taking the expected bit per route, averaging routes over 4-layer groups, then rounding each group to 0.25-bit buckets. This yields an 8-value block profile for Llama 3.1 8B's 32 layers.

Majority profile is included only as a negative control: it takes the majority selected bit per route.

## Scalar Budget Clustering

At 0.25-bit granularity, scalar budgets form only three buckets:

| Scalar Bucket | Requests | Workload Mix | Dominant Purity |
| --- | ---: | --- | ---: |
| 5.50 | 56 | chat 29, code 16, math 11 | 51.8% |
| 5.75 | 119 | math 39, code 34, summarization 25, chat 21 | 32.8% |
| 6.00 | 25 | summarization 25 | 100.0% |

Scalar-budget clustering is therefore weak-to-moderate. It separates the highest-budget summarization subset cleanly, but the largest bucket mixes all workloads and does not distinguish chat, code, and math well.

At 0.10-bit granularity there are five scalar buckets and workload purity rises from 46.5% to 59.5%, but the 5.7 and 5.8 buckets still mix chat, code, and math.

Per-workload scalar means:

| Workload | Mean Avg Bit | Min | Max | 0.25-Bit Buckets |
| --- | ---: | ---: | ---: | --- |
| chat | 5.7465 | 5.6292 | 5.9273 | 5.50: 29, 5.75: 21 |
| code | 5.7921 | 5.6491 | 5.9004 | 5.50: 16, 5.75: 34 |
| math | 5.8037 | 5.6893 | 5.9459 | 5.50: 11, 5.75: 39 |
| summarization | 5.9879 | 5.9530 | 6.0000 | 5.75: 25, 6.00: 25 |

## Block Profile Clustering

Coarse block profiles show more structure than scalar buckets, but they are still not clean prompt-type clusters.

| Profile Type | Unique Profiles | Workload Purity |
| --- | ---: | ---: |
| scalar bucket, 0.25-bit | 3 | 46.5% |
| scalar bucket, 0.10-bit | 5 | 59.5% |
| coarse block profile | 18 | 63.0% |
| majority route profile | 1 | 25.0% |

The top coarse profiles are:

| Profile ID | Requests | Workload Mix | Scalar Buckets | Profile Vector |
| --- | ---: | --- | --- | --- |
| `4e22c8a6` | 73 | math 30, code 22, chat 21 | 5.50: 14, 5.75: 59 | `[5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75]` |
| `8930f68d` | 54 | summarization 50, chat 3, math 1 | 5.75: 29, 6.00: 25 | `[6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]` |
| `96e84645` | 31 | chat 16, code 9, math 6 | 5.50: 31 | `[5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.75, 5.5]` |

Block profiles add detail inside scalar buckets. The 5.75 scalar bucket alone contains 13 distinct coarse profiles, while the 5.50 bucket contains 6. The 6.00 bucket contains a single all-6.0 profile and is entirely summarization in this trace.

A simple nearest-workload-centroid check gives 58.5% assignment accuracy using scalar average bit and 59.5% using the coarse block vector. This is only a descriptive separability check, not a trained or cross-validated classifier.

## Interpretation

Prompts do cluster somewhat by scalar bit budget, but the clustering is mostly a high-budget summarization signal plus broad overlap among chat, code, and math.

Prompts show stronger clustering by coarse block precision profile than by 0.25-bit scalar budget, especially because the all-6.0 coarse profile captures all summarization prompts. However, block profiles still do not cleanly separate chat, code, and math: the largest profile mixes all three non-summarization workloads.

The majority route profile is too coarse for this trace because every request collapses to the same all-6 majority profile.

Prompt length is strongly correlated with scalar bit budget in this trace (`r=0.8286`), so some of the apparent summarization clustering may reflect long-context prompt length rather than workload semantics alone.

## Conclusion

Use scalar bit budget as a cheap first-stage grouping signal only if the goal is coarse high-budget versus lower-budget separation. Use coarse block precision profiles when profile structure matters, because they expose more fragmentation and reuse patterns than scalar budgets.

Do not claim a batching speedup from these results. The correct next validation is a real GPU replay or serving benchmark with synchronized timing, quality checks, and repeated runs.
