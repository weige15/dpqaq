# Precision-Aware Dynamic Batching Design

This document is the canonical research-facing design for precision-aware
dynamic batching in `dpqaq`. The primary topic is dynamic batching; mixed
precision is the control and resource dimension that the batching policy must
exploit. QAQ-style query-adaptive routing is the current precision mechanism,
not the complete research objective.

This is a pre-implementation contract: no batching speedup, quality preservation,
or throughput claim is valid until the validation gates below pass on real traces
and GPU-server measurements.

## Goal

Design and evaluate a precision-aware dynamic batching method for LLM inference
that jointly considers request batching and mixed-precision allocation to improve
throughput, latency, and memory efficiency while preserving model quality.

The central research question is:

> Can dynamically grouping requests with compatible precision profiles improve
> serving efficiency over ordinary dynamic batching, without unacceptable quality
> loss, fallback frequency, or deadline violations?

The working hypothesis is deliberately testable rather than assumed: if requests
with similar precision demand are grouped into compatible batches, the runtime
may reduce precision-profile conflicts, unnecessary high-bit execution, and
quality-risky under-precision decisions. The experiments must also test the
opposite outcome—that profile fragmentation, router overhead, or queueing delay
can erase the potential benefit.

The first implementation must answer four questions:

- Do real prompts produce reusable precision profiles?
- Does profile-aware grouping improve the latency/throughput/quality tradeoff
  over ordinary dynamic batching on the same real request trace?
- Does that improvement remain after queueing delay is included?
- Are fallback and under-precision events rare enough to preserve quality?

The comparison must distinguish the batching contribution from the precision
contribution. At minimum, the evaluation needs ordinary dynamic batching with
fixed precision, QAQ routing without profile-aware grouping, and the proposed
precision-aware batching policy.

## Non-Goals

- Do not replace the existing QAQ router or quantized linear path.
- Do not treat QAQ routing alone as the batching contribution or assume that
  lower effective bits imply higher serving throughput.
- Do not implement a full production serving engine before trace and simulation
  evidence justifies it.
- Do not claim performance from simulator results.
- Do not use random labels, fake profiles, mock routing outputs, or synthetic-only
  traces as completion evidence.
- Do not report CUDA latency or throughput without synchronization.

## Current Integration Points

Existing code already provides the low-level execution hooks:

- `QAQDPLLM_Linear` chooses per-row bits for `fixed_low`, `fixed_high`,
  `mlp_multibit`, `dp_threshold_only`, and `mlp_multibit_dp_guard`.
- `QAQDPLLM_Linear.batch_policy="group"` groups rows by selected bit inside a
  linear call.
- `QAQDPLLM_Linear.batch_policy="max"` uses the maximum selected bit for the
  current tensor.
- `QAQDPLLMForCausalLM.get_router_stats()` reports aggregate bit counts,
  effective bits, fallback counts, DP guard counts, and per-layer stats.
- `scripts/benchmark_qaq_modes.py` provides CUDA-synchronized repeated
  generation timing for fixed prompt batches.

These are not yet serving-level dynamic batching. Serving-level dynamic batching
requires request traces, arrival times, queueing, lane assignment, shared profile
composition, and end-to-end latency accounting.

## Trace Schema

Trace records should be JSONL. Unknown fields must be written as `UNVALIDATED`,
not guessed.

### Request Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `request_id` | string | yes | Stable request identifier. |
| `arrival_time_s` | float | yes | Request arrival time in the replay timeline. |
| `workload_type` | string | yes | Example: `chat`, `code`, `math`, `summarization`, `long_context`, `rag`. |
| `prompt_text_hash` | string | yes | Hash of prompt text; do not require storing raw prompts in every trace. |
| `prompt_length_tokens` | int | yes | Tokenized prompt length. |
| `target_output_length_tokens` | int or string | yes | Desired or replayed output length; use `UNVALIDATED` if unknown. |
| `observed_output_length_tokens` | int or string | yes | Generated token count from measured run. |
| `qos_deadline_ms` | float or string | yes | End-to-end deadline; use `UNVALIDATED` if no QoS policy is assigned. |
| `reference_mode` | string | yes | Usually `fixed_high` or another documented reference. |

### Precision Profile Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `predicted_scalar_bit_budget` | float or string | yes | Request-level predicted average bit budget. |
| `predicted_block_precision_vector` | list or string | yes | Coarse block/layer-group target bits. |
| `profile_id` | string or int | yes | Codebook or lane assignment. |
| `profile_distance` | float or string | yes | Distance to assigned profile. |
| `uncertainty_score` | float or string | yes | Router/profile uncertainty. |
| `fallback_probability` | float or string | yes | Predicted fallback risk. |

### Observed QAQ Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `average_selected_bit` | float | yes | From measured router stats. |
| `effective_bits` | float | yes | Param-weighted effective bits. |
| `per_layer_bit_counts` | object | yes | Per route/layer bit histogram. |
| `fallback_count` | int | yes | Confidence fallback count. |
| `fallback_fraction` | float | yes | Confidence fallback fraction. |
| `dp_guard_trigger_count` | int | yes | DP guard trigger count, if mode uses guard; otherwise zero or `UNVALIDATED`. |
| `dp_guard_trigger_fraction` | float | yes | DP guard trigger fraction. |
| `under_precision_label` | bool or string | yes | Whether shared profile used fewer bits than observed/requested need. |
| `over_precision_label` | bool or string | yes | Whether shared profile spent extra bits beyond observed/requested need. |

### Serving Replay Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `batch_id` | string | yes | Batch or microbatch identifier. |
| `lane_id` | string | yes | Assigned precision lane/profile. |
| `batch_policy` | string | yes | Scheduler policy name. |
| `shared_profile_policy` | string | yes | `none`, `max`, `quantile`, `scalar_bucket`, `block_profile`, etc. |
| `compatibility_threshold` | float or string | yes | Threshold used by scheduler. |
| `queue_delay_ms` | float | yes | Time from arrival to scheduled execution. |
| `gpu_execution_ms` | float or string | yes | CUDA-synchronized measured GPU generation time when available. |
| `end_to_end_latency_ms` | float | yes | `queue_delay_ms + gpu_execution_ms` or replay-equivalent measured latency. |
| `ttft_ms` | float or string | yes | Time to first token, including queueing. |
| `tpot_ms` | float or string | yes | Time per output token. |
| `deadline_missed` | bool or string | yes | Whether request missed QoS deadline. |

### System Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `kernel_launches_per_token` | float or string | yes | `UNVALIDATED` until profiler-backed. |
| `profile_switches_per_token` | float or string | yes | Number of precision/profile switches per generated token. |
| `transfer_bytes_per_token` | float or string | yes | CPU-GPU or GPU-GPU transfer bytes per token; `UNVALIDATED` until measured. |
| `hbm_bytes_per_token` | float or string | yes | HBM traffic per token; `UNVALIDATED` until measured. |
| `prefetch_hit_fraction` | float or string | yes | `UNVALIDATED` until prefetch exists. |
| `cuda_graph_reuse_fraction` | float or string | yes | `UNVALIDATED` unless CUDA graph paths are implemented. |

### Quality Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `quality_metric_name` | string | yes | Example: perplexity, exact match, pass@1, task accuracy. |
| `quality_metric_value` | float or string | yes | Measured value or `UNVALIDATED`. |
| `reference_quality_metric_value` | float or string | yes | Reference mode metric. |
| `quality_delta_vs_reference` | float or string | yes | Difference from reference. |
| `finite_logits` | bool | yes | Whether logits were finite in validation. |

## Scheduler Policies

Every policy must run on the same request trace, prompt set, output length policy,
QoS configuration, model path, router checkpoint, candidate bits, and estimator
artifacts.

### Mandatory Baselines

| Policy | Purpose |
| --- | --- |
| `ordinary_dynamic_batching` | Arrival/deadline/length batching without precision awareness. |
| `fixed_low` | Lowest valid precision baseline. |
| `fixed_high` | Highest valid precision/reference runtime baseline. |
| `static_fixed_bit` | Static fixed candidate bit, evaluated for each available bit when feasible. |
| `qaq_per_request` | QAQ routing without profile-aware request grouping. |
| `dp_threshold_only` | DP-LLM-style threshold baseline when estimator artifacts are available. |
| `mlp_multibit_dp_guard` | QAQ plus DP guard when implemented and validated. |

### Precision-Aware Policies

| Policy | Shared Profile Rule | Expected Use |
| --- | --- | --- |
| `scalar_budget_batching` | Bucket by predicted average bit budget. | Tests whether a scalar profile is enough. |
| `block_profile_batching` | Bucket by coarse layer/block precision vector. | Tests whether structure beyond scalar bits matters. |
| `max_profile_sharing` | Per route, use max requested bit across requests in batch. | Safe but may over-spend bits. |
| `quantile_profile_sharing` | Per route, use quantile requested bit across requests in batch. | Tests quality/efficiency frontier with fallback. |
| `fallback_lane` | High-uncertainty or repeated-fallback requests use conservative lane. | Tests whether hard requests can be isolated. |

### Compatibility Distance

Initial scheduler experiments should use a transparent distance:

```text
D(i, j) =
  alpha * abs(scalar_bits_i - scalar_bits_j)
  + beta * block_profile_distance(i, j)
  + gamma * length_bucket_distance(i, j)
  + delta * qos_incompatibility(i, j)
  + eta * uncertainty_penalty(i, j)
```

All coefficients must be logged. Learned or tuned distances are allowed only
after the fixed-distance baseline is reported.

## Initial Trace Collector

The first collector entry point is:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 python scripts/collect_qaq_profile_traces.py \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --bits 3 4 5 6 \
  --router_mode mlp_multibit_dp_guard \
  --prompt_file <PROMPTS_TXT_OR_JSONL> \
  --max_requests 100 \
  --workload_type mixed \
  --max_new_tokens 16 \
  --device cuda \
  --output_jsonl artifacts/qaq_trace.jsonl \
  --summary_json artifacts/qaq_trace_summary.json
```

The collector runs real QAQ generation one request at a time and writes one
JSONL record per request. It validates trace extraction only. It does not
validate dynamic batching, queueing behavior, quality preservation, transfer
bytes, kernel-switch counts, or throughput improvement.

The first trace-driven simulator entry point is:

```bash
cd /nfs/home/s314511048/dpqaq
python scripts/simulate_qaq_dynamic_batching.py \
  --trace_jsonl artifacts/qaq_trace.jsonl \
  --output_json artifacts/qaq_batching_simulation.json \
  --max_batch_size 4 \
  --max_wait_ms 200 \
  --compatibility_threshold 0.25 \
  --scalar_bucket_size 0.25 \
  --quantile 0.75
```

The simulator consumes observed per-request QAQ profiles and measured
single-request generation times. It produces scheduling and profile-sharing
metrics under explicit assumptions. Its output is `SIMULATED_ONLY` and cannot
support real throughput or latency claims without GPU-server replay.

## Validation Gates

### Gate 0: Existing QAQ Correctness

Dynamic batching work cannot support paper claims until the single-request QAQ
path is validated:

- Router checkpoint reload roundtrip passes.
- Route-map identity is checked at load time.
- Label generation uses real low-bit versus reference-bit relative error.
- `fixed_low`, `fixed_high`, `mlp_multibit`, `dp_threshold_only`, and
  `mlp_multibit_dp_guard` behave as documented for the tested configuration.
- Confidence fallback and DP guard counts are reported separately.
- Generation stats are not contaminated by sanity-check forward passes.
- Effective-bits math is checked against direct per-linear weighted counts.

### Gate 1: Real Trace Collection

Required evidence:

- Trace uses real prompts and real QAQ router/model execution.
- Trace includes at least two workload types before making cross-workload claims.
- Per-request profile records include observed selected bits and per-layer bit
  histograms.
- Fallback and DP guard counts are present.
- Any synthetic arrival process is explicitly marked as synthetic.

Passing Gate 1 does not validate batching performance.

### Gate 2: Trace-Driven Simulation

Required evidence:

- All mandatory baselines and precision-aware policies run on the same trace.
- End-to-end latency includes queue delay.
- Reports p50, p95, and p99 latency, not only mean latency.
- Reports lane occupancy, profile fragmentation, under-precision rate,
  over-precision rate, fallback rate, deadline miss rate, and effective bits.
- Simulator assumptions for service time, transfer cost, and fallback cost are
  written into the output artifact.

Passing Gate 2 validates only simulator behavior, not real throughput.

### Gate 3: GPU-Server Replay

Required evidence:

- Run on the lab GPU server with `CUDA_VISIBLE_DEVICES` explicitly set.
- Use real Any-Precision model path, router checkpoint, estimator path if used,
  candidate bits, prompt set, and max-new-token policy.
- Use warmups and repeated measurements.
- Synchronize CUDA before and after measured generation regions.
- Measure or report `UNVALIDATED` for transfer bytes/token, kernel/profile
  switch counts, and profiler-only metrics.
- Include p50, p95, p99 latency, tokens/sec, requests/sec, effective bits,
  fallback fraction, DP guard fraction, per-layer bit histogram, GPU model, CUDA
  version, and git commit.

Only Gate 3 can support latency or throughput claims.

### Gate 4: Quality Validation

Required evidence:

- Quality is measured on a documented dataset split/subset.
- Same prompts and generation policy are used across compared modes.
- Reference mode is documented.
- Quality degradation, fallback fraction, under-precision rate, and
  over-precision rate are reported together.
- Single-prompt sanity checks are marked `UNVALIDATED` for quality claims.

## Minimum Figures

Before writing paper claims, produce:

- Latency/throughput Pareto frontier by policy.
- p50/p95/p99 latency versus arrival rate.
- Effective bits versus quality metric.
- Queue delay versus compatibility threshold.
- Fallback rate versus quantile profile level.
- Lane occupancy heatmap.
- Profile-codebook size versus fragmentation.
- Transfer bytes/token versus batching policy, or `UNVALIDATED`.
- Kernel/profile switch count versus batching policy, or `UNVALIDATED`.
- Under-precision and over-precision rates by workload type.

## Done Criteria For First Implementation

The first implementation is done only when:

- Trace schema is emitted as JSONL with required fields.
- Simulator can replay the trace under all mandatory scheduler policies.
- Output artifact records all assumptions and marks missing fields
  `UNVALIDATED`.
- Unit tests cover profile construction, compatibility distance, deadline
  handling, queue-delay accounting, and max/quantile shared-profile composition.
- At least one real QAQ trace collection command is documented.
- No performance claim is made without Gate 3.

## Known UNVALIDATED Items

- Whether real prompts produce enough reusable precision-profile diversity.
- Whether scalar budgets are sufficient.
- Whether block profiles improve quality/fallback behavior enough to justify
  added fragmentation.
- Whether queueing delay erases any execution gain.
- Whether fallback is rare enough under quantile sharing.
- Whether transfer bytes/token or profile switch counts improve on real GPU
  execution.
- Whether dynamic batching helps low-load, medium-load, or high-load regimes.
