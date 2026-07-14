# QAQ Profile-Aware Batching Benchmark

`scripts/benchmark_qaq_profile_batching.py` replays a frozen held-out request
stream through six policies: `fcfs`, `scalar_predicted`, `oracle_profile`,
`predicted_profile`, `uncertainty_fallback`, and `fixed_high`.

All policies use identical request IDs, prompt windows, native continuation
lengths, arrival timestamps, candidate bits, model artifacts, and predictor
seed. Each request trace contains 192 requests: 96 native 32-token and 96
native 128-token continuations, including 27 deliberately uncertain held-out
requests. The Any-Precision CUDA kernel caps the batch dimension at 8, so the
benchmark uses `--max_batch_size 8`.

## Completed low/medium sweep

The current sweep covers arrival rates 100 and 300 requests/s, arrival seeds
101/202/303, one warmup batch, two measured repeats, native
`--max_new_tokens 128`, full route-quality audits, and the real local model and
Any-Precision CUDA kernels. The six policies at each rate/seed share one
stream hash:

| Rate | Seed 101 stream hash | Seed 202 stream hash | Seed 303 stream hash |
|---:|---|---|---|
| 100 | `5787a9d7...f21462` | `6171ba79...127542` | `d9fabf35...d1d8b` |
| 300 | `fabf2506...810c49` | `d59f3990...669f70` | `2181413e...24a2cf` |

The JSON results are under
`/tmp/qaq-profile-native128-r2-b8-rate{100,300}-seed{101,202,303}-*`.
Quality auditing is a separate synchronized CUDA replay, excluded from
latency/throughput timing. One overlapping seed-303 medium-rate attempt was
discarded; clean isolated FCFS and uncertainty reruns produced the final
artifacts.

A prior high-load experiment used the same six policies, arrival seeds
101/202/303, ten measured repeats, and rate 1000; its summarized results remain
in `doc/performance-profile.md`.

## Metrics and quality definition

The JSON reports p50/p95/p99 latency, TTFT, TPOT, queue delay, requests/sec,
generated tokens/sec, token-slot throughput, request and prompt occupancy,
predictor/scheduler overhead, profile padding, effective bits, average
selected bit, per-layer histograms, confidence fallbacks, DP guard triggers,
uncertainty fallback rate, and quality violations.

`quality_violation_rate` is the QAQPrecisionAuditor route-decision
underprecision rate: real low-bit/reference-bit output-error labels determine
the smallest safe candidate bit, and the metric counts decisions where the
executed bit is lower. It is not task accuracy or perplexity. Quality auditing
is a separate CUDA-backed replay and is excluded from timed latency and
throughput.

`profile_padding_fraction` is the mean component-wise padding from each
request's scheduler signal to the batch maximum. `predictor_overhead_ms`
measures consuming held-out predictor outputs and applying bucket/fallback
decisions; predictor training is not timed.

## Task-level quality evaluation

`scripts/evaluate_qaq_heldout.py` was run on 16 held-out WikiText2 windows and
16 held-out C4 windows, each with context length 512 and 8,176 scored target
tokens. This is teacher-forced perplexity, complementary to the request-level
route violation audit:

| Dataset | Mode | Perplexity | Delta vs fixed-high | Effective bits | Fallback | DP guard |
|---|---|---:|---:|---:|---:|---:|
| WikiText2 | fixed_low | 12.7770 | +3.2600 | 3.0000 | 0.000% | 0.000% |
| WikiText2 | dp_threshold_only | 9.8755 | +0.3586 | 4.4103 | 0.000% | 0.000% |
| WikiText2 | mlp_multibit | 9.6590 | +0.1420 | 5.1828 | 42.053% | 0.000% |
| WikiText2 | mlp_multibit_dp_guard | 9.6225 | +0.1055 | 5.1958 | 42.053% | 1.206% |
| WikiText2 | fixed_high | 9.5170 | +0.0000 | 6.0000 | 0.000% | 0.000% |
| C4 | fixed_low | 15.0702 | +3.1779 | 3.0000 | 0.000% | 0.000% |
| C4 | dp_threshold_only | 12.3132 | +0.4209 | 4.4490 | 0.000% | 0.000% |
| C4 | mlp_multibit | 12.0831 | +0.1908 | 5.1148 | 37.261% | 0.000% |
| C4 | mlp_multibit_dp_guard | 12.0455 | +0.1533 | 5.1287 | 37.261% | 1.496% |
| C4 | fixed_high | 11.8923 | +0.0000 | 6.0000 | 0.000% | 0.000% |

The current artifacts are `/tmp/qaq-task-quality-wikitext2-current.json` and
`/tmp/qaq-task-quality-c4-current.json`; both report `REAL_GPU_HELDOUT`, finite
logits, and the current repository commit.

## Profiling

Use `scripts/profile_qaq_phases.py` for a representative real CUDA batch. It
writes a CUDA-only `torch.profiler` TensorBoard trace and a JSON containing
synchronized CUDA-event totals for router, estimator, grouping, and dequantized
matmul. CUDA-only tracing is intentional because CPU activity plus memory
profiling caused unbounded profiler bookkeeping on this model; the JSON phase
counters remain the attribution source.
