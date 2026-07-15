# QAQ Profile-Aware Batching Benchmark

## Current v2 execution contract

The deployable `max_profile_sharing` policy uses each request's held-out
`predicted_group_profile`. For the actual scheduled batch it computes the
component-wise continuous maximum once, validates the frozen
`layer_group_size` and profile dimension, maps each route with
`route.layer // layer_group_size`, and projects each demand to that route's
sorted valid bits using the conservative ceiling rule: smallest valid bit at
or above demand, minimum below the valid range, and maximum above it. Route
ceilings are recorded. The resulting complete route map is applied through
`QAQDPLLMForCausalLM.shared_profile(...)` across prefill and every decode step.

During pure shared execution the MLP router, confidence fallback, and DP guard
are bypassed. The linear receives one supplied bit and executes every row at
that bit; actual `comp_count`, per-route histograms, effective bits, and the
decision observer therefore describe execution rather than padding estimates.
A singleton still applies its own predicted profile. `fcfs` remains ordinary
grouped QAQ, `fixed_high` remains fixed-high, and `length_fcfs` remains a
scheduling-only baseline. `predicted_block_fallback_lane` retains its
uncertain fixed-high lane. Oracle profiles are diagnostic grouping inputs only,
not deployable shared-profile inputs. Quantile sharing remains pending.

Scheduler-profile under/exact/over counts compare executed bits with each
request's projected predicted target and report signed and absolute gaps. They
are not real output-error safety metrics. The separate `QAQPrecisionAuditor`
continues to define route-safety underprecision from real reference-bit output
errors.

The CPU validation command is:

    python -m pytest tests/router/test_qaq_dp_guard.py tests/router/test_benchmark_qaq_profile_batching.py tests/router/test_qaq_online_scheduler_replay.py tests/router/test_qaq_shared_profile.py -q

No real CUDA run was performed for this implementation change. The bounded
lab-server command still required is a single-policy run with explicit device
selection, for example:

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qaq_profile_batching.py --collection_dir <REQUEST_DEMAND_COLLECTION> --analysis_json <PREDICTOR_ANALYSIS_JSON> --ap_model_path <AP_MODEL_PATH> --router_checkpoint <ROUTER_CHECKPOINT> --estimator_results <ESTIMATOR_DIR> --datasets wikitext2 --request_limit 1 --max_new_tokens 2 --arrival_rate 20 --arrival_seed 101 --predictor_seed 17 --policies max_profile_sharing --max_batch_size 2 --warmup_batches 1 --repeat 1 --skip_quality_audit --device cuda:0 --output_json <OUTPUT_PATH>

The measured tables below are historical v1 artifacts and used the old
router-max execution hook. They are not v2 shared-profile results and are not
overwritten or reinterpreted.

## Historical v1 report

The v1 report replays one frozen held-out
request stream through six policies: `fcfs`, `scalar_predicted`,
`oracle_profile`, `predicted_profile`, `uncertainty_fallback`, and
`fixed_high`.

All policies use identical request IDs, prompt windows, native continuation
lengths, arrival timestamps, candidate bits, model artifacts, and predictor
seed. Each trace contains 192 requests: 96 native 32-token and 96 native
128-token continuations, including 27 deliberately uncertain held-out requests.
The Any-Precision CUDA kernel caps the batch dimension at 8, so the benchmark
uses `--max_batch_size 8`.

## Final low/medium sweep

The final sweep contains 36 real-CUDA artifacts: arrival rates 100 and 300
requests/s, arrival seeds 101/202/303, six policies, one warmup batch, ten
measured repeats, native `--max_new_tokens 128`, full route-quality audits,
and the local model, router, estimator, tokenizer, and Any-Precision CUDA
kernels. The quality audit is a separate synchronized replay and is excluded
from timed latency and throughput.

The exact stream hashes are:

| Rate | Seed | Stream SHA-256 |
|---:|---:|---|
| 100 | 101 | `5787a9d7b4e04f9792b60c3e8225a191227431975e175c2ed755a3a491f21462` |
| 100 | 202 | `6171ba79f8b6bc2f9f780be95a436296beb9bd24fb256bbf02b048a138127542` |
| 100 | 303 | `d9fabf35eac3ba7c5b2b43b66a012db6071959d6121294c5bba7c7df292d1d8b` |
| 300 | 101 | `fabf25062c84316c4a99ee109e318a9178ae8a90bc8cda2336e868fe7c810c49` |
| 300 | 202 | `d59f39904c79df91581556a0b6928bb972736fe5e2e926f1715d3a4b71669f70` |
| 300 | 303 | `2181413e1492305cccceeaaf384fb8a734ebf5f49b20e8b0fb931ec91824a2cf` |

For each rate/seed group, all six policy artifacts have the same hash. The
result files are:

    /tmp/qaq-profile-native128-r10-b8-rate{100,300}-seed{101,202,303}-*.json

The report uses normal-approximate 95% CIs over 30 repeat-by-seed timing
observations per rate and policy. Quality violation is audited once per
artifact, so its interval is across the three arrival traces.

## Metrics and quality definition

The JSON reports p50/p95/p99 latency, TTFT, TPOT, queue delay, requests/sec,
generated tokens/sec, token-slot throughput, request and prompt occupancy,
predictor/scheduler overhead, profile padding, effective bits, average
selected bit, per-layer histograms, confidence fallbacks, DP guard triggers,
uncertainty fallback rate, and quality violations.

`quality_violation_rate` is the QAQPrecisionAuditor route-decision
underprecision rate: real low-bit/reference-bit output-error labels determine
the smallest safe candidate bit, and the metric counts decisions where the
executed bit is lower. It is not task accuracy or perplexity.

`profile_padding_fraction` is the mean component-wise padding from each
request's scheduler signal to the batch maximum. `predictor_overhead_ms`
measures consuming held-out predictor outputs and applying bucket/fallback
decisions; predictor training is not timed.

The final routing tables distinguish confidence fallback (router fallback
decisions per routed token) from uncertainty fallback (the request-level
conservative held-out lane). The uncertainty lane is 14.063% because 27 of 192
requests are intentionally uncertain.

## Task-level held-out quality

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

The current task-quality artifacts are `/tmp/qaq-task-quality-wikitext2-current.json`
and `/tmp/qaq-task-quality-c4-current.json`; both report `REAL_GPU_HELDOUT`,
finite logits, and the current repository commit.

## GPU sampling and phase profiling

The continuous sampler recorded 99,000 rows (12,375 samples per GPU) with zero
sampler errors. In the final benchmark window, GPUs1--7 contributed 2,598
samples each; GPU0 was occupied by unrelated work and was excluded. Peak
residency on the benchmark GPUs was 10,184--11,132 MiB, and mean sampled
utilization was 14.15--33.54% with observed peaks up to 100%.

Use `scripts/profile_qaq_phases.py` for a representative real CUDA batch. It
writes a CUDA-only `torch.profiler` TensorBoard trace and a JSON containing
synchronized CUDA-event totals for router, estimator, grouping, and
dequantized matmul. CPU activity plus memory profiling is intentionally
limited because it caused unbounded profiler bookkeeping on this model.
