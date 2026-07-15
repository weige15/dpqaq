# Performance Profile

## Semantics and validation status

This file's historical tables and phase measurements predate the explicit
`max_profile_sharing` execution contract. They describe the old router-max
hook and must not be read as evidence for v2 scheduler-supplied profile
execution. Existing artifacts are retained unchanged.

The v2 policy uses held-out `predicted_group_profile` values, validates their
frozen layer-group metadata, takes one component-wise maximum over the actual
batch, maps groups using the validated route map, and conservatively projects
each demand onto each route's actual valid bits. The complete route profile is
held constant through prefill and decode by
`QAQDPLLMForCausalLM.shared_profile(...)`. Router, confidence fallback, and
DP guard are bypassed, and actual route bits—not padding estimates—feed
statistics and the decision observer. Singleton batches use the same path.

Scheduler-profile under/exact/over accounting compares execution with each
request's projected predicted target. It is separate from real route-safety
underprecision reported by `QAQPrecisionAuditor` from output-error labels.
`fcfs` remains grouped QAQ; `fixed_high` remains fixed-high; `length_fcfs`
remains scheduling-only; and quantile sharing is pending.

The required CPU validation command is:

    CUDA_VISIBLE_DEVICES='' /nfs/home/s314511048/.venv/bin/python -m pytest tests/router/test_qaq_dp_guard.py tests/router/test_benchmark_qaq_profile_batching.py tests/router/test_qaq_online_scheduler_replay.py tests/router/test_qaq_shared_profile.py -q

The targeted command passed with 33 tests and the full router suite passed with
93 tests. The implementation was then exercised on GPU 4, an isolated RTX 3090,
using the real CUDA kernels. The two bounded timing runs and the separate audit
were written under `/tmp` and are not repository artifacts.

    CUDA_VISIBLE_DEVICES=4 /nfs/home/s314511048/.venv/bin/python scripts/benchmark_qaq_profile_batching.py --collection_dir artifacts/qaq-request-demand-preregistered-v1 --analysis_json artifacts/qaq-request-demand-preregistered-v1-analysis/analysis.json --ap_model_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 --router_checkpoint checkpoints/qaq_router_llama31_8b_th005.pt --estimator_results estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam --tokenizer_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 --datasets wikitext2 --request_limit 4 --min_uncertain_requests 1 --max_new_tokens 8 --arrival_rate 20 --arrival_seed 101 --predictor_seed 17 --policies fixed_high fcfs max_profile_sharing --max_batch_size 2 --max_wait_ms 50 --warmup_batches 1 --repeat 3 --confidence_threshold 0.6 --device cuda:0 --torch_dtype float16 --skip_quality_audit --local_files_only --output_json /tmp/qaq-shared-profile-bounded-gpu4.json

The first run used `request_limit=4`, `min_uncertain_requests=1`,
`max_batch_size=2`, `repeat=3`, and compared `fixed_high`, `fcfs`, and
`max_profile_sharing` on GPU 4, an RTX 3090. The second run used eight
requests, `min_uncertain_requests=0`, `max_batch_size=4`, and compared
`fixed_high` with `max_profile_sharing` for three repeats on the same device.
The runs completed as `REAL_CUDA_BENCHMARK` on commit
`7fd62b150fb126ff48cceef2cdef02dbaf196a09`, with PyTorch 2.4.0+cu124 and CUDA
12.4. The second run measured fixed-high at 905.337 ms p50 and 33.4677
generated tokens/s, versus 916.279 ms p50 and 33.0366 tokens/s for shared
execution; both used effective bit 6.
The shared run used shared execution for 100% of rows, with zero fallbacks and
zero DP guards; all 5,376 scheduler-profile decisions were exact. Its separate
real-output-error audit made 241,920 decisions and found zero route-safety
underprecision violations, with 216,870 over-precision and 25,050 exact
decisions.

The frozen held-out predictor is the immediate research bottleneck: across all
96 WikiText2 test requests, every group demand was above 5 (group maxima ranged
from 5.4675 to 5.5362), so conservative projection onto valid bits 3/4/5/6
selected bit 6 for every route. This is a calibration/discretization finding,
not evidence that the shared execution mechanism is broken.

## Benchmark Command

The final low/medium sweep used the existing real-CUDA benchmark path:

    CUDA_VISIBLE_DEVICES=<free-gpu> python scripts/benchmark_qaq_profile_batching.py \
      --collection_dir artifacts/qaq-request-demand-preregistered-v1 \
      --analysis_json artifacts/qaq-request-demand-preregistered-v1-analysis/analysis.json \
      --ap_model_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --router_checkpoint checkpoints/qaq_router_llama31_8b_th005.pt \
      --estimator_results estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam \
      --tokenizer_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --datasets wikitext2 c4_new --request_limit 0 --min_uncertain_requests 1 \
      --max_new_tokens 128 --arrival_rate <100-or-300> --arrival_seed <101-or-202-or-303> \
      --predictor_seed 17 --policies <policy> --max_batch_size 8 \
      --max_wait_ms 50 --warmup_batches 1 --repeat 10 \
      --confidence_threshold 0.6 --device cuda:0 --local_files_only \
      --output_json /tmp/qaq-profile-native128-r10-b8-rate<rate>-seed<seed>-<policy>.json

Benchmark provenance: repository commit `066c8e02d1520c7579361638ea76b53c4fa6995d`.

The final matrix contains 36 real-CUDA runs: rates 100 and 300 requests/s,
arrival seeds 101/202/303, and six policies (`fcfs`, `scalar_predicted`,
`oracle_profile`, `predicted_profile`, `uncertainty_fallback`, and
`fixed_high`). Every trace contains 192 requests: 96 native 32-token and 96
native 128-token continuations, including 27 deliberately uncertain held-out
requests. Each policy at a given rate and seed has the same frozen-stream hash.
There was one warmup batch and ten measured repeats. CUDA synchronization was
used before and after each measured batch. Quality auditing was a separate
synchronized CUDA replay and was excluded from timed latency and throughput.

The report below uses normal-approximate 95% CIs over 30 repeat-by-seed
observations per rate and policy. Quality violation is audited once per
artifact, so its CI uses the three independent arrival traces rather than
pretending that the ten timing repeats are ten independent audits.

## Baseline Result

The fixed-high safety baseline is the fastest reference in both sweeps. At rate
100 it reaches 0.7509 +/- 0.0282 requests/s and 60.074 +/- 2.255 generated
tokens/s; at rate 300 it reaches 1.3185 +/- 0.0175 requests/s and
105.478 +/- 1.403 tokens/s. It uses six effective bits and has zero audited
route violations, confidence fallbacks, uncertainty fallbacks, and DP guards.

The earlier rate-1000 ten-repeat baseline remains available in the repository
history and is not mixed into the low/medium confidence intervals below.

## Current Result

All latency, TTFT, and TPOT values are milliseconds. Each cell is mean +/- 95%
CI; latency, TTFT, and TPOT show p50/p95. Quality violation is the
route-level underprecision rate from `QAQPrecisionAuditor`, not task accuracy
or perplexity.

Rate 100 requests/s:

| Policy | Latency p50/p95 | TTFT p50/p95 | TPOT p50/p95 | Requests/s | Tokens/s | Quality violation |
|---|---:|---:|---:|---:|---:|---:|
| fcfs | 804441 +/- 9696 / 1543231 +/- 50872 | 782276 +/- 9802 / 1505907 +/- 49503 | 359.47 +/- 2.26 / 399.15 +/- 33.55 | 0.1188 +/- 0.0030 | 9.503 +/- 0.238 | 4.467 +/- 0.039% |
| scalar_predicted | 677794 +/- 9513 / 1282453 +/- 16968 | 659358 +/- 10366 / 1250334 +/- 15644 | 293.28 +/- 1.76 / 315.01 +/- 7.44 | 0.1418 +/- 0.0022 | 11.343 +/- 0.176 | 0.916 +/- 0.072% |
| oracle_profile | 672852 +/- 9494 / 1271886 +/- 16550 | 654721 +/- 10101 / 1239817 +/- 14950 | 291.94 +/- 1.84 / 310.24 +/- 6.60 | 0.1429 +/- 0.0022 | 11.433 +/- 0.177 | 0.916 +/- 0.072% |
| predicted_profile | 669557 +/- 8789 / 1288890 +/- 38838 | 651222 +/- 9286 / 1252814 +/- 30877 | 292.17 +/- 1.94 / 321.80 +/- 32.64 | 0.1416 +/- 0.0035 | 11.328 +/- 0.281 | 0.916 +/- 0.072% |
| uncertainty_fallback | 662354 +/- 9258 / 1286015 +/- 33077 | 645644 +/- 9117 / 1252188 +/- 26131 | 289.91 +/- 2.28 / 335.12 +/- 22.86 | 0.1408 +/- 0.0041 | 11.262 +/- 0.330 | 0.882 +/- 0.118% |
| fixed_high | 130080 +/- 10173 / 244508 +/- 11638 | 127051 +/- 10135 / 239273 +/- 11650 | 54.54 +/- 1.20 / 65.77 +/- 10.11 | 0.7509 +/- 0.0282 | 60.074 +/- 2.255 | 0.000 +/- 0.000% |

Rate 300 requests/s:

| Policy | Latency p50/p95 | TTFT p50/p95 | TPOT p50/p95 | Requests/s | Tokens/s | Quality violation |
|---|---:|---:|---:|---:|---:|---:|
| fcfs | 427033 +/- 2681 / 821128 +/- 10469 | 391864 +/- 4058 / 775140 +/- 10329 | 370.37 +/- 2.83 / 382.73 +/- 5.04 | 0.2187 +/- 0.0025 | 17.498 +/- 0.201 | 4.268 +/- 0.018% |
| scalar_predicted | 346212 +/- 5083 / 668203 +/- 9864 | 317876 +/- 3844 / 630454 +/- 9626 | 298.09 +/- 2.16 / 308.88 +/- 3.11 | 0.2675 +/- 0.0040 | 21.402 +/- 0.317 | 0.584 +/- 0.017% |
| oracle_profile | 349274 +/- 6479 / 671429 +/- 12157 | 320663 +/- 5449 / 633562 +/- 12045 | 297.32 +/- 2.34 / 315.98 +/- 9.12 | 0.2664 +/- 0.0045 | 21.308 +/- 0.362 | 0.584 +/- 0.017% |
| predicted_profile | 346986 +/- 4583 / 669499 +/- 11206 | 318503 +/- 4092 / 632009 +/- 11062 | 298.06 +/- 2.06 / 309.49 +/- 4.42 | 0.2671 +/- 0.0043 | 21.368 +/- 0.344 | 0.584 +/- 0.017% |
| uncertainty_fallback | 330053 +/- 7681 / 651623 +/- 17021 | 323485 +/- 7295 / 624688 +/- 15074 | 295.86 +/- 7.09 / 314.93 +/- 10.74 | 0.2732 +/- 0.0070 | 21.856 +/- 0.559 | 0.530 +/- 0.003% |
| fixed_high | 71364 +/- 1292 / 136496 +/- 2156 | 66236 +/- 1094 / 129711 +/- 2123 | 53.45 +/- 0.92 / 55.40 +/- 1.49 | 1.3185 +/- 0.0175 | 105.478 +/- 1.403 | 0.000 +/- 0.000% |

Routing, batching, padding, and overhead:

| Rate | Policy | Effective bits | Batch occupancy | Profile padding bits / fraction | Predictor ms/repeat | Confidence fallback | Uncertainty fallback | DP guard |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | fcfs | 5.1396 +/- 0.0001 | 0.4366 +/- 0.0040 | 0.0000 / 0.000% | 0.000 +/- 0.000 | 35.861 +/- 0.015% | 0.000% | 1.230 +/- 0.001% |
| 100 | scalar_predicted | 5.7212 +/- 0.0023 | 0.4366 +/- 0.0040 | 0.0167 / 0.325% | 0.841 +/- 0.022 | 36.127 +/- 0.008% | 0.000% | 1.228 +/- 0.001% |
| 100 | oracle_profile | 5.7212 +/- 0.0023 | 0.4366 +/- 0.0040 | 0.0299 / 0.559% | 2.361 +/- 0.043 | 36.127 +/- 0.008% | 0.000% | 1.228 +/- 0.001% |
| 100 | predicted_profile | 5.7212 +/- 0.0023 | 0.4366 +/- 0.0040 | 0.0157 / 0.294% | 2.183 +/- 0.041 | 36.127 +/- 0.008% | 0.000% | 1.228 +/- 0.001% |
| 100 | uncertainty_fallback | 5.7461 +/- 0.0028 | 0.3382 +/- 0.0025 | 0.0109 / 0.204% | 2.316 +/- 0.033 | 31.778 +/- 0.037% | 14.063 +/- 0.000% | 1.083 +/- 0.001% |
| 100 | fixed_high | 6.0000 +/- 0.0000 | 0.4366 +/- 0.0040 | 0.0000 / 0.000% | 0.000 +/- 0.000 | 0.000% | 0.000% | 0.000% |
| 300 | fcfs | 5.1383 +/- 0.0000 | 0.8579 +/- 0.0091 | 0.0000 / 0.000% | 0.000 +/- 0.000 | 35.568 +/- 0.005% | 0.000% | 1.215 +/- 0.000% |
| 300 | scalar_predicted | 5.7721 +/- 0.0006 | 0.8579 +/- 0.0091 | 0.0281 / 0.546% | 0.554 +/- 0.014 | 35.901 +/- 0.010% | 0.000% | 1.212 +/- 0.000% |
| 300 | oracle_profile | 5.7721 +/- 0.0006 | 0.8579 +/- 0.0091 | 0.0464 / 0.867% | 1.836 +/- 0.087 | 35.901 +/- 0.010% | 0.000% | 1.212 +/- 0.000% |
| 300 | predicted_profile | 5.7721 +/- 0.0006 | 0.8579 +/- 0.0091 | 0.0244 / 0.456% | 1.647 +/- 0.088 | 35.901 +/- 0.010% | 0.000% | 1.212 +/- 0.000% |
| 300 | uncertainty_fallback | 5.7983 +/- 0.0003 | 0.6219 +/- 0.0097 | 0.0157 / 0.293% | 1.560 +/- 0.030 | 31.281 +/- 0.065% | 14.063 +/- 0.000% | 1.058 +/- 0.003% |
| 300 | fixed_high | 6.0000 +/- 0.0000 | 0.8579 +/- 0.0091 | 0.0000 / 0.000% | 0.000 +/- 0.000 | 0.000% | 0.000% | 0.000% |

“Confidence fallback” is the router-statistics fallback rate (fallback decisions
per routed token). “Uncertainty fallback” is the request-level conservative
held-out lane rate. The latter is 14.063% because 27 of 192 requests are
intentionally uncertain.

Task-level held-out quality is separate from route-level violations. The real
GPU evaluator used 16 WikiText2 windows and 16 C4 windows, context length 512,
and 8,176 scored target tokens per dataset. It reports teacher-forced
perplexity against fixed-high:

| Dataset / policy | Perplexity | Delta vs fixed-high | Effective bits | Fallback | Guard |
|---|---:|---:|---:|---:|---:|
| WikiText2 / fixed_low | 12.7770 | +3.2600 | 3.0000 | 0.000% | 0.000% |
| WikiText2 / dp_threshold_only | 9.8755 | +0.3586 | 4.4103 | 0.000% | 0.000% |
| WikiText2 / mlp_multibit | 9.6590 | +0.1420 | 5.1828 | 42.053% | 0.000% |
| WikiText2 / mlp_multibit_dp_guard | 9.6225 | +0.1055 | 5.1958 | 42.053% | 1.206% |
| WikiText2 / fixed_high | 9.5170 | +0.0000 | 6.0000 | 0.000% | 0.000% |
| C4 / fixed_low | 15.0702 | +3.1779 | 3.0000 | 0.000% | 0.000% |
| C4 / dp_threshold_only | 12.3132 | +0.4209 | 4.4490 | 0.000% | 0.000% |
| C4 / mlp_multibit | 12.0831 | +0.1908 | 5.1148 | 37.261% | 0.000% |
| C4 / mlp_multibit_dp_guard | 12.0455 | +0.1533 | 5.1287 | 37.261% | 1.496% |
| C4 / fixed_high | 11.8923 | +0.0000 | 6.0000 | 0.000% | 0.000% |

## Hotspots
The phase percentages below are historical router-routed/v1 evidence and must
not be used to attribute v2 shared-profile execution. A current full-process
Nsight capture of the bounded shared run recorded model-load host-to-device
transfers as 99.9% of H2D time; because loading was inside the capture, this is
not a serving-step bottleneck. A post-load phase-isolated trace remains pending.

The successful CUDA-only torch.profiler traces were paired with synchronized
CUDA-event phase timing in `QAQDPLLM_Linear`. On a representative native-32
batch of seven requests, FCFS grouped execution accumulated 22.745 s of CUDA
phase time over the profiled window: router 7.996 s (35.2%), estimator 1.193 s
(5.2%), grouping 4.302 s (18.9%), and dequantized matmul 3.287 s (14.4%). The
remaining time is attention, cache, sampling, and framework work.

The predicted-profile shared-maximum path accumulated 20.488 s: router
10.926 s (53.3%) and estimator 1.672 s (8.2%); grouping and dequantized matmul
are intentionally zero because that execution policy bypasses the per-bit
grouped path. This makes the router the clearest adaptive-path hotspot and
separates scheduler overhead from grouped dequantization overhead.

The CUDA-only profiler trace has valid kernel/device events and a TensorBoard
trace directory. Its CPU `record_function` phase rows are empty because CPU
activity was disabled to keep memory bounded; the phase attribution above is
from synchronized CUDA events, not inferred from wall-clock percentages.

## Time Breakdown

The final matrix used one warmup batch and ten measured repeats, with CUDA
synchronization before and after each cached prefill/decode batch. Quality
auditing was a separate CUDA replay and was excluded from latency and
throughput. Predictor overhead covers consuming held-out predictor outputs and
applying scheduler decisions; it excludes model loading, tokenization,
predictor training, JSON output, and quality auditing.

## Memory Breakdown

The final workers used roughly 10--11 GiB of model residency on 24 GiB RTX 3090
devices. The successful profiler runs disabled `profile_memory` and wrote
compact CUDA traces. Earlier CPU+CUDA profile-memory attempts caused multi-GiB
profiler bookkeeping growth and were terminated; they are not used as
benchmark results.

## I/O Breakdown

All inputs were local. The 36 result JSON files and sampler CSV were written
under `/tmp`; profiler JSON and TensorBoard traces were also written under
`/tmp`. No model, checkpoint, dataset, or large benchmark artifact was added to
the repository.

## GPU Utilization

The continuous sampler recorded 99,000 rows (12,375 samples per GPU) from
00:26:33 to 18:07:58 with zero sampler errors. In the final benchmark window
starting 14:25:00, each GPU contributed 2,598 samples. GPU0 remained occupied by
unrelated work and was deliberately excluded. GPUs1--7 reached 10,184--11,132
MiB peak residency and mean sampled utilization of 14.15--33.54%, with 100%
observed peaks. These are device samples, not kernel-level attribution; the
mean includes synchronization and scheduling gaps.

## Bottleneck Hypotheses

1. QAQ routing is the dominant adaptive execution cost: the grouped FCFS
   profile measured 35.2% of profiled CUDA phase time in the router, while the
   shared-profile path measured 53.3%. Confidence: high.

2. Grouping and dequantized matmul are material only on the per-bit grouped
   path: together they were 33.3% of the FCFS profiled CUDA phase time and were
   bypassed by the shared-maximum path. Confidence: high.

3. Arrival rate changes occupancy and queueing more than route quality: normal
   policy occupancy rose from 0.437 at rate 100 to 0.858 at rate 300, while
   audited violations remained in the same low-single-digit range. Confidence:
   medium.

4. Conservative uncertainty fallback trades occupancy for safety: it used
   14.063% request-level fallback at both rates, reduced route violations to
   0.882% and 0.530%, and had occupancy 0.338 and 0.622. Confidence: medium.

The next optimization should target router execution and grouped
dequantization separately, using the CUDA-only trace plus synchronized phase
counters as the regression gate. The ten-repeat low/medium matrix is suitable
as research-prototype evidence; publication-level claims still need a broader
task suite and independent GPU-isolation/repetition controls.
