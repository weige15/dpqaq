# Performance Profile

## Benchmark Command

The low/medium sweep used the existing real CUDA benchmark path with the same
frozen held-out request stream for every policy:

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
      --max_wait_ms 50 --warmup_batches 1 --repeat 2 \
      --confidence_threshold 0.6 --device cuda:0 --local_files_only \
      --output_json /tmp/qaq-profile-native128-r2-b8-rate<rate>-seed<seed>-<policy>.json

The 36 completed runs cover rates 100 and 300 requests/s, arrival seeds
101/202/303, six policies, one warmup, and two measured repeats. Each trace
contains 192 requests: 96 native 32-token and 96 native 128-token
continuations, including 27 deliberately uncertain held-out requests. Every
policy at a given rate and seed reports the same stream hash. Timing used
CUDA synchronization before and after cached prefill/decode batches. Quality
auditing was a separate synchronized CUDA replay and was excluded from timed
latency and throughput. Continuous `nvidia-smi` sampling ran during the
seed-101 rate-100/rate-300 waves and the seed-202/303 rate-100 waves; the
isolated seed-202/303 rate-300 reruns have no dedicated sampler CSV.

Task-level quality evaluation used the real held-out evaluator on 16 WikiText2
and 16 C4 windows of context length 512, scoring 8,176 target tokens per
dataset:

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qaq_heldout.py \
      --ap_model_path <AP_MODEL_PATH> --router_checkpoint <ROUTER_CHECKPOINT> \
      --estimator_results <ESTIMATOR_DIR> --tokenizer_path <TOKENIZER_PATH> \
      --context_length 512 --dataset_start 0 --num_examples 16 \
      --bits 3 4 5 6 --confidence_threshold 0.6 --fallback_bits 1 \
      --device cuda:0 --dataset <wikitext2-or-c4> --output_json <JSON>

Phase profiling used `scripts/profile_qaq_phases.py` with
`torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)` and a CUDA-only
trace on a representative native-32 batch. The environment was PyTorch
2.4.0+cu124 on an RTX 3090 with local model, router, estimator, tokenizer, and
held-out data artifacts.

## Baseline Result

The previously completed high-load baseline used rates of 1000 requests/s,
arrival seeds 101/202/303, ten repeats, and the same six policies. Means across
those three traces were:

| Policy | p50/p95 latency ms | TTFT p50 ms | TPOT p50 ms | Requests/s | Tokens/s | Effective bits | Occupancy | Quality violation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 488943 / 1101993 | 444651 | 496.8 | 0.2112 | 16.90 | 5.1382 | 1.0000 | 4.2705% |
| scalar_predicted | 316165 / 596381 | 286651 | 295.6 | 0.3123 | 24.99 | 5.7800 | 1.0000 | 0.5283% |
| oracle_profile | 309979 / 580342 | 283671 | 296.0 | 0.3151 | 25.21 | 5.7800 | 1.0000 | 0.5283% |
| predicted_profile | 314111 / 593250 | 286777 | 298.1 | 0.3118 | 24.94 | 5.7800 | 1.0000 | 0.5283% |
| uncertainty_fallback | 286738 / 558688 | 249102 | 299.7 | 0.3301 | 26.41 | 5.8073 | 0.8677 | 0.4715% |
| fixed_high | 70149 / 131248 | 64644 | 58.2 | 1.4037 | 112.29 | 6.0000 | 1.0000 | 0.0000% |

## Current Result

The following are means plus population standard deviation across arrival seeds
101, 202, and 303. Latencies, TTFT, TPOT, and predictor overhead are
milliseconds. Occupancy and profile padding are fractions. Fallback, guard, and
quality violation are percentages. Quality violation is the route-level
underprecision rate from `QAQPrecisionAuditor`, not task accuracy or
perplexity.

Rate 100 requests/s:

| Policy | p50 | p95 | TTFT | TPOT | Requests/s | Tokens/s | Effective bits | Occupancy | Padding | Quality violation | Predictor ms | Fallback | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 794653±19270 | 1521060±13027 | 772874±20087 | 356.2±4.2 | 0.1198±0.0024 | 9.580±0.190 | 5.1396±0.0002 | 0.4366±0.0110 | 0.0000 | 4.467±0.028% | 0.000±0.000 | 0.000% | 1.230±0.002% |
| scalar_predicted | 709118±85456 | 1328761±131510 | 684066±84478 | 305.1±30.6 | 0.1377±0.0126 | 11.017±1.010 | 5.7212±0.0063 | 0.4366±0.0110 | 0.0032±0.0001 | 0.916±0.052% | 0.766±0.010 | 0.000% | 1.228±0.004% |
| oracle_profile | 735215±120314 | 1421690±232927 | 709288±106880 | 287.0±6.3 | 0.1381±0.0092 | 11.048±0.738 | 5.7212±0.0063 | 0.4366±0.0110 | 0.0056±0.0002 | 0.916±0.052% | 2.250±0.023 | 0.000% | 1.228±0.004% |
| predicted_profile | 721269±109340 | 1403976±242404 | 707409±113110 | 284.6±1.8 | 0.1400±0.0112 | 11.198±0.893 | 5.7212±0.0063 | 0.4366±0.0110 | 0.0029±0.0001 | 0.916±0.052% | 2.106±0.007 | 0.000% | 1.228±0.004% |
| uncertainty_fallback | 638732±14429 | 1229490±28117 | 619499±8408 | 282.3±6.9 | 0.1475±0.0052 | 11.802±0.414 | 5.7461±0.0076 | 0.3382±0.0068 | 0.0020±0.0001 | 0.882±0.085% | 2.197±0.065 | 14.063% | 1.083±0.004% |
| fixed_high | 124177±7966 | 231374±9940 | 121003±6757 | 54.8±5.7 | 0.7873±0.0330 | 62.987±2.639 | 6.0000±0.0000 | 0.4366±0.0110 | 0.0000 | 0.000% | 0.000±0.000 | 0.000% | 0.000% |

Rate 300 requests/s:

| Policy | p50 | p95 | TTFT | TPOT | Requests/s | Tokens/s | Effective bits | Occupancy | Padding | Quality violation | Predictor ms | Fallback | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 424893±11206 | 815004±18920 | 389530±6899 | 368.5±1.6 | 0.2202±0.0047 | 17.616±0.377 | 5.1383±0.0001 | 0.8579±0.0250 | 0.0000 | 4.268±0.013% | 0.000±0.000 | 0.000% | 1.215±0.000% |
| scalar_predicted | 359646±29580 | 686199±70110 | 339818±28218 | 309.8±35.7 | 0.2643±0.0231 | 21.144±1.848 | 5.7721±0.0016 | 0.8579±0.0250 | 0.0055±0.0000 | 0.584±0.012% | 0.501±0.022 | 0.000% | 1.212±0.001% |
| oracle_profile | 340785±7995 | 652646±12886 | 312971±4213 | 292.2±3.8 | 0.2732±0.0063 | 21.854±0.503 | 5.7721±0.0016 | 0.8579±0.0250 | 0.0087±0.0001 | 0.584±0.012% | 1.669±0.068 | 0.000% | 1.212±0.001% |
| predicted_profile | 336195±2529 | 637699±16355 | 309272±4528 | 282.5±0.9 | 0.2783±0.0060 | 22.260±0.482 | 5.7721±0.0016 | 0.8579±0.0250 | 0.0046±0.0001 | 0.584±0.012% | 1.451±0.013 | 0.000% | 1.212±0.001% |
| uncertainty_fallback | 315187±6469 | 627631±24948 | 308913±8915 | 284.5±8.4 | 0.2835±0.0139 | 22.683±1.112 | 5.7983±0.0009 | 0.6219±0.0267 | 0.0029±0.0002 | 0.531±0.002% | 1.536±0.071 | 14.063% | 1.058±0.007% |
| fixed_high | 71527±1062 | 137165±1566 | 65880±432 | 51.9±2.1 | 1.3176±0.0117 | 105.411±0.940 | 6.0000±0.0000 | 0.8579±0.0250 | 0.0000 | 0.000% | 0.000±0.000 | 0.000% | 0.000% |

Task-level held-out quality is separate from route-level violations. The
current real-GPU evaluator uses 16 windows from each dataset, context length
512, and 8,176 scored target tokens. It reports teacher-forced perplexity
against fixed-high:

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

The successful CUDA-only torch.profiler traces were paired with the existing
CUDA-event phase timing in `QAQDPLLM_Linear`. On a representative native-32
batch of seven requests, FCFS grouped execution accumulated 22.745 s of CUDA
phase time over the profiled window: router 7.996 s (35.2%), estimator 1.193 s
(5.2%), grouping 4.302 s (18.9%), and dequantized matmul 3.287 s (14.4%). The
remaining time is attention, cache, sampling, and framework work.

The predicted-profile shared-maximum path accumulated 20.488 s: router
10.926 s (53.3%) and estimator 1.672 s (8.2%); grouping and dequantized
matmul are intentionally zero because that execution policy bypasses the
per-bit grouped path. This makes the router the clearest adaptive-path hotspot
and separates scheduler overhead from grouped dequantization overhead.

The CUDA-only profiler trace has valid kernel/device events and a TensorBoard
trace directory. Its CPU `record_function` phase rows are empty because CPU
activity was disabled to keep memory bounded; the phase attribution above is
from synchronized CUDA events, not inferred from wall-clock percentages.

## Time Breakdown

All benchmark timing used one warmup batch followed by two measured repeats,
with CUDA synchronization before and after each cached prefill/decode batch.
Quality auditing was a separate CUDA replay and was excluded from latency and
throughput. The profiler used one warmup and one active step on the same
native-32 request format. Predictor overhead covers consuming held-out
predictor outputs and applying scheduler decisions; it excludes model loading,
tokenization, predictor training, JSON output, and quality auditing.

## Memory Breakdown

The low/medium sampler traces observed roughly 10--12 GiB model residency on
24 GiB RTX 3090 devices during valid waves. The successful profiler runs
disabled `profile_memory` and wrote compact CUDA traces. Earlier CPU+CUDA
profile-memory attempts caused multi-GiB profiler bookkeeping growth and were
terminated; they are not used as benchmark results.

## I/O Breakdown

All inputs were local. Result JSON files and sampler CSVs were written under
`/tmp`; profiler JSON and TensorBoard traces were also written under `/tmp`.
No model, checkpoint, dataset, or large benchmark artifact was added to the
repository.

## GPU Utilization

Continuous `nvidia-smi` sampling was active during the seed-101 low/medium
waves and the seed-202/303 low-rate waves. The sampled seed-101 rate-100
policies showed roughly 32--37% mean utilization with p95 74--81%; the
rate-300 adaptive wave averaged 33.1% with p95 90%, while fixed-high averaged
63.2% with p95 100%. The seed-202/303 low-rate samplers provide additional
trace coverage. These are device samples, not kernel-level attribution; the
isolated seed-202/303 rate-300 reruns were timed with CUDA synchronization but
did not have dedicated sampler CSVs.

## Bottleneck Hypotheses

1. QAQ routing is the dominant adaptive execution cost: the grouped FCFS
   profile measured 35.2% of profiled CUDA phase time in the router, while the
   shared-profile path measured 53.3%. Confidence: high.

2. Grouping and dequantized matmul are material only on the per-bit grouped
   path: together they were 33.3% of the FCFS profiled CUDA phase time and were
   bypassed by the shared-maximum path. Confidence: high.

3. Arrival rate changes occupancy and queueing more than it changes route
   quality: across the three traces, occupancy rose from 0.437 to 0.858 for
   the normal policies, while quality violations changed only modestly.
   Confidence: medium.

4. Conservative uncertainty fallback trades occupancy for safety: it used
   14.063% fallback at both rates, reduced route violations to 0.882%/0.531%,
   and had occupancy 0.338/0.622. Confidence: medium.

The next optimization should target router execution and the grouped
dequantization path separately, using the CUDA-only trace plus synchronized
phase counters as the regression gate; the current results support a
research-prototype scheduler claim but are not publication-level evidence
until the low/medium sweep is repeated with the requested ten measured
repeats and a broader task suite.
