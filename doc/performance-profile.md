# Performance Profile

## Executive Summary

The native-continuation frozen-stream benchmark is complete for six policies,
three deterministic arrival traces, ten measured repeats, and 192 held-out
requests per trace. The stream contains 96 native 32-token and 96 native
128-token continuations, plus 27 deliberately uncertain held-out requests per
trace. All timed model work used real CUDA execution with one warmup,
CUDA synchronization before and after measured batches, and continuous
nvidia-smi sampling.

Across traces, uncertainty-aware fallback is the strongest adaptive policy:
26.41 generated tokens/s, 286.7/558.7 ms p50/p95 latency, 5.807 effective
bits, and 0.471% route-level quality violations. Fixed-high remains the
safety/throughput ceiling at 112.29 tokens/s and zero violations. These are
route-level precision-audit results, not task accuracy or perplexity results.

## Scope and Baseline

The benchmark compares fcfs, scalar_predicted, oracle_profile,
predicted_profile, uncertainty_fallback, and fixed_high using identical
request IDs, prompt windows, continuation lengths, arrival traces, model
artifacts, and predictor seed. Profile policies use the existing
QAQDPLLM_Linear.batch_policy="max" path for multi-request batches;
FCFS and fixed-high use grouped execution. The installed Any-Precision CUDA
kernel enforces M <= 8, so --max_batch_size 8 is the largest valid batch in
this environment.

The real inputs are the preregistered WikiText-2 and C4 held-out request
collection, the packed Any-Precision Llama 3.1 8B model, the QAQ router
checkpoint, and the estimator artifacts. Arrival seeds are 101, 202, and 303;
the stream hashes are recorded in each JSON artifact. Predictor seed is 17 and
confidence threshold is 0.6.

## Current Result

### Native 32/128-token frozen stream

Run configuration:

    CUDA_VISIBLE_DEVICES=<free-gpu> python scripts/benchmark_qaq_profile_batching.py \
      --collection_dir artifacts/qaq-request-demand-preregistered-v1 \
      --analysis_json artifacts/qaq-request-demand-preregistered-v1-analysis/analysis.json \
      --ap_model_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --router_checkpoint checkpoints/qaq_router_llama31_8b_th005.pt \
      --estimator_results estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam \
      --tokenizer_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --datasets wikitext2 c4_new --request_limit 0 --min_uncertain_requests 1 \
      --max_new_tokens 128 --arrival_rate 1000 --arrival_seed 101 \
      --predictor_seed 17 --policies <policy> --max_batch_size 8 \
      --max_wait_ms 50 --warmup_batches 1 --repeat 10 \
      --confidence_threshold 0.6 --device cuda:0 --local_files_only \
      --output_json /tmp/qaq-profile-native128-r10-b8-seed101-<policy>.json

The command was repeated with arrival seeds 202 and 303, one policy process
per GPU, and a continuous nvidia-smi sampler writing a corresponding
-gpu.csv file. The seed-303 fixed-high process initially encountered an
external process occupying GPU 5; it was rerun with the identical configuration
on free GPU 6 and is included as fixed_high-retry.

Values below are means across the three arrival traces; the JSON artifacts
retain the per-trace values and standard deviations. Latency, TTFT, and TPOT
are milliseconds; padding is the profile-padding fraction; predictor is CPU
predictor-consumption overhead per request.

| Policy | p50 / p95 latency | TTFT p50 | TPOT p50 | Requests/s | Tokens/s | Effective bits | Occupancy | Padding | Quality violation | Predictor ms | Fallback / guard | Uncertainty fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 488943 / 1101993 | 444651 | 496.8 | 0.2112 | 16.90 | 5.1382 | 1.0000 | 0.0000 | 4.2705% | 0.0000 | 35.5524% / 1.2142% | 0.0000% |
| scalar_predicted | 316165 / 596381 | 286651 | 295.6 | 0.3123 | 24.99 | 5.7800 | 1.0000 | 0.0058 | 0.5283% | 0.6780 | 35.8953% / 1.2111% | 0.0000% |
| oracle_profile | 309979 / 580342 | 283671 | 296.0 | 0.3151 | 25.21 | 5.7800 | 1.0000 | 0.0095 | 0.5283% | 3.0818 | 35.8953% / 1.2111% | 0.0000% |
| predicted_profile | 314111 / 593250 | 286777 | 298.1 | 0.3118 | 24.94 | 5.7800 | 1.0000 | 0.0049 | 0.5283% | 2.5473 | 35.8953% / 1.2111% | 0.0000% |
| uncertainty_fallback | 286738 / 558688 | 249102 | 299.7 | 0.3301 | 26.41 | 5.8073 | 0.8677 | 0.0036 | 0.4715% | 2.2868 | 31.0330% / 1.0469% | 14.0625% |
| fixed_high | 70149 / 131248 | 64644 | 58.2 | 1.4037 | 112.29 | 6.0000 | 1.0000 | 0.0000 | 0.0000% | 0.0000 | 0.0000% / 0.0000% | 0.0000% |

The quality violation is the QAQPrecisionAuditor route-decision
underprecision rate: the fraction of real output-error oracle decisions where
the executed bit is below the smallest safe candidate at the configured
threshold. It is not a task-level quality metric. Each audit made 25,417,728
route decisions, except uncertainty fallback (25,331,712 because fallback
requests are audited through the fixed-high safety lane). Mean violation counts
were 1,085,474 FCFS, 134,271 for scalar/oracle/predicted, 119,566 for
uncertainty fallback, and zero for fixed-high.

## Hotspots

FCFS is dominated by queueing and long mixed-length batches in the high-load
arrival trace. Its p95 latency is 1.10 million ms, compared with 0.56--0.60
million ms for adaptive profile policies. Profile-aware execution improves
throughput by avoiding per-row profile fragmentation, but it does not remove
the substantial QAQ router, estimator, grouping, and dequantized matmul cost.

The uncertainty lane reduces fallback and DP-guard rates while increasing
effective bits from 5.7800 to 5.8073. That tradeoff improves mean throughput
and route-level violation rate on these traces, but the lower batch occupancy
(0.8677) shows that safety routing also fragments batches.

## Time Breakdown

Each measured batch used synchronized CUDA barriers around cached prefill and
decode. Warmups were excluded. TTFT includes queue delay, synchronized
prefill, and first-token selection. TPOT is synchronized decode time divided
by decode forwards after prefill. Tokenization, model loading, JSON writing,
and quality auditing were outside the timed region. Predictor overhead is the
CPU cost of consuming held-out predictor outputs and applying scheduler
decisions; predictor fitting is not included.

## Memory Breakdown

The benchmark used an RTX 3090 with 24 GiB VRAM and --max_batch_size 8.
The continuous sampler observed a maximum resident GPU memory of 24,120 MiB
across the benchmark processes. GPU 5 was externally occupied during the
failed seed-303 fixed-high attempt; no external process was terminated.

## I/O Breakdown

All model, tokenizer, router, estimator, and held-out data inputs were local.
Result JSON and sampler CSV files were written under /tmp; no generated
checkpoints, datasets, or benchmark logs were added to the repository. The
18 result JSON files and 18 sampler files are reproducible from the command
above by substituting the three arrival seeds and six policies.

## GPU Utilization

Continuous nvidia-smi sampling covered all 18 policy/trace runs with 120,225
samples. Sampled GPU utilization was mean 38.53%, median 23%, and p95 100%;
the low mean reflects idle gaps between batches and concurrent sampling across
the long FCFS trace. The sampler's per-file row counts ranged from 85 to
13,090, with observed durations from 98.44 s to 14,710.27 s. This is
device-level telemetry, not a kernel profiler attribution.

## Bottleneck Hypotheses

1. The primary serving bottleneck is QAQ dynamic routing plus queueing, not
   the fixed-high CUDA path. Evidence: adaptive policies deliver 16.90--26.41
   tokens/s versus 112.29 for fixed-high, and FCFS has the largest queue-delay
   tail. Confidence: high.

2. Profile-aware scheduling reduces route fragmentation but still pays the
   router/estimator and grouped execution overhead. Evidence: predicted and
   oracle profiles are close in throughput and quality, while their padding
   fractions remain below 1%. Confidence: medium.

3. The uncertainty fallback lane is promising but needs a less saturated
   arrival regime to separate scheduler fragmentation from safety overhead.
   Evidence: it has the best adaptive throughput and lowest adaptive violation
   rate here, but occupancy falls to 0.8677. Confidence: medium.

Recommendation: next run a synchronized low-load/medium-load arrival sweep with the same native stream and continuous sampling, then use torch.profiler or nsys on one representative repeat to split QAQ router, estimator, grouping, and dequantized matmul costs before optimizing kernels.