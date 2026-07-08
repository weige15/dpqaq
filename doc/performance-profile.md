# Performance Profile

## Benchmark Command

Target: repeated CUDA-synchronized generation benchmark for QAQ DP guard runtime modes on the Any-Precision Llama 3.1 8B checkpoint.

Artifact: `artifacts/qaq_dp_guard_benchmark_20260708_230117`

Follow-up phase-timer artifact: `artifacts/qaq_phase_timers_20260709_011516`

Command:

```bash
OUT=artifacts/qaq_dp_guard_benchmark_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=0
export AP_MODEL_PATH='/nfs/home/s314511048/dpqaq/cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512'
export ROUTER_CHECKPOINT='./checkpoints/qaq_router_llama31_8b_th005.pt'
export ESTIMATOR_RESULTS='./estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam'
/usr/bin/time -v python scripts/benchmark_qaq_modes.py   --ap_model_path "$AP_MODEL_PATH"   --router_checkpoint "$ROUTER_CHECKPOINT"   --estimator_results "$ESTIMATOR_RESULTS"   --bits 3 4 5 6   --modes fixed_low fixed_high qaq dp_threshold_only mlp_multibit_dp_guard dp_threshold   --prompt "Explain mixed-precision inference in one sentence."   --max_new_tokens 16   --warmup 1   --repeat 3   --device cuda   --confidence_threshold 0.6   --output_json "$OUT/benchmark.json"   2>&1 | tee "$OUT/run.log"
```

Environment recorded in `manifest.txt`: commit `681beb1ec2756c335d68e13e0073694ee1da3426`, CUDA_VISIBLE_DEVICES=0, PyTorch `2.4.0+cu124`, GPU `NVIDIA GeForce RTX 3090`, driver `580.159.03`, CUDA reported by `nvidia-smi` as `13.0`. Prompt count was 1, prompt token count was 11, `max_new_tokens=16`, warmup count was 1, repeat count was 3.

Phase-timer follow-up command, run on 2026-07-09 with commit `a9a3f44f68fc487f5ebf9a7c5fb5cfb68d02b945` plus local timer changes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qaq_modes.py --ap_model_path /nfs/home/s314511048/dpqaq/cache/packed/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512 --router_checkpoint ./checkpoints/qaq_router_llama31_8b_th005.pt --estimator_results ./estimator_private_values/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam --bits 3 4 5 6 --modes mlp_multibit_dp_guard --prompt "Explain mixed-precision inference in one sentence." --max_new_tokens 16 --warmup 1 --repeat 3 --device cuda --confidence_threshold 0.6 --output_json artifacts/qaq_phase_timers_20260709_011516/benchmark.json
```

## Baseline Result

No reliable repeated benchmark baseline was found. The closest prior artifact is `artifacts/qaq_dp_guard_sanity_20260708_223930`, but it was a single-run sanity check and is not a statistically comparable benchmark.

## Current Result

All measured modes produced finite logits. The benchmark wrote raw per-repeat timings and aggregate stats to `artifacts/qaq_dp_guard_benchmark_20260708_230117/benchmark.json`.

| Mode | p50 latency (s) | p95 latency (s) | Mean tokens/s | Effective bits mean | Notes |
|---|---:|---:|---:|---:|---|
| `fixed_low` | 0.4995 | 0.5068 | 31.96 | 4.269 | QAQ wrapper, prefill at high precision and decode at low precision |
| `fixed_high` | 0.4986 | 0.5002 | 32.18 | 6.000 | QAQ wrapper, high precision baseline |
| `qaq` | 2.9914 | 3.0077 | 5.35 | 5.468 | 3075 confidence fallbacks over 3 repeats |
| `dp_threshold_only` | 2.0166 | 2.0199 | 7.93 | 5.095 | 10080 DP threshold decisions, 4227 high-bit decisions |
| `mlp_multibit_dp_guard` | 3.9109 | 3.9223 | 4.09 | 5.478 | 3075 confidence fallbacks and 147 DP guard triggers |
| `dp_threshold` | 1.0434 | 1.0452 | 15.35 | 4.411 | Original DP-LLM wrapper baseline |

The process-level `/usr/bin/time -v` output reported 1:19.07 wall-clock time for the whole benchmark command, 75.09 user CPU seconds, 12.07 system CPU seconds, 110% CPU utilization, and 9,572,424 KB maximum resident set size.

The phase-timer follow-up measured only `mlp_multibit_dp_guard` with 1 warmup, 3 repeats, and 16 generated tokens per repeat. It produced finite logits and wrote raw stats to `artifacts/qaq_phase_timers_20260709_011516/benchmark.json`. The measured p50 latency was 5.1616 s, p95 latency was 5.1891 s, mean throughput was 3.13 tokens/s, effective bits mean was 5.478, average selected bit was 5.607, with 3,075 confidence fallbacks and 147 DP guard triggers over 17,472 routed linear-token operations. Because CUDA event phase timing is enabled inside every guarded linear call, this run should be used for internal attribution rather than direct speed comparison against the no-timer mode benchmark.

## Hotspots

Mode-level timing shows the runtime overhead is concentrated in dynamic routing paths rather than the fixed precision paths. `qaq` is about 6.0x slower than `fixed_high` by p50 latency. `dp_threshold_only` is about 4.0x slower than `fixed_high`. `mlp_multibit_dp_guard` is about 7.8x slower than `fixed_high` and about 1.3x slower than plain `qaq`.

The combined guard mode performs both MLP router inference and DP estimator/threshold work, and its p50 latency is the largest measured value. The phase-timer follow-up shows router time is the largest measured internal phase for `mlp_multibit_dp_guard`, followed by grouping and dequantized matmul. No kernel-level profiler was run in this pass.

## Time Breakdown

Generation timing excludes tokenization and uses CUDA synchronization before and after each measured `generate` call. Each mode used 1 warmup and 3 measured repeats, generating 16 new tokens per repeat.

The fastest measured modes were `fixed_high` and `fixed_low` at roughly 0.50 s p50. The original DP-LLM threshold baseline was 1.04 s p50. QAQ MLP routing was 2.99 s p50. QAQ threshold-only routing was 2.02 s p50. QAQ MLP plus DP guard was 3.91 s p50.

`QAQDPLLM_Linear` phase-timer totals from the 16-token, 3-repeat guarded follow-up were:

| Phase | CUDA event total (ms) | Wall total (ms) | Count |
|---|---:|---:|---:|
| router | 5,578.72 | 5,747.75 | 10,080 |
| estimator | 766.18 | 1,081.51 | 10,080 |
| grouping | 1,743.14 | 2,217.04 | 20,160 |
| dequant_matmul | 1,160.92 | 1,334.34 | 10,080 |
| total | 13,578.90 | 13,809.73 | 10,080 |

Using CUDA event time, router work accounts for about 41.1% of the measured guarded linear total, grouping about 12.8%, dequantized matmul about 8.6%, and estimator work about 5.6%. The remaining measured total includes uninstrumented work inside the guarded linear path and nesting/measurement overhead.

CUDA event timings measure queued GPU work; wall timings also include Python launch overhead and CPU-side synchronization such as grouping decisions. Model loading, JSON writing, and manifest creation were included in the process wall time but not in per-mode latency measurements.

## Memory Breakdown

Peak CUDA memory reported by the benchmark script was stable across modes. QAQ fixed and MLP modes reported about 7.28 GiB max allocated and 7.35 GiB max reserved. QAQ DP threshold modes reported about 7.33 GiB max allocated and 7.40 GiB max reserved. The original DP threshold baseline reported about 7.33 GiB max allocated and 7.38 GiB max reserved.

The `/usr/bin/time -v` maximum resident set size was about 9.13 GiB. No CUDA out-of-memory or host swapping was observed.

## I/O Breakdown

The benchmark loaded the AP model checkpoint, router checkpoint (`15M`), and estimator files: `jl_d.pt` (`56M`), `linear_reg_d.pt` (`22K`), `max_mem_dict.pt` (`3.0K`), and `T_d.pt` (`58K`). Runtime output files were `benchmark.json` (`2.0M`), `run.log` (`121K`), and `manifest.txt` (`5.9K`).

The `/usr/bin/time -v` output reported zero major page faults and 14,760 filesystem output blocks. File I/O was visible during model and estimator loading, but per-mode generation timing did not include explicit I/O work beyond normal runtime logging.

## GPU Utilization

The run used GPU 0, an NVIDIA GeForce RTX 3090 with 24 GiB VRAM. The manifest `nvidia-smi` snapshot after completion showed no running processes and 1 MiB used, so it does not capture active utilization during the measured repeats. CUDA memory telemetry from the benchmark script indicates peak reserved memory was about 7.35-7.40 GiB.

The benchmark did not sample GPU utilization during the active generation windows and did not run `nsys`, `ncu`, or `torch.profiler`, so GPU-bound versus CPU/synchronization-bound attribution remains uncertain.

## Bottleneck Hypotheses

1. Hypothesis: per-token QAQ MLP routing and grouped per-bit execution dominate `qaq` latency. Evidence: `qaq` p50 latency was 2.9914 s versus 0.4986 s for `fixed_high`, while effective bits were lower than fixed high. Confidence: High. Next measurement: use `torch.profiler` around one QAQ repeat to split router MLP, estimator, dequantization, matmul, and Python grouping overhead. Hand off to `optimization-diagnosis`: yes.

2. Hypothesis: DP threshold estimator work in the QAQ wrapper is materially slower than the original DP-LLM threshold implementation. Evidence: `dp_threshold_only` p50 was 2.0166 s while original `dp_threshold` p50 was 1.0434 s. Confidence: Medium. Next measurement: profile estimator calls and compare QAQ synchronous estimator code against DP-LLM async residual estimator behavior. Hand off to `optimization-diagnosis`: yes.

3. Hypothesis: `mlp_multibit_dp_guard` is dominated by router and grouped execution overhead more than by the DP estimator itself. Evidence: guarded no-timer p50 was 3.9109 s, slower than both `qaq` at 2.9914 s and `dp_threshold_only` at 2.0166 s; it performed 147 DP guard triggers and 10,080 threshold decisions across 3 repeats. The repeated phase-timer follow-up measured 5,578.72 ms CUDA time in router work, 766.18 ms in estimator work, 1,743.14 ms in grouping, and 1,160.92 ms in dequant/matmul across the guarded run. Confidence: High. Next measurement: run `torch.profiler` or `nsys` on one guarded repeat to validate whether event-level phase timing matches kernel-level attribution and to identify the exact router kernels and Python synchronization sites. Hand off to `optimization-diagnosis`: yes.

Recommendation: run `optimization-diagnosis` next with `torch.profiler` or `nsys` focused first on the `mlp_multibit_dp_guard` router and grouping phases, because the new internal phase timers identify router work as the largest measured component but do not yet provide kernel-level attribution.
