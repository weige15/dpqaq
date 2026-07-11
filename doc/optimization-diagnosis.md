# Optimization Diagnosis

## Baseline

Known reliable prior benchmark baseline: Unknown. The closest latency references are in-run baselines from `artifacts/qaq_dp_guard_benchmark_20260708_230117/benchmark.json`.

* benchmark command: `CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qaq_modes.py --ap_model_path /nfs/home/s314511048/dpqaq/cache/packed/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512 --router_checkpoint ./checkpoints/qaq_router_llama31_8b_th005.pt --estimator_results ./estimator_private_values/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam --bits 3 4 5 6 --modes fixed_low fixed_high qaq dp_threshold_only mlp_multibit_dp_guard dp_threshold --prompt "Explain mixed-precision inference in one sentence." --max_new_tokens 16 --warmup 1 --repeat 3 --device cuda --confidence_threshold 0.6 --output_json artifacts/qaq_dp_guard_benchmark_20260708_230117/benchmark.json`
* evaluator command: Unknown
* baseline score: Unknown
* baseline runtime: `fixed_high` p50 0.4986 s, p95 0.5002 s, mean 32.18 tokens/s; original `dp_threshold` p50 1.0434 s, p95 1.0452 s, mean 15.35 tokens/s
* baseline memory usage: `fixed_high` max CUDA allocated 7.28 GiB, max reserved 7.35 GiB; original `dp_threshold` max CUDA allocated 7.33 GiB, max reserved 7.38 GiB
* baseline quality metric: finite logits were true; real quality metric Unknown
* baseline environment: commit `681beb1ec2756c335d68e13e0073694ee1da3426`, PyTorch `2.4.0+cu124`, GPU `NVIDIA GeForce RTX 3090`, `CUDA_VISIBLE_DEVICES=0`, prompt count 1, prompt tokens 11, `max_new_tokens=16`, warmup 1, repeat 3

## Current Performance

Current target mode is `mlp_multibit_dp_guard`, because it is the slowest QAQ mode and includes both QAQ MLP routing and DP threshold guarding.

* current score: Unknown
* current runtime: p50 3.9109 s, p95 3.9223 s, mean 4.09 tokens/s in `artifacts/qaq_dp_guard_benchmark_20260708_230117/benchmark.json`
* current memory usage: max CUDA allocated 7.33 GiB, max reserved 7.40 GiB
* current quality metric: finite logits were true; real quality metric Unknown
* current environment: same benchmark environment as baseline for the no-phase-timer run
* logs or files inspected: `README.md`, `doc/performance-profile.md`, `doc/qaq-dynamic-batching-design.md`, `doc/qaq-router.md`, `doc/repo-intake.md`, `doc/qaq-validation-plan.md`, `scripts/benchmark_qaq_modes.py`, `any_precision/modules/QAQDPLLM_Linear.py`, `any_precision/modules/QAQRouter.py`, `any_precision/modules/DPLLM_Linear.py`, `artifacts/qaq_dp_guard_benchmark_20260708_230117/benchmark.json`, `artifacts/qaq_dp_guard_benchmark_20260708_230117/manifest.txt`, `artifacts/qaq_dp_guard_benchmark_20260708_230117/run.log`, `artifacts/qaq_phase_timers_20260709_011516/benchmark.json`, and `artifacts/qaq_phase_timers_20260709_011516/manifest.txt`

The phase-timer follow-up in `artifacts/qaq_phase_timers_20260709_011516/benchmark.json` measured `mlp_multibit_dp_guard` with phase timers enabled. Its p50 latency was 5.1616 s, but that run should be used for attribution, not direct speed comparison, because phase timing instruments every guarded linear call. Aggregated CUDA event time across the guarded run was:

| Phase | CUDA Time | Count | Fraction Of Timed Total |
| --- | ---: | ---: | ---: |
| router | 5.5787 s | 10,080 | 41.1% |
| grouping | 1.7431 s | 20,160 | 12.8% |
| dequant_matmul | 1.1609 s | 10,080 | 8.5% |
| estimator | 0.7662 s | 10,080 | 5.6% |
| total | 13.5789 s | 10,080 | 100.0% |

## Gap

Compared with the `fixed_high` in-run reference, `mlp_multibit_dp_guard` is slower by 3.4123 s p50 and runs at 7.84x the p50 latency. Mean throughput drops from 32.18 tokens/s to 4.09 tokens/s, an 87.3% reduction.

Compared with plain QAQ MLP routing (`qaq`), `mlp_multibit_dp_guard` is slower by 0.9195 s p50 and runs at 1.31x the p50 latency. Compared with original `dp_threshold`, it is slower by 2.8675 s p50 and runs at 3.75x the p50 latency.

* absolute difference: +3.4123 s p50 versus `fixed_high`; +0.9195 s p50 versus `qaq`
* relative difference: 7.84x versus `fixed_high`; 1.31x versus `qaq`
* expected target: Unknown; a credible near-term target is to reduce `mlp_multibit_dp_guard` without losing finite logits, effective-bit accounting, confidence fallback counts, or DP guard counts
* gap type: latency and throughput

## Bottleneck

Observed symptoms:

* Dynamic QAQ modes are much slower than fixed precision modes even though they often use lower effective bits.
* `qaq` p50 is 2.9914 s, about 6.00x slower than `fixed_high`.
* `mlp_multibit_dp_guard` p50 is 3.9109 s, about 7.84x slower than `fixed_high`.
* Memory is nearly flat across modes, so the gap does not look like memory capacity pressure.

Supporting evidence:

* The phase-timer run attributes 41.1% of timed guarded CUDA event time to the `router` phase, the largest measured internal phase.
* The DP estimator phase is much smaller at 5.6% of timed guarded CUDA event time.
* The quantized/dequantized matmul phase is also smaller at 8.5% of timed guarded CUDA event time.
* `QAQDPLLM_Linear._choose_router_bits()` invokes the shared MLP router for each routed linear call, then applies softmax/confidence fallback and maps class ids to bit widths.
* The routed path also performs CPU-visible grouping through `chosen_bits.detach().cpu().tolist()`, but the measured grouping phase is smaller than router time.
* Original `DPLLM_Linear` avoids QAQ MLP inference in decode, matching its lower p50 latency of 1.0434 s.

Rejected explanations:

* Pure quantized matmul cost is unlikely to be the primary bottleneck because `fixed_high` and `fixed_low` both run near 0.5 s p50, while dynamic MLP routing is much slower.
* DP threshold estimation alone is unlikely to be the primary bottleneck because the phase timer reports estimator at 5.6% of timed guarded CUDA event time, and plain QAQ without the DP guard is already 6.00x slower than `fixed_high`.
* File I/O is unlikely to explain per-mode latency because generation timing is synchronized around `model.generate`, and `/usr/bin/time` reported zero major page faults.
* CUDA memory capacity is unlikely to explain the gap because peak reserved memory stays around 7.35-7.40 GiB across modes on a 24 GiB RTX 3090.

Most likely root cause:

Per-routed-linear QAQ MLP router execution during decode is the single most likely bottleneck. The current evidence points to many small router invocations and associated feature construction/kernel launch overhead dominating guarded QAQ latency more than DP estimation, grouped matmul, or memory pressure.

## Candidate Optimizations

| Candidate | Target Bottleneck | Expected Impact | Risk | Complexity | Verification |
| --------- | ----------------- | --------------: | ---: | ---------: | ------------ |
| Batch or fuse QAQ router inference across routed linear calls in decode | Per-call router MLP overhead | Medium to high if profiler confirms router kernels/launches dominate | Medium | High | `nsys profile` followed by the same synchronized benchmark |
| Keep bit grouping decisions GPU-local and avoid per-call CPU list conversion | CPU-GPU synchronization and grouping overhead | Medium if profiler shows synchronization around grouping | Medium | Medium | Compare grouped path p50 and profiler CPU/GPU sync events |
| Avoid unnecessary DP guard work when router confidence and threshold state make the guard decision redundant | Secondary DP guard overhead | Low to medium because estimator timing is smaller | Medium | Medium | Compare DP guard trigger/fallback counts and p50 latency |

## Risk

Optimizing the wrong thing could make the research result less trustworthy.

* correctness regression risk: router batching or fusion must preserve per-route ids, valid-bit clamping, confidence fallback, DP guard max semantics, and per-layer statistics
* overfitting to validation cases: the current benchmark has one prompt, 11 prompt tokens, 16 generated tokens, and only three measured repeats
* benchmark noise: the phase-timer run has instrumentation overhead and was run at a different commit with local timer changes
* increased complexity: batching router calls across linears may complicate clear per-layer accounting and rollback
* worse memory usage: larger router batches may add activation buffers or temporary logits
* slower performance on hidden tests: an optimization tuned for batch size 1 decode may not help longer prompts, different output lengths, or serving batches
* loss of reproducibility: the benchmark artifact records dirty worktrees, so new runs must record commit and dirty status carefully

## Expected Impact

If the profiler confirms the phase-timer attribution, reducing router invocation overhead is expected to improve `mlp_multibit_dp_guard` latency materially. A cautious target is a 15-30% p50 latency reduction for the guarded mode, because router CUDA event time is the largest measured phase but not the entire runtime. Larger gains are possible only if kernel launch overhead and CPU synchronization around router calls are also dominant in the profiler.

This optimization is not expected to improve model quality directly. It should preserve effective bits, average selected bit, fallback fraction, DP guard trigger fraction, and finite-logit status.

## Recommended First Optimization

### Hypothesis

`mlp_multibit_dp_guard` latency is dominated by many small per-routed-linear QAQ MLP router invocations during decode.

### Optimization

First validate the hypothesis with a GPU-server profiler run, then test one scoped implementation that batches or fuses QAQ router inference across routed decode calls without changing routing semantics.

### Why First

The router phase is the largest measured internal phase at 41.1% of timed guarded CUDA event time. It is larger than grouping, dequantized matmul, and DP estimation, and plain QAQ is already much slower than fixed precision before the DP guard is added.

### Implementation Scope

Smallest possible implementation scope after profiler confirmation:

* `any_precision/modules/QAQDPLLM_Linear.py`
* `any_precision/modules/QAQRouter.py`
* optional benchmark-only instrumentation in `scripts/benchmark_qaq_modes.py`

Do not change CUDA kernels, DP-LLM baseline behavior, public CLI semantics, router checkpoint format, or reported router-stat definitions.

### Verification Command

Run this profiler command on the GPU server before implementation and again after implementation, with a distinct `--output` name for the second run:

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx,osrt --sample=cpu --cuda-memory-usage=true --force-overwrite=true --output artifacts/qaq_nsys_router_guard python scripts/benchmark_qaq_modes.py --ap_model_path /nfs/home/s314511048/dpqaq/cache/packed/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512 --router_checkpoint ./checkpoints/qaq_router_llama31_8b_th005.pt --estimator_results ./estimator_private_values/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam --bits 3 4 5 6 --modes mlp_multibit_dp_guard --prompt "Explain mixed-precision inference in one sentence." --max_new_tokens 16 --warmup 1 --repeat 1 --device cuda --confidence_threshold 0.6 --no_phase_timers --output_json artifacts/qaq_nsys_router_guard.json
```

### Success Criteria

Profiler success criteria:

* Router MLP kernels, launch overhead, or adjacent feature construction account for the largest actionable share of `mlp_multibit_dp_guard` runtime.
* The profiler result is consistent with the phase-timer finding that router work is larger than estimator and dequantized matmul work.

Implementation success criteria:

* `mlp_multibit_dp_guard` p50 latency improves by at least 15% on the same benchmark command.
* Finite logits remain true.
* Effective bits, average selected bit, fallback fraction, DP guard trigger fraction, and per-layer bit histograms remain semantically equivalent for the same prompt and checkpoint.
* `fixed_low`, `fixed_high`, `qaq`, `dp_threshold_only`, and original `dp_threshold` still run through `scripts/benchmark_qaq_modes.py`.

### Rollback Condition

Abandon or revert this optimization if the profiler shows router work is not the largest actionable bottleneck, if p50 latency improves by less than 5%, or if router decisions/statistics change for the same checkpoint and prompt.

## Handoff to implementation-loop-manager

```text
Use `implementation-loop-manager` in optimization mode.

Goal:
Reduce `mlp_multibit_dp_guard` p50 generation latency without changing QAQ routing semantics or reported statistics.

Hypothesis:
`mlp_multibit_dp_guard` latency is dominated by many small per-routed-linear QAQ MLP router invocations during decode.

Allowed change scope:
`any_precision/modules/QAQDPLLM_Linear.py`, `any_precision/modules/QAQRouter.py`, and optional benchmark-only instrumentation in `scripts/benchmark_qaq_modes.py`.

Do not change:
CUDA kernels, original DP-LLM behavior, router checkpoint format, public CLI semantics, label semantics, confidence fallback semantics, DP guard max semantics, or router-stat definitions.

Verification command:
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda,nvtx,osrt --sample=cpu --cuda-memory-usage=true --force-overwrite=true --output artifacts/qaq_nsys_router_guard python scripts/benchmark_qaq_modes.py --ap_model_path /nfs/home/s314511048/dpqaq/cache/packed/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512 --router_checkpoint ./checkpoints/qaq_router_llama31_8b_th005.pt --estimator_results ./estimator_private_values/anyprec-\(Meta-Llama-3.1-8B\)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam --bits 3 4 5 6 --modes mlp_multibit_dp_guard --prompt "Explain mixed-precision inference in one sentence." --max_new_tokens 16 --warmup 1 --repeat 1 --device cuda --confidence_threshold 0.6 --no_phase_timers --output_json artifacts/qaq_nsys_router_guard.json

Success criteria:
The profiler confirms router work is the largest actionable bottleneck, and the optimized guarded mode improves p50 latency by at least 15% on the synchronized benchmark while preserving finite logits, effective bits, fallback counts, DP guard counts, and per-layer bit histograms.

Rollback condition:
Revert or abandon if the profiler does not rank router work as the largest actionable bottleneck, if p50 latency improves by less than 5%, or if same-prompt router decisions/statistics change.
```
