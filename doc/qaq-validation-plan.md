# QAQ Validation Plan

This plan defines checks required before using QAQ results in a paper table.
Passing script help or a single smoke test is not enough.

## Correctness Tests

Add or run focused tests for:

- Router shape checks: variable input widths, padding, route ids, optional norm
  feature, and optional estimated-error feature.
- Label generation: relative-error labels from real or fixture dequantized
  weights; multibit chooses the smallest safe bit; binary requires exactly two
  bits.
- Checkpoint roundtrip: save, reload, same config, same logits, same candidate
  bits, same route map.
- Route-map validation: saved route map must match inference route order exactly.
- Runtime modes: `fixed_low`, `fixed_high`, `mlp_binary`, and `mlp_multibit`
  choose only valid bits and respect `max_mem_dict`.
- Confidence fallback: fallback count and fallback fraction are reported
  separately from normal router decisions.
- Stats isolation: generation router stats are not contaminated by finite-logit
  sanity forwards or earlier mode runs.
- Effective-bits math: computed effective bits match direct per-linear weighted
  counts.
- Estimated-error feature: missing estimator artifacts fail loudly when a
  checkpoint requires them.

Tests that need quantized kernels or real model weights should run on the GPU
server. CPU-only unit tests should use small fixtures and avoid fake success
paths for router-label semantics.

## GPU-Server Validation Procedure

Run heavy validation only on the lab GPU server, not the local RTX 4050.

1. Record environment:
   - git commit
   - hostname
   - GPU model
   - CUDA version
   - Python environment
   - `ap_model_path`
   - original `model_path`
   - dataset name and subset

2. Preflight:
   - install Python requirements
   - install editable package
   - install Any-Precision CUDA extension
   - set `CUDA_VISIBLE_DEVICES`
   - run `python scripts/train_qaq_router.py --help`
   - run `python scripts/run_qaq_inference.py --help`

3. Train a small real-data router run:
   - use real calibration data
   - use real Any-Precision weights
   - save checkpoint and JSON metadata
   - inspect label counts for every route
   - reload checkpoint before inference

4. Validate inference:
   - run `fixed_low`, `fixed_high`, `dp_threshold`, and `mlp_multibit`
   - use the same prompts or evaluation set across modes
   - clear stats before each measured generation
   - collect finite-logit checks without contaminating measured router stats

5. Validate quality:
   - run a real evaluator or perplexity command on a documented subset
   - report the exact dataset split, prompt count, context length, and max new
     tokens
   - mark any single-prompt sanity result as `UNVALIDATED`

6. Validate performance only with synchronized timing:
   - warmup count
   - repeat count
   - CUDA synchronize before and after timing
   - batch size / prompt count
   - prompt length
   - max new tokens

Until step 6 is completed, latency and throughput numbers are `UNVALIDATED` and
must not be used as performance claims.

## Required Metrics For A Paper Table

Each row should include:

- model and quantized checkpoint
- candidate bits
- mode or baseline
- dataset / benchmark
- number of examples
- context length and generation length
- quality metric, such as perplexity, exact match, accuracy, or task score
- average selected bit
- effective bits
- per-layer bit histogram artifact path
- fallback count and fallback fraction
- DP guard count and fraction, if implemented
- prefill/decode accounting policy
- router validation accuracy or unsafe-label rate
- finite-logit status
- latency p50 and p95, only if CUDA-synchronized
- tokens/sec, only if CUDA-synchronized
- GPU model and CUDA version
- git commit
- checkpoint path and config JSON path

Any missing metric should be written as `UNVALIDATED`, not inferred.
