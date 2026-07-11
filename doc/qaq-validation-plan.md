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

## Held-Out QAQ Quality Evaluation

Use `scripts/evaluate_qaq_heldout.py` for the quality/routing comparison. It
always evaluates these modes through the same loaded QAQ model:

- `fixed_low`
- `fixed_high`
- `dp_threshold_only`
- `mlp_multibit`
- `mlp_multibit_dp_guard`

The command is CUDA-only and writes the artifact atomically after every mode
finishes. A CPU invocation fails before model or dataset loading and does not
write an artifact. This command does not time execution and its artifact must
not be used for latency or throughput claims.

### Documented held-out subset

The default paper-development subset is WikiText-2
`Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`. The tokenizer encodes the full joined
test text with `add_special_tokens=False`; evaluation selects the first 16
non-overlapping 512-token windows (`--dataset_start 0 --num_examples 16`).
The router trainer uses only the named dataset's `train` split, so this
evaluation split is held out. The artifact records a SHA-256 digest of every
example's token ids and a combined subset digest, making cross-mode and
cross-run subset identity checkable.

The alternative `--dataset c4_new` uses the first 1,100 rows of
`allenai/c4` validation shard
`en/c4-validation.00000-of-00008.json.gz`, joined with spaces before the same
non-overlapping token-window selection. Do not mix datasets or window settings
within a comparison.

### Lab-GPU command

Run only after manually checking GPU availability. Replace every angle-bracket
path with the real lab-server path:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qaq_heldout.py \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --dataset wikitext2 \
  --context_length 512 \
  --dataset_start 0 \
  --num_examples 16 \
  --bits 3 4 5 6 \
  --confidence_threshold 0.6 \
  --fallback_bits 1 \
  --device cuda:0 \
  --output_json artifacts/qaq-heldout-wikitext2-test-16x512.json
```

The router checkpoint must have `label_mode=multibit`, a real
`error_threshold`, and candidate bits matching `--bits`. The estimator
directory must contain the DP estimator artifacts, `max_mem_dict.pt`, and
`T_d.pt` needed by both DP modes.

### Precision-label semantics

For each real activation at each quantized linear, the evaluator dequantizes
the available candidate weights and computes:

```text
rel_error_b = ||W_ref x - W_b x|| / (||W_ref x|| + 1e-8)
```

The required bit is the smallest available bit whose relative error is at most
the router checkpoint's training threshold; the highest available bit is the
reference. Under-precision means the actual executed bit is below this
required bit. Over-precision means it is above. These labels are recomputed on
each mode's own activations rather than copied from training data.

### Artifact schema

The top-level JSON object has:

- `schema_version`: currently `qaq_heldout_eval_v1`.
- `validation_status`: `REAL_GPU_HELDOUT` only after a successful CUDA run.
- `environment`: timestamp, git commit, hostname, Python/PyTorch/CUDA
  and library versions, GPU name, `CUDA_VISIBLE_DEVICES`, argv, dirty-worktree
  status, and SHA-256 hashes of the evaluator and QAQ runtime source files.
- `inputs`: model, router, estimator, and tokenizer paths; candidate bits;
  error and confidence thresholds; fallback increment; prefill policy.
- `dataset`: Hugging Face identity, split/data file, text joining and window
  policies, start/count/context length, per-run token digest, and training
  dataset/split provenance.
- `metric_definitions`: exact definitions used by the evaluator.
- `modes`: one object for each of the five required modes.

Each mode object contains aggregate `mean_nll`, `perplexity`, finite-logit status, their deltas
versus `fixed_high`, target-token count, `runtime_stats`,
`precision_metrics`, `per_layer_precision_metrics`, and `examples`.
Runtime stats include effective bits, average selected bit, per-layer bit
counts, fallback count/fraction, and DP guard count/fraction. Each example
contains its token digest, mean NLL, perplexity, NLL/perplexity deltas versus
the same fixed-high example, and under/over/exact precision counts and rates.

A real artifact is necessary completion evidence. Passing CPU unit tests,
`--help`, or compilation alone leaves held-out model-quality validation
`NOT COMPLETE`.
