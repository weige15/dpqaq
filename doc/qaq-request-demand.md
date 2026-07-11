# QAQ Request-Level Precision Demand

## Purpose

This workflow tests two serving hypotheses on held-out requests:

1. Whether knowing each request's observed QAQ precision profile gives a
   profile-aware batching scheduler an oracle precision-work advantage.
2. Whether prompt and fixed-high prefill features can predict minimum safe
   precision or the observed QAQ profile before decode.

It does not claim measured latency or throughput improvement.

## Dataset Definition

The validated artifact uses 32 non-overlapping requests from
`Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test`. Each
request contains:

- 128 prompt tokens available before decode.
- 64 held-out continuation tokens used only for teacher-forced quality and
  observed-profile targets.
- The same token ids for every fixed and QAQ mode.

The router was trained on C4 train data, so this WikiText-2 test subset is held
out from router training. The combined request-token SHA-256 is
`7fa109c6d31332d20604eb8631167c1e52a149554753b1bd2b6e42c9160abcc5`.

## Quality and Demand Semantics

Quality is mean next-token NLL over the 64 continuation tokens. Prompt labels
are masked, while the prompt remains model context. Every request runs through:

- `fixed_low` (requested bit 3)
- `fixed_4`
- `fixed_5`
- `fixed_high` (requested bit 6)
- `dp_threshold_only`
- `mlp_multibit`
- `mlp_multibit_dp_guard`

Intermediate fixed modes use the real Any-Precision kernels. A route whose
`max_mem_dict` ceiling is lower than the requested bit uses its highest valid
bit at or below the request.

The minimum safe precision is the smallest requested fixed bit whose
continuation NLL is no more than `0.02` above fixed-high NLL. The artifact
contains all fixed-bit deltas, not only the selected label.

## Prompt and Prefill Features

Predictors may consume only values available before decode:

- prompt length, token diversity, token-id moments, and token entropy
- decoded character, whitespace, digit, alphabetic, punctuation, and line
  statistics
- fixed-high prompt NLL
- last-prompt-token entropy, top-1 probability, and probability margin
- final hidden-state norm and magnitude summaries from fixed-high prefill

No continuation tokens, continuation quality, QAQ decisions, fallback counts,
or observed profiles are included as predictor inputs.

## Observed QAQ Profile

For each QAQ mode, every record contains:

- effective and average selected bits
- fallback and DP-guard counts/fractions
- per-route bit counts, expected bits, and majority bits for all 224 routes
- an eight-dimensional coarse profile formed by averaging expected route bits
  within each consecutive four-layer group

Profiles are observed during the teacher-forced prompt-plus-continuation
forward. They are targets for analysis, not pre-decode inputs.

## GPU Collection Command

Check GPU availability manually and keep device visibility explicit:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 /nfs/home/s314511048/.venv/bin/python \
  scripts/build_qaq_request_demand_dataset.py \
  --ap_model_path 'cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512' \
  --router_checkpoint checkpoints/qaq_router_llama31_8b_th005.pt \
  --estimator_results 'estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam' \
  --tokenizer_path 'cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512' \
  --dataset wikitext2 \
  --dataset_start 0 \
  --num_requests 32 \
  --prompt_length 128 \
  --continuation_length 64 \
  --bits 3 4 5 6 \
  --safe_nll_delta 0.02 \
  --profile_layer_group_size 4 \
  --confidence_threshold 0.6 \
  --fallback_bits 1 \
  --device cuda:0 \
  --output_jsonl artifacts/qaq-request-demand-wikitext2-32x128p64c.jsonl \
  --summary_json artifacts/qaq-request-demand-wikitext2-32x128p64c-summary.json
```

## Offline Analysis Command

The scheduler oracle uses exact pair batching. All 496 possible pairs are
considered by a binary MILP, so the reported result is exact for batch size 2.
A batch-size-4 set-partition run exceeded the 120-second solver limit and was
not used as evidence.

```bash
/nfs/home/s314511048/.venv/bin/python \
  scripts/analyze_qaq_request_demand.py \
  --dataset_jsonl artifacts/qaq-request-demand-wikitext2-32x128p64c.jsonl \
  --output_json artifacts/qaq-request-demand-wikitext2-32x128p64c-analysis.json \
  --profile_mode mlp_multibit_dp_guard \
  --batch_size 2 \
  --cv_folds 5 \
  --seed 0 \
  --trees 300 \
  --milp_time_limit_s 120 \
  --material_oracle_advantage 0.01 \
  --predictability_mae_improvement 0.10 \
  --predictability_r2 0.10
```

The oracle cost is
`sum(batch_size * mean(componentwise maximum group bit))`. It is a
precision-work proxy. It does not model CUDA kernel efficiency, queueing,
memory transfer, or latency.

Predictors use shuffled request-level five-fold cross-validation. A profile is
reported as predictably supported only when profile MAE improves on the
fold-training mean by at least 10% and variance-weighted R² is at least 0.10.

## Artifact Schemas

Each `qaq_request_demand_v1` JSONL record contains request token hashes,
prompt features, `quality_by_mode`, `minimum_safe_precision`, and
`observed_qaq_profiles`. Raw prompt and continuation text/tokens are not
stored.

The collection summary has status `REAL_GPU_REQUEST_DEMAND`, exact model and
checkpoint paths, CUDA/software versions, source hashes, dataset identity,
subset hash, demand distribution, and aggregate quality/effective-bit metrics.

The `qaq_request_demand_analysis_v1` artifact contains the dataset hash,
analysis source hash, exact oracle batches/costs, cross-validated predictions,
metrics, feature importance, conservative verdicts, and limitations.

## Validated Results

Minimum safe precision at ΔNLL ≤ 0.02:

- bit 4: 5/32 requests
- bit 5: 18/32 requests
- bit 6: 9/32 requests

Exact observed-profile pair oracle:

- improvement over FCFS: 0.1001%
- improvement over scalar effective-bit sorting: 0.0544%
- material advantage threshold: 1%
- verdict: positive but not material on this subset

Pre-decode prediction:

- safe-bit accuracy: 46.88% versus 56.25% majority baseline
- safe-bit balanced accuracy: 31.48%
- safe-bit verdict: not predictable
- group-profile MAE: 0.01903 bits versus 0.02061 mean baseline
- group-profile MAE improvement: 7.65%
- profile variance-weighted R²: 0.0433
- effective-bit R²: -0.1039
- profile verdict: weak signal, not established predictability

## Limitations

- Thirty-two requests from one dataset are adequate for a first falsification
  test, not a general serving claim.
- The exact oracle result is for pair batching; larger exact set partitioning
  needs a stronger solver or more time.
- The precision-work proxy is not measured throughput or latency.
- Teacher forcing supplies observed continuation profiles; production decode
  profiles may differ.
- Results use the installed Transformers/Datasets stack recorded in the
  artifacts and include a non-fatal Accelerate device-map warning.

## Preregistered Large Collection

The preregistered collector uses document-level deterministic manifests, pinned
dataset revisions, atomic validated JSONL shards, and resumable execution. It
writes to a dedicated directory and does not read, replace, or append to the
32-request pilot artifact.

Pinned sources:

- WikiText-2 test revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`.
- C4 validation shard `en/c4-validation.00000-of-00008.json.gz` at revision
  `607bd4c8450a42878aa9ddc051a65a055450ef87`.

CPU-only manifest preflight:

```bash
HF_DATASETS_OFFLINE=1 python scripts/build_qaq_request_demand_dataset.py \
  --protocol preregistered_large_v1 \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --datasets wikitext2 c4_new \
  --bits 3 4 5 6 \
  --manifest_only \
  --local_files_only \
  --output_dir artifacts/qaq-request-demand-preregistered-v1
```

Real Device 0 collection:

```bash
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 \
  python scripts/build_qaq_request_demand_dataset.py \
  --protocol preregistered_large_v1 \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --datasets wikitext2 c4_new \
  --bits 3 4 5 6 \
  --safe_nll_delta 0.02 \
  --profile_layer_group_size 4 \
  --confidence_threshold 0.6 \
  --fallback_bits 1 \
  --shard_size 8 \
  --device cuda:0 \
  --local_files_only \
  --output_dir artifacts/qaq-request-demand-preregistered-v1
```

Re-running the same command validates and skips every completed shard. A
different manifest, source hash, input hash, or collection configuration fails
rather than mixing results. An interruption can lose only the uncommitted
in-memory shard; finalized shards are never overwritten.

Post-run validation without loading the model:

```bash
HF_DATASETS_OFFLINE=1 python scripts/build_qaq_request_demand_dataset.py \
  --protocol preregistered_large_v1 \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --datasets wikitext2 c4_new \
  --bits 3 4 5 6 \
  --validate_only \
  --local_files_only \
  --output_dir artifacts/qaq-request-demand-preregistered-v1
```

The output contains document/request manifests, eight-request JSONL shards,
one validation sidecar per shard, a run manifest, per-dataset summaries, and
`combined-summary.json`. Records contain hashes and numeric features/metrics
only; raw source, prompt, and continuation text and token arrays are prohibited
by shard validation.


## Preregistered Offline Analysis

The completed collection is frozen by
`artifacts/qaq-request-demand-preregistered-v1-freeze.json`. The analyzer
verifies that recursive file manifest before and after analysis and writes only
to a separate output directory.

```bash
python scripts/analyze_qaq_request_demand_preregistered.py \
  --collection_dir artifacts/qaq-request-demand-preregistered-v1 \
  --freeze_manifest artifacts/qaq-request-demand-preregistered-v1-freeze.json \
  --output_json artifacts/qaq-request-demand-preregistered-v1-analysis/analysis.json \
  --bootstrap_replicates 10000 \
  --bootstrap_seed 1729 \
  --predictor_seeds 17 29 43 \
  --trees 300 \
  --group_folds 5
```

This analysis uses document-grouped development diagnostics, the frozen pooled
development/calibration split, separate WikiText-2 and C4 test reports, all
three registered predictor seeds, and document-cluster bootstrap intervals.
H1 and H2 are evaluated from the collected endpoints. H3 reports a definitive
quality-gate failure when present and separately marks route-level
under-precision unavailable because the collection lacks per-decision required
bits. H4 remains not run until real online GPU replay exists. The dirty-source
precommit deviation is disclosed in both the protocol and analysis output.
