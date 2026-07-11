# QAQ Request-Demand Preregistered Experimental Protocol

Status: **PREREGISTRATION DRAFT - NO CONFIRMATORY RUNS MAY START UNTIL THIS
DOCUMENT IS COMMITTED**

Protocol date: 2026-07-11

This document freezes the confirmatory protocol for request-level precision
demand, pre-decode demand prediction, and precision-aware scheduling in `dpqaq`.
It is an experimental contract, not an implementation claim. Any change after
the first protocol commit must be recorded in the deviations section before
results are inspected.

## Research Questions And Hypotheses

The experiment addresses four questions.

1. **H1, heterogeneous demand.** Requests have materially different minimum
   safe fixed precisions and QAQ route profiles after prompt/continuation length
   and source document are controlled.
2. **H2, pre-decode predictability.** Features available by the end of a
   fixed-6-bit prefill predict minimum safe precision and the guarded-MLP coarse
   route profile better than training-fold constant baselines on unseen
   documents.
3. **H3, guard safety.** `mlp_multibit_dp_guard` preserves continuation quality
   and reduces unsafe routing relative to unguarded `mlp_multibit`, without
   collapsing to fixed 6-bit execution.
4. **H4, scheduling value.** A deployable scheduler using predicted request
   demand improves throughput over ordinary FCFS dynamic batching without
   violating preregistered quality or tail-latency gates. Oracle-profile
   scheduling is an upper-bound diagnostic and cannot by itself support H4.

All four hypotheses are confirmatory only on the fresh test partitions defined
below. Existing artifacts informed this design and are pilot evidence only.

## Audit Of Current Artifacts

| Artifact or path | What is real and useful | Why it is not confirmatory evidence |
| --- | --- | --- |
| `artifacts/qaq-request-demand-wikitext2-32x128p64c.jsonl` and summary | 32 real GPU requests; fixed 3/4/5/6, DP threshold, MLP, and guarded MLP; teacher-forced NLL; real route profiles | One dataset, one 128/64 length cell, first contiguous corpus windows, and no document identity |
| `artifacts/qaq-request-demand-wikitext2-32x128p64c-analysis.json` | Exact pair oracle and request-level predictor analysis | Only seed 0; shuffled request-level K-fold CV; adjacent or same-document windows can cross folds; only 32 requests |
| `artifacts/qaq-heldout-wikitext2-test-16x512.json` | Real CUDA held-out quality and real output-error precision labels | 16 contiguous windows from a document-concatenated stream; only WikiText-2; not request/prompt-continuation evaluation |
| `artifacts/qaq_mixed_trace_20260709_050923/qaq_trace.jsonl` | 200 real single-request guarded-MLP generation traces over four prompt categories | Hand-assembled workload, at most 8 decode tokens, variable early stopping, no quality labels, no pre-decode predictions, and prompt length is confounded with workload |
| `artifacts/qaq_mixed_trace_20260709_050923/qaq_batching_simulation.json` | Deterministic simulator exercise over five policies | Uses observed post-decode profiles, a heuristic service-time formula, and simulated rather than measured queueing; quality is `UNVALIDATED` |
| `artifacts/qaq_mixed_trace_20260709_050923/gpu_replay_comparison.json` | CUDA-synchronized real batched generation for ordinary and scalar grouping | One warmup and one repeat; no queue-delay replay, quality metric, predicted profile, or shared-profile override |
| `scripts/build_qaq_request_demand_dataset.py` | Real fixed-bit NLL demand and pre-decode feature collection | Concatenates dataset text before fixed token windowing; lacks document IDs, length grid, and partition manifests |
| `scripts/analyze_qaq_request_demand.py` | Reproducible RF predictors and exact pair precision-work oracle | Request-level rather than document-grouped folds; one seed per invocation; no untouched calibration/test partition |
| `scripts/simulate_qaq_dynamic_batching.py` | Explicit FCFS, scalar, block, max, and quantile policy definitions | Consumes observed majority-bit profiles; does not implement predictor uncertainty, deadlines, stochastic-load seeds, or real service |
| `scripts/replay_qaq_batching_policy.py` | Real batched `model.generate` with CUDA synchronization | Replays batch membership only; fixed `do_sample=False`; no real online queue or shared precision profiles; default repeat is one |

The existing WikiText-2 request-demand result and the 200-request mixed trace
must not be pooled with the confirmatory sample. To prevent pilot reuse, the new
WikiText manifest must exclude every source article contributing tokens to the
first 8,192 tokens of the legacy concatenated WikiText-2 test stream. This
covers both existing contiguous-window pilot artifacts.

## Frozen System Under Test

- Base model: the existing Any-Precision Llama 3.1 8B `w6_orig3` checkpoint.
- Candidate requested bits: exactly `[3, 4, 5, 6]`.
- Reference mode: requested fixed 6-bit (`fixed_high`). Per-route bit ceilings
  remain in force and must be recorded; comparisons use actual executed bits.
- Router: freeze `checkpoints/qaq_router_llama31_8b_th005.pt`, SHA-256
  `59a7d6591722eae4b5f511dbd0accd303c595c3d9ec317ba7266a745684342b4`.
  Its metadata records multibit labels, seed 0, 20 C4 training contexts of 512
  tokens, target bits 4.5, and real relative-output-error labels. Replacing or
  retraining this router is a separate experiment and a protocol deviation.
- Router relative-error threshold: exactly `0.05`; preflight fails if checkpoint
  metadata differs.
- DP threshold: the frozen `T_d.pt` and associated estimator directory, both
  recorded by recursive manifest and SHA-256. The registered `T_d.pt` SHA-256
  is `509e3a83c08b8099d255cdb62df61d1fde82719d87048b831a877625e5aa7936`.
  No threshold retuning is allowed on test documents.
- Confidence fallback: threshold `0.60`, increment `1` bit, capped at the
  route's valid maximum.
- Prefill policy: route both prefill and continuation (`prefill_by_router=True`)
  for dynamic modes. Fixed modes remain fixed during both phases.
- Runtime batch policy: `group` unless the scheduler policy explicitly uses a
  preregistered shared profile.
- Numeric type: FP16 model execution. Any BF16 result is a separate ablation.

The protocol commit, source commit, model config, tokenizer files, router
checkpoint, estimator manifest, CUDA extension build identity, host, GPU,
driver, CUDA, PyTorch, Transformers, Datasets, and Accelerate versions must be
stored with every run.

## Datasets And Document Boundaries

Two source splits are mandatory.

| Short name | Hugging Face identity | Source split | Document unit |
| --- | --- | --- | --- |
| WikiText-2 | `Salesforce/wikitext`, `wikitext-2-raw-v1` | `test` | One article, beginning at a top-level `= title =` line and ending before the next top-level title |
| C4 | `allenai/c4`, English | `validation`, shard `en/c4-validation.00000-of-00008.json.gz` | One dataset row |

Blank WikiText rows stay within their article. Dataset text must never be joined
across document boundaries before request construction. Store the dataset
fingerprint and SHA-256 of normalized text for each selected document. Raw text
need not be committed.

Documents shorter than the requested prompt plus continuation are ineligible
for that length cell. Do not pad, wrap, or borrow tokens from another document.

## Request Counts And Length Grid

Collect **256 requests per dataset**, 512 total, allocated as follows. This is a
pre-run feasibility amendment recorded in Deviations: WikiText-2 test contains
only 62 nonempty source articles and cannot supply the original 576 requests
while preserving document isolation, nonoverlap, and 128-token gaps.

| Prompt tokens | Continuation tokens | Development | Calibration | Confirmatory test | Per dataset |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 32 | 8 | 24 | 64 |
| 128 | 128 | 32 | 8 | 24 | 64 |
| 512 | 32 | 32 | 8 | 24 | 64 |
| 512 | 128 | 32 | 8 | 24 | 64 |
| **Total** | | **128** | **32** | **96** | **256** |

Thus each dataset has exactly 256 requests overall and 96 untouched
confirmatory requests. Continuation tokens are real held-out tokens from the
same document. They are used for teacher-forced quality and observed-profile
targets, never as predictor inputs.

## Document Splits And Window Selection

The split unit is the source document, never a token window.

1. Canonical document ID is
   `sha256(dataset_id || source_split || normalized_document_text)`.
2. Assign documents by
   `uint64(sha256("dpqaq-request-demand-v1" || document_id)[:8]) mod 9`:
   residues `0-3` are development, `4` is calibration, and `5-8` are test.
3. A document and all of its windows belong to exactly one partition. Predictor
   folds are `GroupKFold`-style folds over document IDs, never shuffled request
   folds.
4. Tokenize each document independently with the frozen model tokenizer and
   `add_special_tokens=False`.
5. Enumerate candidate starts deterministically from offset zero. Selected
   token spans may not overlap. Leave at least 128 unused tokens between spans
   from the same document. No token may appear in two selected requests,
   including requests in different length cells.
6. Within each dataset, partition, and length cell, sort candidates by
   `sha256(document_id || start_token || prompt_length || continuation_length)`
   and take the first candidates meeting the quota.
7. Cap contribution at 16 requests per document per partition and at four per
   length cell. If a quota cannot be filled, the protocol is `NOT COMPLETE`;
   do not relax the split or duplicate requests.

The manifest must contain dataset fingerprint, document ID, partition, token
start/end, length cell, request token hash, and a global subset hash. A validator
must reject cross-partition document overlap, token overlap, pilot overlap, and
quota mismatch before GPU collection.

## Modes

Every request runs teacher-forced evaluation through the same loaded QAQ model
in this fixed mode set:

1. `fixed_3`
2. `fixed_4`
3. `fixed_5`
4. `fixed_6`
5. `dp_threshold_only`
6. `mlp_multibit`
7. `mlp_multibit_dp_guard`

Mode order is rotated by a Latin square within each length cell. Model/router
statistics are cleared immediately before each measured request. Finite-logit
checks and feature-prefill passes must not contaminate recorded statistics.

## Quality And Safety Definitions

For a prompt of length `P` and real continuation of length `C`, quality is mean
next-token NLL over exactly the `C` continuation tokens. Prompt labels are
masked but the prompt remains context. Perplexity is `exp(mean NLL)` and is
secondary because averaging request perplexities is not equivalent to corpus
perplexity.

The request-level fixed-bit safety label is frozen as:

```text
safe(b, request) := NLL_fixed_b - NLL_fixed_6 <= 0.02 nats/token
minimum_safe_bit := smallest b in {3,4,5,6} satisfying safe(b, request)
```

Runtime route safety uses the router checkpoint's real-output threshold:

```text
rel_error_b = ||W_6 x - W_b x|| / (||W_6 x|| + 1e-8)
required_bit = smallest valid b with rel_error_b <= 0.05
under_precision := executed_bit < required_bit
```

If a route's maximum valid precision is below 6, its highest valid bit is its
reference for this calculation. Confidence fallback and DP-guard triggers are
counted separately.

The following safety gates are evaluated separately on WikiText-2 and C4 test
data. Quality and request-failure gates produce a separate pass/fail result for
each dynamic mode; a failed baseline does not invalidate another mode:

- All modes: finite logits for 100% of requests and no missing mode output.
- Dynamic-mode quality: one-sided 95% document-cluster-bootstrap upper bound
  for mean NLL delta versus fixed 6 must be at most `0.02` nats/token.
- Request failures: point estimate of requests with NLL delta above `0.02` must
  be at most 5%, and its one-sided 95% upper bound must be at most 10%.
- Guarded MLP route safety: under-precision decision rate must be at most 1%,
  with a one-sided 95% upper bound at most 2%.
- Guard efficacy: guarded MLP must not have a higher under-precision rate or
  unsafe-request fraction than unguarded MLP.
- Non-collapse: guarded MLP mean effective bits must be at least `0.10` bits
  below fixed 6 on each dataset. Otherwise safety may pass, but H3 fails.

`dp_threshold_only` and unguarded MLP remain valid baselines even if they fail a
safety gate; they must be reported as failed rather than removed.

## Predictor Protocol

Predictors may use only prompt token features and fixed-6 prefill features that
exist before decode. Continuation tokens, continuation NLL, generated tokens,
observed route choices, fallbacks, guard decisions, and test labels are
forbidden inputs.

The primary predictor family is the repository's random forest with 300 trees,
`min_samples_leaf=2`, balanced class weights for classification, and no
post-registration hyperparameter search. Run predictor seeds **17, 29, and 43**.

- Training: pooled WikiText-2 and C4 development documents (256 requests).
- Calibration: pooled calibration documents (64 requests), used only to choose
  uncertainty cutoffs and map predictions to scheduler lanes.
- Test: each dataset's 96 test requests, reported separately and pooled.
- Internal development estimates: five grouped folds by document. These are
  diagnostic and cannot replace final test metrics.
- Constant baselines: training-partition majority safe bit, training mean
  effective bits, and training mean profile vector.

Primary predictor endpoints are safe-bit balanced accuracy and macro-F1,
effective-bit MAE/R2, and eight-component guarded-MLP profile MAE and
variance-weighted R2. The profile uses consecutive four-layer groups, matching
the current eight-dimensional representation.

H2 passes only if, on both datasets and for all three seeds:

- safe-bit balanced accuracy and macro-F1 both exceed the constant baseline;
- effective-bit MAE improves by at least 10% and R2 is at least 0.10; and
- group-profile MAE improves by at least 10% and variance-weighted R2 is at
  least 0.10.

Report seed-wise values, mean, standard deviation, confusion matrices,
calibration/coverage, feature importance, and metrics by length cell. Do not
select or omit a seed after observing test results.

## Generation And Decode Settings

Teacher-forced quality above is primary. Generated decoding is used only for
real scheduler/performance replay and uses:

- greedy decoding: `do_sample=False`;
- `temperature`, `top_p`, and `top_k`: unset and recorded as not applicable;
- `num_beams=1`;
- `use_cache=True`;
- `min_new_tokens=max_new_tokens=C`, where `C` is 32 or 128, so every request
  contributes the registered number of generated token slots;
- tokenizer `padding_side="left"`; `pad_token_id=eos_token_id` if the tokenizer
  lacks a pad token;
- no chat template, BOS insertion, or special tokens beyond the independently
  tokenized registered prompt;
- identical prompt IDs and decode configuration for every compared mode/policy.

Set Python, NumPy, and Torch seeds, enable deterministic algorithms where the
installed kernels permit them, disable TF32, and record any kernel that remains
nondeterministic. Greedy token differences across modes are expected and are
reported by output length and token-hash, not suppressed.

## Scheduler Workload

Scheduler evaluation uses only the 96 confirmatory requests from one dataset
at a time. Run scheduling seeds **101, 202, and 303**. Each seed deterministically
permutes requests within length cells and generates exponential inter-arrival
times.

Arrival rates are 50%, 80%, and 95% of the saturated request rate measured for
ordinary guarded-MLP batching on calibration documents. The calibration rate
is frozen before any test replay. Every policy receives identical arrivals,
requests, mode, length mix, and deadlines for a given dataset/load/seed.

- Maximum batch size: 4.
- Maximum scheduler wait: 50 ms from the oldest queued request.
- Queue discipline: oldest arrival first; ties by request ID.
- Length buckets: exact `(P,C)` cells; batching across continuation lengths is
  forbidden because it changes token-slot accounting.
- Deadline: arrival time plus twice the calibration p95 ordinary guarded-MLP
  end-to-end latency for that exact length cell.
- Uncertain predictor lane: calibration cutoff giving 90% predictor coverage;
  uncertain requests use a conservative fixed-6 lane. The cutoff is frozen
  before test replay.

### Mandatory Policies

1. `ordinary_fcfs`: no precision grouping.
2. `length_fcfs`: exact length-cell grouping, otherwise FCFS.
3. `predicted_scalar_025`: group within 0.25-bit predicted effective-bit bins.
4. `predicted_block_l1_025`: group when mean absolute distance between predicted
   eight-component profiles is at most 0.25 bits.
5. `predicted_block_fallback_lane`: policy 4 plus the registered uncertainty
   fallback lane.
6. `oracle_scalar_025`: same as policy 3 using observed scalar demand; diagnostic
   upper bound only.
7. `oracle_block_l1_025`: same as policy 4 using observed profiles; diagnostic
   upper bound only.
8. `max_profile_sharing`: layer-group-wise maximum predicted profile bit,
   projected onto the valid bits of every route in that group.
9. `quantile_profile_sharing_q075`: layer-group-wise 75th percentile predicted
   profile bit, projected onto valid route bits, with guarded fallback.

Policies 8 and 9 may be reported only after real shared-profile execution is
implemented; simulator-only values are labeled `SIMULATED_ONLY`. The primary H4
comparison is policy 5 versus policies 1 and 2 under guarded MLP. Fixed 3/4/5/6,
DP threshold, MLP, and guarded MLP are all benchmarked with `ordinary_fcfs`;
the full scheduler matrix is not crossed with every execution mode.

## GPU Timing Protocol

Heavy runs execute on the lab GPU server with `CUDA_VISIBLE_DEVICES` explicit
and no competing GPU process. Record GPU clocks/power policy when available.

- Warm up model load separately, then run five unreported complete batches for
  each mode/policy/length cell.
- Run three complete measured replays per scheduling seed. Rotate policy order
  with a Latin square; do not always run the baseline first.
- Call `torch.cuda.synchronize()` immediately before and after every measured
  prefill/decode or batch region.
- End-to-end request latency includes queue delay. Separately report TTFT,
  decode time, TPOT, and CUDA-synchronized batch service time.
- Exclude model loading, dataset loading, manifest writing, and warmups from
  latency. Include scheduler decision time in end-to-end latency and report it
  separately.
- Count actual generated tokens, token slots including padding, requests, and
  elapsed wall time. Throughput must be reported as both tokens/s and requests/s.

A replay that only reuses simulator batch membership without reproducing the
online arrival queue cannot pass H4.

## Metrics And Statistical Analysis

### Demand And Quality

- mean NLL and corpus perplexity; paired NLL delta versus fixed 6;
- minimum-safe-bit distribution and entropy;
- request unsafe fraction;
- average selected bit and parameter/execution-weighted effective bits;
- per-route and per-layer bit histograms;
- under-, exact-, and over-precision counts/rates and signed/absolute bit gaps;
- confidence fallback and DP-guard counts/fractions;
- all metrics by dataset, prompt length, continuation length, and mode.

### Predictor

- accuracy, balanced accuracy, macro-F1, confusion matrix;
- MAE, RMSE, R2, and improvement over fold-training constant baselines;
- calibration error and coverage for the uncertainty cutoff;
- per-seed and across-seed mean/standard deviation.

### Scheduler And Runtime

- queue-delay and end-to-end latency p50/p95/p99;
- TTFT p50/p95/p99 and TPOT p50/p95/p99;
- tokens/s, non-padding tokens/s, and requests/s;
- deadline miss fraction;
- batch-size distribution, occupancy, lane count, fragmentation, and wait time;
- scheduler CPU overhead;
- effective bits, profile switches, fallback fraction, and guard fraction;
- under-/over-precision rate for real shared-profile policies;
- peak allocated/reserved CUDA memory and OOM count.

Use 10,000 document-cluster bootstrap replicates with bootstrap seed **1729**.
For scheduler comparisons, resample documents and preserve paired seed/load
results. Report two-sided 95% intervals generally and the preregistered
one-sided upper bounds for safety. Apply Holm correction within each hypothesis
family across the two datasets and four length cells. Do not replace failed
corrected tests with uncorrected results.

## Pass/Fail Criteria

### Data Integrity Gate

Pass only if both dataset manifests meet every quota, contain no cross-split
document overlap or token overlap, exclude the WikiText pilots, and reproduce
their global hashes. Failure makes the experiment `NOT COMPLETE`.

### H1: Heterogeneous Demand

Pass separately per dataset if at least two fixed-bit safety classes each
contain at least 10% of test requests and the document-cluster bootstrap lower
bound for the standard deviation of minimum safe bit is above zero. Report
length-cell interactions regardless of pass/fail.

### H2: Predictor

Pass only under all predictor criteria in the Predictor Protocol on both
datasets and all three predictor seeds. Partial success is reported by target
and dataset, not generalized as predictor success.

### H3: Guarded MLP

Pass only if the guarded-MLP Quality and Safety gates pass on both datasets,
guard efficacy passes against unguarded MLP, and guarded MLP satisfies
non-collapse. Failure of the DP or unguarded baseline is reported but does not
by itself fail H3. A quality-safe result that spends fixed-6 effective bits
fails the adaptive-efficiency claim.

### H4: Scheduling

Policy 5 passes against ordinary FCFS only if, on both datasets and at 80% and
95% offered load:

- median requests/s improvement is at least 5%;
- Holm-corrected paired 95% lower bound for throughput improvement is above 0;
- p95 end-to-end latency does not increase by more than 5%;
- deadline miss fraction does not increase by more than 1 percentage point;
- all guarded-MLP quality and routing safety gates still pass; and
- the direction holds for at least two of three scheduling seeds at each load.

Passing only against `ordinary_fcfs` but not `length_fcfs` is reported as a
batch-composition improvement, not a precision-aware scheduling win. Oracle or
simulation improvements never satisfy H4 without the real online GPU replay.

## Run Order And Information Barriers

1. Commit this protocol and record its commit hash.
2. Implement and unit-test document parsing, manifests, grouped splits, and
   online replay without viewing confirmatory labels.
3. Build manifests and run integrity validation.
4. Collect development data; train the three predictor seeds.
5. Use calibration data only for uncertainty cutoff, capacity, deadlines, and
   operational validation.
6. Freeze code/config/checkpoint hashes and write a signed run manifest.
7. Run confirmatory teacher-forced modes.
8. Run confirmatory scheduling replays in registered Latin-square order.
9. Execute the frozen analysis once. Corrections require a declared deviation
   and a complete rerun; hand-editing result JSON is prohibited.

Test labels and test-mode summaries must not be inspected during predictor,
threshold, scheduler, or deadline development.

## Required Artifacts

- protocol commit and source commit;
- per-dataset document/request manifests and integrity report;
- exact commands and environment manifests;
- source, model, tokenizer, checkpoint, `T_d.pt`, and estimator hashes;
- one atomic raw JSONL record per request/mode;
- predictor models/configs and seed-wise predictions;
- scheduler arrival traces, batch decisions, and real timing records;
- aggregate tables with bootstrap samples or reproducible bootstrap inputs;
- failure/OOM log and a machine-readable deviations list.

No large model, checkpoint, dataset text, or duplicated generated continuation
artifact should be committed. Store hashes and documented external paths.

## Deviations

Pre-run feasibility amendment, 2026-07-11: before any new model result was
collected, dataset inspection found 62 nonempty WikiText-2 test articles. The
original 576-request allocation was infeasible under the registered
document-isolation, nonoverlap, 128-token-gap, and eight-request-per-document
constraints. The allocation is amended to 256 requests per dataset
(128 development, 32 calibration, 96 test), balanced across the four length
cells. The document cap is amended to 16 requests total and four per length
cell. Document-level partitions, pilot exclusion, nonoverlap, gaps, test
information barriers, and document-clustered inference are unchanged.

Every later deviation must record date, protocol section, old value, new value,
reason, whether any development/calibration/test result had been inspected, and
which outputs are exploratory as a consequence. Confirmatory criteria may not
be relaxed after test inspection.

## Readiness Statement

The request-demand collection stage now provides document-level request
construction, leakage-proof partitions, pilot exclusion, pinned datasets, and
resumable validated shards. The full protocol is still **not complete** because
three-seed predictor/scheduling orchestration, fixed-length online arrival replay
with queue delay, and real shared-profile execution remain later implementation
stages.
