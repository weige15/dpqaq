# dpqaq

dpqaq is a research repository for precision-aware dynamic batching of
mixed-precision large language model serving. It builds on
[Any-Precision LLM](https://github.com/SNU-ARC/any-precision-llm) for nested
quantized weights and on
[DP-LLM](https://arxiv.org/abs/2508.06041) for runtime precision estimation
and threshold routing.

The current mixed-precision mechanism is a QAQ-style query-adaptive MLP router
running on Any-Precision weights. The larger research question is whether
requests with compatible precision demand can be grouped and executed with
less profile conflict while preserving quality and service constraints. QAQ
routing is one mechanism and ablation condition in that study; it is not the
whole serving contribution.

This repository is a research prototype, not a production serving engine. The
sections below distinguish behavior implemented in code, behavior measured on
real CUDA runs, simulator or replay behavior, and ideas that remain
unvalidated.

## Status at a glance

| Area | Current status | Interpretation |
| --- | --- | --- |
| QAQ router and checkpoint format | Implemented and covered by unit tests | Router labels use real low-bit/reference-bit output-error measurements; checkpoints include configuration, route map, and training metadata. |
| QAQ runtime modes | Implemented; GPU/model validation requires the project inputs | fixed_low, fixed_high, mlp_binary, mlp_multibit, dp_threshold_only, mlp_multibit_dp_guard, and explicit shared-profile execution are implemented. |
| QAQ quality and phase measurements | Real-CUDA artifacts exist | Held-out teacher-forced quality, route-level safety audits, synchronized generation, and CUDA phase profiles are separate evidence types. |
| Request-demand predictors | Implemented held-out analysis path | scripts/predecode_predictors.py uses prompt/prefill-only features and conservative uncertainty handling; current documents do not establish scheduler predictability across all registered datasets. |
| Precision-aware batching | Research implementations exist | Trace collection, a trace-driven simulator, a frozen-stream CUDA benchmark, and an online queue replay are available. They are evaluation paths, not a production scheduler. |
| Dynamic-batching performance claim | Not established | The latest committed profile reports fixed-high as the fastest safety baseline in its two measured sweeps; no general batching improvement is claimed. |
| Production serving, general workload quality, transfer/HBM accounting, and kernel-switch optimization | Pending or unvalidated | These require broader isolated GPU experiments and serving integration. |

The latest evidence is summarized in
[doc/qaq-profile-batching-benchmark.md](doc/qaq-profile-batching-benchmark.md)
and [doc/performance-profile.md](doc/performance-profile.md). The design
contract in [doc/qaq-dynamic-batching-design.md](doc/qaq-dynamic-batching-design.md)
is research guidance; newer code and reports supersede its earlier
“pre-implementation” wording where they provide evidence of an actual
prototype path.

## Current capabilities

### QAQ routing

- scripts/train_qaq_router.py captures real calibration activations and
  creates labels from
  ||W_ref x - W_b x|| / (||W_ref x|| + eps). Multibit labels select the
  smallest candidate bit under the configured error threshold; binary mode
  requires exactly two candidate bits.
- QAQRouter combines hidden-state features with a route embedding, an
  optional norm feature, and an optional DP-style estimated-error feature.
- QAQDPLLMForCausalLM loads the router checkpoint, validates its route map
  against the runtime model, and replaces quantized linear layers with the
  QAQ runtime layer.
- Checkpoints contain the router state, configuration, candidate bits, route
  map, label mode, error threshold, target bits, training configuration, and
  training statistics.

### Runtime modes and measurements

The runtime implementation in
any_precision/modules/QAQDPLLM_Linear.py supports:

- fixed_low and fixed_high for the lowest and highest valid precision;
- mlp_binary and mlp_multibit for router-selected precision;
- confidence fallback, counted separately from ordinary routed decisions;
- dp_threshold_only, which uses DP-style estimator outputs and T_d.pt to
  select the low or high threshold branch; and
- mlp_multibit_dp_guard, which applies
  max(router_bit, dp_threshold_bit) and reports DP-guard triggers separately
  from confidence fallbacks.

get_router_stats() reports average selected bits, parameter-weighted effective
bits, fallback and guard fractions, per-layer bit histograms, threshold counts,
and optional phase timing. A finite-logit check or a generation sanity check is
not a task-quality evaluation. Route-level “quality violation” is also a
different metric: it counts runtime decisions below the bit required by a
real output-error auditor, not task accuracy or perplexity.

For a tensor containing multiple decode rows, the low-level layer can either:

1. group rows by their selected bit and run the corresponding quantized
   operations; or
2. use the maximum selected bit for the tensor when batch_policy="max".

These are execution hooks, not proof that a particular policy is faster. The
phase-separated decode collector clears prefill counters before measuring
decode decisions; see
[doc/qaq-autoregressive-decode-trace.md](doc/qaq-autoregressive-decode-trace.md).

### Request-demand and batching research paths

- scripts/predecode_predictors.py is the strict held-out predictor path. It
  trains and calibrates on document-disjoint development/calibration splits,
  evaluates untouched test documents, rejects post-decode feature names, and
  exposes a conservative fixed-high fallback decision for uncertain requests.
- scripts/collect_qaq_profile_traces.py and
  scripts/collect_qaq_decode_traces.py collect real QAQ request traces. The
  latter separates synchronized prefill and autoregressive decode timing and
  reports per-token route profiles.
- scripts/simulate_qaq_dynamic_batching.py replays trace records under
  ordinary, scalar-budget, block-profile, maximum-profile, and
  quantile-profile policies. It uses an explicit service-time model and marks
  quality, transfer, and kernel-switch fields as unvalidated; its outputs are
  simulated metrics only.
- scripts/benchmark_qaq_profile_batching.py replays one frozen held-out
  request stream on CUDA through fcfs, scalar_predicted, oracle_profile,
  predicted_profile, uncertainty_fallback, fixed_high, and
  max_profile_sharing policies. It
  measures synchronized timing and performs a separate route-safety audit.
- scripts/run_qaq_online_scheduler_replay.py provides a resumable
  real-CUDA online-queue replay for registered scenarios. It is a research
  harness with fixed policies, load fractions, and replay artifacts; it is not
  an online production serving engine.

The request-grouping signal and execution policy must not be conflated. In the
profile benchmark, scalar budgets or predicted coarse profiles decide which
requests are compatible, while the multi-request execution path currently uses
the existing per-route maximum-bit hook. The predicted profile therefore does
not directly set every executed layer. oracle_profile uses observed profiles
for an upper-bound grouping diagnostic, not a deployable pre-decode signal.

## Requirements

The dependency set follows Any-Precision LLM and DP-LLM. The checked-in
requirements.txt currently pins, among other packages, PyTorch 2.2.x,
Transformers 4.39.x, Datasets 2.17.x, Accelerate 0.29.x, FlashAttention 2.7.x,
scikit-learn, CVXPY, and CVXOPT.

Use an environment with:

- Python 3.11;
- CUDA Toolkit 12 or newer and a CUDA-enabled PyTorch installation;
- gcc 9 or newer for the Any-Precision CUDA extension; and
- enough GPU memory for the selected model and quantized artifacts.

Large Llama 3.1 8B runs are intended for the lab GPU server, not a local RTX
4050. Set CUDA_VISIBLE_DEVICES explicitly for every CUDA job and check GPU
availability before a long run. The repository does not include a local
quantized model checkpoint or the built any_precision_ext extension.

## Installation

From a clean checkout:

~~~bash
git clone https://github.com/weige15/dpqaq.git
cd dpqaq
pip install -r requirements.txt
pip install -e .
cd any_precision/modules/kernels
pip install .
cd ../../..
~~~

The editable install packages the Python modules from setup.py; the separate
kernel install builds the any_precision_ext CUDA extension required for real
quantized execution. Return to the repository root before running project
scripts.

Any-Precision model configuration must contain anyprec.size_d. If it does
not, run this once after preparing the quantized model:

~~~bash
python 0_set_configs.py <AP_MODEL_PATH>
~~~

The main QAQ workflow additionally needs:

- an original Hugging Face model path for router training:
  <HF_MODEL_PATH>;
- an Any-Precision quantized model and compatible tokenizer:
  <AP_MODEL_PATH> and, optionally, <TOKENIZER_PATH>;
- a trained QAQ checkpoint: <ROUTER_CHECKPOINT>; and
- DP estimator artifacts in <ESTIMATOR_DIR> when using
  dp_threshold_only, mlp_multibit_dp_guard, or a router trained with
  --include_estimated_error. That directory must contain the files loaded by
  the runtime (max_mem_dict.pt, linear_reg_d.pt, jl_d.pt, and T_d.pt).

Model, tokenizer, checkpoint, estimator, and dataset paths are deliberately
placeholders. Do not put private machine paths or credentials into the README.

## QAQ workflow

### 1. Prepare the quantized model configuration

Run 0_set_configs.py as shown above when size_d is missing. This mutates
the Any-Precision model config; keep the model artifact outside the source
tree when possible.

### 2. Train a router from real error labels

The following is a small real-data calibration run using C4 and the same
training path as a larger run. It is not a synthetic smoke test and it does
not by itself validate quality or serving performance.

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qaq_router.py \
  --model_path <HF_MODEL_PATH> \
  --ap_model_path <AP_MODEL_PATH> \
  --dataset c4 \
  --context_length 512 \
  --dataset_length 40 \
  --bits 3 4 5 6 \
  --label_mode multibit \
  --error_threshold 0.01 \
  --target_bits 4.5 \
  --lambda_budget 0.1 \
  --router_hidden_dim 256 \
  --router_layers 2 \
  --save_path <ROUTER_CHECKPOINT> \
  --device cuda
~~~

The labels are generated by comparing candidate-bit and reference-bit
quantized outputs for captured activations. To add the optional estimated-error
feature, append --include_estimated_error --estimator_results <ESTIMATOR_DIR>
after creating the DP estimator files.

### 3. Run a generation sanity check and mode comparison

This command compares QAQ and DP-guard modes on the same prompt. It reports
generation and sanity-check statistics separately; treat it as a runtime sanity
check, not as full task-quality or serving validation.

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_qaq_inference.py \
  --ap_model_path <AP_MODEL_PATH> \
  --tokenizer_path <TOKENIZER_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --bits 3 4 5 6 \
  --modes qaq fixed_low fixed_high dp_threshold_only mlp_multibit_dp_guard \
  --prompt "Explain mixed-precision inference in one sentence." \
  --max_new_tokens 16 \
  --confidence_threshold 0.6 \
  --device cuda:0 \
  --output_json <OUTPUT_PATH>
~~~

### 4. Evaluate held-out task quality when a quality result is needed

Teacher-forced held-out evaluation is separate from route-level auditing and
from generation timing:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_qaq_heldout.py \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --dataset wikitext2 \
  --context_length 512 \
  --num_examples 16 \
  --bits 3 4 5 6 \
  --device cuda:0 \
  --output_json <OUTPUT_PATH>
~~~

This command requires real model and dataset inputs and should run on an
explicitly selected server GPU. The reported metric is task-level
teacher-forced perplexity; the evaluator also records the separate real
output-error route audit.

## Precision-aware dynamic batching

The current batching work has four layers:

1. **Demand collection.** Real QAQ runs produce request-level route profiles,
   effective bits, fallbacks, DP-guard counts, prompt features, and provenance.
2. **Held-out prediction.** predecode_predictors.py predicts scalar or coarse
   profile demand from prompt/prefill-only features and sends uncertain requests
   to a conservative fixed-high lane. Current LODO reports do not pass all
   endpoint gates, so scheduler integration remains disabled as a claim.
3. **Scheduling and simulation.** Requests can be grouped by arrival window,
   continuation length, scalar budget, observed/oracle profile, predicted
   profile, or uncertainty lane. The simulator is explicit about its synthetic
   service-time model and cannot establish CUDA speedup.
4. **CUDA execution and replay.** The profile benchmark measures a frozen
   held-out stream with CUDA-synchronized prefill/decode timing. The online
   replay measures a registered queue timeline with real batches. Both are
   research evaluation paths with fixed scope, not production serving.

The important semantic boundary is:

~~~text
predicted scalar/profile       -> request compatibility and lane assignment
predicted_group_profile        -> max_profile_sharing batch demand
shared route profile           -> exact bit used by every row in each route
ordinary QAQ modes             -> router/guard/grouped execution
batch_policy="max"             -> legacy low-level router-max execution hook
batch_policy="group"           -> rows grouped by selected bit for separate calls
~~~

`max_profile_sharing` is the explicit scheduler-supplied execution policy. It
takes the component-wise maximum of the actual batch’s held-out predicted
group profiles, validates frozen layer-group metadata, maps routes through the
validated route map, and conservatively projects each demand onto each route’s
actual valid bits. The complete route profile is held constant through
prefill and decode, including singleton batches. The MLP router, confidence
fallback, and DP guard are bypassed; actual route bits are recorded separately
from scheduler-profile under/exact/over accounting and the real output-error
route-safety audit. `fcfs` remains grouped QAQ, `fixed_high` remains fixed-high,
and `length_fcfs` remains scheduling-only. Quantile sharing remains pending;
the simulator’s quantile policy is simulation-only.

Historical benchmark JSON files use the old router-max semantics and are not
overwritten or reinterpreted. No v2 CUDA performance or quality claim is made
until the bounded server-GPU validation is run.

### Trace collection

Collect a real single-request trace before using a simulator or replay. The
collector accepts repeated prompts or a text/JSONL prompt file. The resulting
trace is a measurement artifact, not a batching result.

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_qaq_profile_traces.py \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --prompt_file <PROMPT_FILE> \
  --bits 3 4 5 6 \
  --router_mode mlp_multibit_dp_guard \
  --max_requests 8 \
  --max_new_tokens 16 \
  --device cuda:0 \
  --output_jsonl <TRACE_JSONL> \
  --summary_json <SUMMARY_PATH>
~~~

For phase-separated autoregressive traces across ablation modes, use
scripts/collect_qaq_decode_traces.py; its current interface is documented in
[doc/qaq-autoregressive-decode-trace.md](doc/qaq-autoregressive-decode-trace.md).

### Held-out demand analysis

The strict predictor path uses the registered v2 collection and a
document-disjoint held-out split:

~~~bash
python scripts/predecode_predictors.py \
  --dataset_path <REQUEST_DEMAND_COLLECTION> \
  --datasets wikitext2 c4_new \
  --evaluation_mode pooled \
  --output_json <OUTPUT_PATH> \
  --seeds 17 29 43 \
  --trees 300 \
  --alpha 0.10 \
  --bootstrap_repetitions 1000 \
  --model_dir <PREDICTOR_MODEL_DIR>
~~~

The input collection is a real research artifact. Generated JSON and model
bundles should be written to an external or ignored output directory. The
legacy scripts/analyze_qaq_request_demand.py pilot uses older v1 data and
shuffled request-level folds; it is not the recommended predictor workflow.

### Trace-driven simulation

~~~bash
python scripts/simulate_qaq_dynamic_batching.py \
  --trace_jsonl <TRACE_JSONL> \
  --output_json <OUTPUT_PATH> \
  --policies ordinary_dynamic_batching scalar_budget_batching block_profile_batching max_profile_sharing quantile_profile_sharing \
  --max_batch_size 4 \
  --max_wait_ms 100 \
  --compatibility_threshold 0.25 \
  --scalar_bucket_size 0.25 \
  --quantile 0.75
~~~

This simulator consumes observed per-layer majority bits from the trace rather
than a predictor and uses a heuristic service-time formula. Its output is
SIMULATED_ONLY; quality, transfer bytes, and kernel-switch metrics remain
unvalidated.

### Frozen-stream CUDA benchmark

The benchmark below matches the current parser and the scope described in the
latest committed performance report. It is expensive and must run on a server
GPU with explicit device visibility.

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qaq_profile_batching.py \
  --collection_dir <REQUEST_DEMAND_COLLECTION> \
  --analysis_json <PREDICTOR_ANALYSIS_JSON> \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --datasets wikitext2 c4_new \
  --request_limit 0 \
  --min_uncertain_requests 1 \
  --max_new_tokens 128 \
  --arrival_rate 100 \
  --arrival_seed 101 \
  --predictor_seed 17 \
  --policies fcfs scalar_predicted oracle_profile predicted_profile uncertainty_fallback fixed_high max_profile_sharing \
  --max_batch_size 8 \
  --max_wait_ms 50 \
  --warmup_batches 1 \
  --repeat 10 \
  --confidence_threshold 0.6 \
  --device cuda:0 \
  --local_files_only \
  --output_json <OUTPUT_PATH>
~~~

The benchmark uses one identical frozen request stream for all policies and
separates synchronized timing from the route-safety audit. Its synthetic
arrival process is part of the experiment configuration. Do not generalize its
results to arbitrary traffic or call route-level violations task accuracy.
The committed report currently records fixed-high as the fastest safety
baseline in its reported rate sweeps; it does not establish a batching win.

### Online queue replay

For a registered artifact set, the replay script can first validate the frozen
inputs without running generation:

~~~bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_qaq_online_scheduler_replay.py \
  --collection_dir <REQUEST_DEMAND_COLLECTION> \
  --freeze_manifest <FREEZE_MANIFEST> \
  --analysis_json <PREDICTOR_ANALYSIS_JSON> \
  --route_safety_dir <ROUTE_SAFETY_DIR> \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --tokenizer_path <TOKENIZER_PATH> \
  --output_dir <OUTPUT_DIR> \
  --bits 3 4 5 6 \
  --device cuda:0 \
  --local_files_only \
  --validate_only
~~~

Omit --validate_only only for the registered real-CUDA replay. The replay
requires immutable collection and route-safety inputs and writes results
outside them. Its summary can remain pending performance analysis or fail a
preregistered guarded-quality gate; either outcome is evidence about the
research prototype, not a production-serving result.

See [doc/qaq-dynamic-batching-design.md](doc/qaq-dynamic-batching-design.md),
[doc/qaq-request-demand-preregistered-protocol.md](doc/qaq-request-demand-preregistered-protocol.md),
[doc/qaq-profile-batching-benchmark.md](doc/qaq-profile-batching-benchmark.md),
and [doc/performance-profile.md](doc/performance-profile.md) for schemas,
scope, timing requirements, and provenance.

## DP-LLM and upstream tooling

The original Any-Precision and DP-LLM paths remain in the repository for
quantization, calibration, estimator creation, threshold creation, and
perplexity evaluation. They are useful baselines and prerequisites for some
QAQ modes, but they are not the primary project narrative.

The available upstream-style sequence is:

1. Use quantize.py and the
   [Any-Precision quantization instructions](https://github.com/SNU-ARC/any-precision-llm#quantization)
   to create an Any-Precision model.
2. Run 0_set_configs.py if the model lacks anyprec.size_d.
3. Optionally run 1_find_maxmem.py, 2_finetune.py,
   3_save_estimator.py, and 4_save_th.py to create DP-LLM calibration and
   threshold artifacts.
4. Use test_pp.py for the existing DP-LLM perplexity workflow.

The current positional/options interfaces for this legacy workflow are:

~~~bash
python quantize.py <HF_MODEL_PATH> \
  --seed_precision 3 \
  --parent_precision 6 \
  --yaml_path <ARCHITECTURE_YAML> \
  --cache_dir <CACHE_DIR>

python 1_find_maxmem.py <HF_MODEL_PATH> <AP_MODEL_PATH> \
  --hessian_path <HESSIAN_PATH> \
  --memory_budget 4.0 5.0

python 2_finetune.py <AP_MODEL_PATH> \
  --maxmem 5.0 \
  --targ_bits 3.5

python 3_save_estimator.py <HF_MODEL_PATH> <AP_MODEL_PATH> \
  --finetuned_result <FINETUNED_RESULT>

python 4_save_th.py <AP_MODEL_PATH> \
  --finetuned_result <FINETUNED_RESULT>

python test_pp.py <AP_MODEL_PATH> \
  --estimator_results <ESTIMATOR_DIR>
~~~

These commands require the datasets, calibration inputs, and generated
intermediate paths expected by the corresponding scripts. They are retained as
base/legacy tooling; use the QAQ workflow above for the current research path.

## Validation and evidence

The repository contains multiple evidence classes:

- **Unit-test evidence:** tests/router/ covers router shapes and checkpoint
  round trips, real-label selection logic, route-map validation, guard and
  fallback accounting, decode-stat separation, predictor split rules, and
  simulator/benchmark/replay policy logic. These tests do not replace a real
  model run.
- **Real-CUDA evidence:** committed reports describe real QAQ generation,
  teacher-forced held-out evaluation, route-safety audits, synchronized
  profile-batching timing, and CUDA phase profiling under specified model,
  checkpoint, estimator, workload, warmup, repeat, and device conditions.
- **Simulated evidence:** the dynamic-batching simulator uses trace inputs and
  an explicit heuristic service model. Its latency and throughput fields are
  simulated, not measured GPU performance.
- **Replay evidence:** the batched replay and online scheduler replay exercise
  real CUDA generation and queue accounting for registered artifacts, but they
  remain bounded research harnesses with fixed policies and input streams.
- **Unvalidated ideas:** production serving integration, broad workload
  generalization, a causal batching improvement, transfer/HBM traffic,
  profile-switch and kernel-launch improvements, and all unrun hardware/model
  combinations remain open.

For any performance claim, retain the model/checkpoint/estimator paths,
candidate bits, request stream, arrival process, batch size, prompt/output
lengths, warmups, repeats, CUDA synchronization, fallback and guard counts,
effective bits, p50/p95 latency, throughput, per-layer histograms, and git
commit in the result record. Do not infer speedup from lower effective bits or
from a simulator.

Useful static checks from the repository root are:

~~~bash
python -m compileall any_precision dp_llm_utils scripts
pytest tests/router -q
~~~

The first checks syntax without loading a model. The second requires the
checked-in Python dependencies; a passing unit suite still does not prove
CUDA-kernel or full-model behavior.

## Known limitations and pending work

- Real quantized execution requires the compiled Any-Precision CUDA extension,
  a compatible quantized checkpoint, tokenizer, and matching estimator/router
  artifacts. None of the large Llama 3.1 8B inputs is bundled here.
- Large experiments require an explicitly selected GPU server. The local RTX
  4050 is not the target for large-model inference, training, or benchmarking.
- The repository does not provide a production continuous-batching service,
  admission control, request cancellation, KV-cache paging, multi-tenant
  isolation, or deployment packaging.
- The batching benchmark’s arrival times are generated from a deterministic
  synthetic replay process. Its profile grouping, padding, and maximum-bit
  execution are bounded research policies rather than a general scheduler.
- Predicted request profiles are not yet demonstrated to control all executed
  layer precisions, and current held-out predictor reports keep scheduler
  integration disabled when endpoint gates fail.
- Route-level underprecision/quality-violation measurements are not a
  substitute for task accuracy, perplexity, or human evaluation. Use the
  separate held-out evaluator for task quality.
- Transfer bytes, HBM traffic, prefetch reuse, CUDA-graph reuse, and complete
  kernel/profile switch accounting are not established by the current reports.
- fuse_layers remains a TODO/pass path in the DP-LLM-derived wrappers and is
  not a validated optimization route.
- Some profiler attempts with CPU activity and memory profiling caused
  unbounded bookkeeping; the committed phase profile intentionally focuses on
  CUDA events and synchronized phase totals.
- The preregistered request-demand collection is a frozen input artifact, and
  the expanded dataset/predictor analyses are generalization studies rather
  than proof of a serving benefit.

## Repository artifacts and papers

The following materials are research inputs or frozen outputs and must be
preserved:

- QAQ.pdf, dp_llm.pdf, fineserve.pdf, and other imported paper/reference
  files;
- frozen/preregistered request-demand collections and their manifests, in
  particular artifacts/qaq-request-demand-preregistered-v1/;
- model and tokenizer files, router checkpoints, DP estimator artifacts, and
  training data dumps; and
- large generated datasets, predictor bundles, benchmark logs, profiler
  traces, and replay outputs.

Do not modify paper files or rewrite frozen collections. Keep new checkpoints,
datasets, estimator outputs, training dumps, and benchmark artifacts outside
the source tree or in an explicitly documented ignored output directory. Do
not commit secrets or machine-specific paths.

## Citation and acknowledgements

This repository builds on the Any-Precision LLM implementation and the
DP-LLM research code and paper. Please cite the DP-LLM paper when using that
upstream method:

~~~bibtex
@inproceedings{kwon2025dp,
  title={DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment},
  author={Sangwoo Kwon and Seong Hoon Seo and Jae W. Lee and Yeonhong Park},
  year={2025},
  booktitle={Proceedings of the 39th Conference on Neural Information Processing Systems}
}
~~~

Also cite Any-Precision LLM and QAQ according to their original publications
when using those components. The local QAQ.pdf and dp_llm.pdf files are
reference inputs only and are not modified by this project documentation.
