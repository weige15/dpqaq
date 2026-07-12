# Repository Intake

## Stack

Python package for Any-Precision / DP-LLM quantized causal language model inference and calibration. `requirements.txt` pins PyTorch, Transformers, Accelerate, Datasets, FlashAttention, lm-eval, CVX dependencies, and scikit-learn. `setup.py` packages `any_precision`.

## Entry Points

- `quantize.py`: top-level Any-Precision quantization entry point.
- `0_set_configs.py`: records per-linear size metadata into an Any-Precision model config.
- `1_find_maxmem.py`: finds layer-wise maximum precision under a memory budget.
- `2_finetune.py`: tunes average precision assignments for DP-LLM.
- `3_save_estimator.py`: records real calibration hidden states and builds linear/JL error estimators.
- `4_save_th.py`: saves DP-LLM threshold values.
- `test_pp.py`: evaluates perplexity with `DPLLMForCausalLM`.
- `demo.py` / `run_eval.py` / `evaluate.sh`: example and evaluation helpers.
- `scripts/build_qaq_request_demand_dataset.py`: collects real fixed-bit quality targets, observed QAQ profiles, and prompt/prefill-only features for the preregistered two-dataset request-demand study.
- `scripts/analyze_qaq_request_demand.py`: legacy pilot analyzer for the v1 request-level oracle and shuffled K-fold predictor analysis.
- `scripts/predecode_predictors.py`: held-out v2 predictor training/evaluation with document-cluster bootstrap confidence intervals and conservative uncertainty fallback; accepts multiple source collections.
- `scripts/collect_qaq_fineweb_request_demand.py`: real CUDA FineWeb-Edu source-document collection using the shared QAQ callback.
- The FineWeb collector's `short_transfer` profile supplies real 32/64-token support-matched windows for HellaSwag transfer evaluation.
- `scripts/collect_qaq_hellaswag_request_demand.py`: real CUDA HellaSwag source-ID collection using only ctx_a/ctx_b and the shared QAQ callback.
- `scripts/qaq_request_demand_protocol.py`: deterministic source-document manifests, document partitions, shard validation, and v2 record schemas.

## Install Commands

- Documented: `cd /nfs/home/s314511048/dpqaq && pip install -r requirements.txt`.
- Documented: `cd /nfs/home/s314511048/dpqaq/any_precision/modules/kernels && pip install .`.
- Inferred editable package install: `cd /nfs/home/s314511048/dpqaq && pip install -e .`.

## Build Commands

- Documented CUDA extension build/install: `cd /nfs/home/s314511048/dpqaq/any_precision/modules/kernels && pip install .`.
- No separate project build system was found beyond `setup.py` and the CUDA extension `any_precision/modules/kernels/setup.py`.

## Run Commands

- Documented quantization: `cd /nfs/home/s314511048/dpqaq && python quantize.py <model> [options]`.
- Documented config setup: `cd /nfs/home/s314511048/dpqaq && python 0_set_configs.py <ap_model_path>`.
- Documented DP-LLM max precision search: `cd /nfs/home/s314511048/dpqaq && python 1_find_maxmem.py <model> <ap_model_path> --hessian_path <path> --memory_budget <bits>`.
- Documented DP-LLM average precision tuning: `cd /nfs/home/s314511048/dpqaq && python 2_finetune.py <ap_model_path> --maxmem <budget> --targ_bits <bits>`.
- Documented estimator generation: `cd /nfs/home/s314511048/dpqaq && python 3_save_estimator.py <model> <ap_model_path> --finetuned_result <path>`.
- Documented threshold generation: `cd /nfs/home/s314511048/dpqaq && python 4_save_th.py <ap_model_path> --arr_path <path>`.
- Documented perplexity evaluation: `cd /nfs/home/s314511048/dpqaq && python test_pp.py <ap_model_path> --estimator_results <estimator_dir>`.

## Test Commands

- Inferred static check: `cd /nfs/home/s314511048/dpqaq && python -m compileall .`.
- No pytest configuration or CI workflow was found.

## Existing Modules

- `any_precision/modules/AnyPrecisionLinear.py`: quantized linear layer using `dequant_kbit` and `matmul_kbit`.
- `any_precision/modules/AnyPrecisionForCausalLM.py`: loads an Any-Precision quantized model and replaces linears.
- `any_precision/modules/DPLLM_Linear.py`: DP-LLM runtime precision selector using linear-regression or JL relative-error estimates and thresholds.
- `any_precision/modules/DPLLMForCausalLM.py`: model wrapper that wires DP-LLM dictionaries into per-layer `DPLLM_Linear` modules.
- `any_precision/modules/DPLLM_Linear_Finetune.py` and `DPLLM_Finetune.py`: differentiable average precision tuning path.
- `dp_llm_utils/record_x.py`: captures real hidden-state inputs for calibration.
- `dp_llm_utils/dataset_tokenize.py`: C4/WikiText calibration dataloader utilities.
- `dp_llm_utils/model_def.py`: architecture module lists and asynchronous estimator metadata.
- `any_precision/quantization/`: Any-Precision quantization implementation.
- `any_precision/analyzer/`: architecture analyzers and YAML architecture metadata.

## Data Files

- `QAQ.pdf`: user-provided research paper input; do not modify.
- `dp_llm.pdf`: user-provided research paper input; do not modify.
- `figures/*.png`: imported documentation assets from DP-LLM.
- Generated estimator outputs are expected under paths like `estimator_private_values/` and `estimator_shared_values/`; avoid committing large generated artifacts.
- `artifacts/qaq-request-demand-preregistered-v1/`: real 512-request v2 request-demand collection with separate WikiText-2 and C4 document partitions. Treat it as an input artifact; do not rewrite it during offline predictor analysis.
- Cached FineWeb-Edu sample parquet files under the user Hugging Face cache are external read-only inputs for the new collector; generated FineWeb records are written outside the repository.

- Cached Rowan/hellaswag train parquet is an external read-only input; generated HellaSwag records and expanded LODO outputs are written outside the repository.
## Config Files

- `requirements.txt`: Python dependency pins.
- `setup.py`: package metadata.
- `Dockerfile`: CUDA/Python environment recipe.
- `any_precision/analyzer/architectures/*.yaml`: model architecture analyzer configs.
- Any-Precision model configs must include `config.anyprec`; `0_set_configs.py` mutates the model config to add `size_d`.

## Known Broken Parts

- `README.md` states the repository currently contains performance evaluation code and that latency measurement code will be updated later.
- `DPLLMForCausalLM.fuse_layers` and related `fuse_layers` methods are TODO/pass placeholders.
- The CUDA extension `any_precision_ext` is mandatory for real quantized execution.
- No local quantized model checkpoint is present in this workspace.
- The preregistered v2 collection exists, but the legacy analyzer only consumes `qaq_request_demand_v1` and uses shuffled request-level folds. The strict held-out path is `scripts/predecode_predictors.py`.
- The FineWeb-Edu source extension is a bounded cached-parquet subset; it does not claim full FineWeb coverage.

- HellaSwag source documents concatenate real ctx_a/ctx_b fields per source_id and deliberately exclude answer endings and labels; this is a domain-transfer extension, not a claim of natural-document continuity.
## External Dependencies

- CUDA Toolkit 12+, gcc-9+, PyTorch with CUDA, FlashAttention, Transformers, Accelerate, Datasets, lm-eval, CVX packages.
- Hugging Face model/tokenizer downloads or local model paths are required.
- C4/WikiText dataset access is required for default calibration data.
- Real QAQ/DP-LLM validation with Llama 3.1 8B should run on a GPU server, not the local RTX 4050 per user instruction.

## What Not To Touch

- Do not modify `QAQ.pdf` or `dp_llm.pdf`.
- Do not modify `.git/`.
- Avoid changing CUDA kernels unless required.
- Avoid committing large generated checkpoints, datasets, estimator artifacts, or router training data.
- Do not weaken existing DP-LLM behavior or remove existing scripts.
- Preserve the preregistered request-demand collection and its manifests; predictor outputs and model bundles belong in a separate output path.

## Open Questions

- The exact local `--model_path` and `--ap_model_path` for Llama 3.1 8B are not present in this workspace.
- The CUDA extension is not installed in this environment yet.
- GPU-server run command/location is not documented in the imported repository.
