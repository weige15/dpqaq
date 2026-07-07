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

## Open Questions

- The exact local `--model_path` and `--ap_model_path` for Llama 3.1 8B are not present in this workspace.
- The CUDA extension is not installed in this environment yet.
- GPU-server run command/location is not documented in the imported repository.
