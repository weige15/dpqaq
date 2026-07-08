# AGENTS.md

## Project identity

This repository is `dpqaq`.

It is built on DP-LLM / Any-Precision LLM, but it is no longer just a plain DP-LLM fork. The current research goal is:

> QAQ-style query-adaptive precision routing on Any-Precision weights, with optional DP-LLM-inspired error estimation / threshold guarding.

Do not treat this repository as a fresh implementation. QAQ-specific work already exists and should be audited, tested, and improved incrementally.

## Existing QAQ implementation

Important QAQ files:

- `any_precision/modules/QAQRouter.py`
  - MLP router for precision selection.
  - Supports route embeddings, candidate bits, norm feature, optional estimated-error feature, and checkpoint save/load.

- `any_precision/modules/QAQDPLLM_Linear.py`
  - Runtime quantized linear layer using QAQ router decisions.
  - Supports `fixed_low`, `fixed_high`, `mlp_binary`, `mlp_multibit`.
  - Supports confidence fallback and grouped row execution by selected bit.

- `any_precision/modules/QAQDPLLMForCausalLM.py`
  - Model wrapper that loads QAQ router checkpoints.
  - Replaces quantized linears with `QAQDPLLM_Linear`.
  - Builds route map and reports router stats.

- `scripts/train_qaq_router.py`
  - Trains router from real relative-error labels.
  - Labels must be computed from low-bit vs reference-bit output error:
    `rel_error_b = ||W_ref x - W_b x|| / (||W_ref x|| + eps)`.

- `scripts/run_qaq_inference.py`
  - Generation sanity check and mode comparison script.

- `doc/qaq-router.md`
  - Current QAQ router documentation.

- `doc/repo-intake.md`
  - Repository structure and current known limitations.

## Core research objective

The goal is not just to make code run.

The goal is a research-quality implementation that can support reliable claims about:

- output quality
- effective bits
- latency
- throughput
- router overhead
- fallback behavior
- per-layer precision distribution
- comparison against static precision and DP-LLM threshold routing

When adding features, preserve the ability to run ablations:

- `fixed_low`
- `fixed_high`
- `dp_threshold_only`
- `mlp_multibit`
- `mlp_multibit_dp_guard`

## Non-negotiable rules

- Do not replace the existing QAQ implementation from scratch unless explicitly asked.
- Do not create fake labels, random router decisions, fake checkpoints, or smoke-test-only logic and call it complete.
- Router labels must come from real low-bit vs reference-bit output error.
- Do not silently bypass real quantized kernels when claiming performance.
- Do not claim latency or throughput improvements without CUDA-synchronized benchmark results.
- Do not let a sanity-check forward pass contaminate generation router statistics.
- Do not commit large generated artifacts:
  - checkpoints
  - datasets
  - estimator artifacts
  - training data dumps
  - benchmark logs unless intentionally small and documented
- Do not weaken existing DP-LLM behavior.
- Do not modify `QAQ.pdf`, `dp_llm.pdf`, or imported paper files.
- Avoid changing CUDA kernels unless the task explicitly requires kernel work.
- Never use `git add .`; stage files by logical purpose.

## Hardware and execution constraints

This project involves large LLM inference and CUDA kernels.

- Local RTX 4050 is not suitable for large Llama 3.1 8B experiments.
- Heavy experiments must run on the lab GPU server.
- Always set `CUDA_VISIBLE_DEVICES` explicitly for GPU jobs.
- Before running long jobs, check GPU availability manually.
- Prefer small CPU-only/unit tests for code correctness.
- Use real CUDA/model runs only for final validation or benchmark tasks.

Example GPU command pattern:

CUDA_VISIBLE_DEVICES=0 python scripts/run_qaq_inference.py ...

Setup commands

Install dependencies:

pip install -r requirements.txt
pip install -e .

Install Any-Precision CUDA kernels:

cd any_precision/modules/kernels
pip install .

Return to repo root before running project scripts:

cd /nfs/home/s314511048/dpqaq

If the Any-Precision model config lacks anyprec.size_d, run:

python 0_set_configs.py <AP_MODEL_PATH>
Validation commands

For syntax-level checks:

python -m compileall any_precision dp_llm_utils scripts

For router unit tests, once tests exist:

pytest tests/router -q

For runtime tests, once tests exist:

pytest tests/router tests/runtime -q

For script interface checks:

python scripts/train_qaq_router.py --help
python scripts/run_qaq_inference.py --help

For real inference sanity check on GPU:

CUDA_VISIBLE_DEVICES=0 python scripts/run_qaq_inference.py \
  --ap_model_path <AP_MODEL_PATH> \
  --router_checkpoint <ROUTER_CHECKPOINT> \
  --estimator_results <ESTIMATOR_DIR> \
  --bits 3 4 5 6 \
  --prompt "Explain mixed-precision inference in one sentence." \
  --max_new_tokens 16 \
  --device cuda \
  --output_json qaq_inference_stats.json
Benchmark requirements

Any benchmark or performance claim must include:

warmup count
repeat count
CUDA synchronization before and after timing
model path
router checkpoint path
estimator path, if used
candidate bits
batch size / prompt count
prompt length or dataset
max new tokens
average selected bit
effective bits
fallback count / fraction
DP guard trigger count / fraction, if used
p50 latency
p95 latency
tokens/sec
per-layer bit histogram
git commit hash

If timing CUDA work, use:

if torch.cuda.is_available():
    torch.cuda.synchronize()

before starting and after finishing the measured region.

Correctness expectations

For router training:

Labels must be generated from real captured activations.
Labels must compare low-bit / candidate-bit output against a reference bit.
Binary mode must require exactly two bits.
Multibit mode must choose the smallest safe bit.
Saved checkpoint must include:
router state dict
router config
candidate bits
route map
label mode
error threshold
target bits
training stats
training config

For runtime routing:

fixed_low must always use the lowest valid precision.
fixed_high must always use the highest valid precision.
mlp_multibit must use router predictions.
Confidence fallback must be counted separately.
DP threshold guard, if implemented, must be counted separately from confidence fallback.
Router stats must reflect the measured operation only.
Current high-priority tasks

Prefer these tasks before adding new research features:

Update root README.md so it describes dpqaq, not only DP-LLM.
Add unit tests for QAQRouter.
Add checkpoint save/load roundtrip tests.
Add tests for label generation logic.
Fix inference statistics so generation stats are not contaminated by extra forward passes.
Add CUDA-synchronized benchmark script.
Add true DP-threshold guard using T_d.pt.
Only after the single-request path is trustworthy, add serving-aware batching.
Suggested task workflow

For complex changes, start with a plan before editing.

Use this structure:

Goal:
What should change?

Context:
Which files matter?

Constraints:
What must not break?

Done when:
Which tests or commands prove completion?

For implementation tasks:

Inspect relevant files first.
Propose a small file-level plan.
Modify only the necessary files.
Add or update tests.
Run the most relevant validation command.
Summarize changed files, commands run, and remaining risks.
Definition of done

A task is done only when:

the requested behavior is implemented
relevant tests or checks pass
existing QAQ modes still work
existing DP-LLM path is not broken
generated stats are trustworthy
limitations are documented
no fake data or fake success path is introduced
the diff is small enough to review
Review checklist

Before finishing, check:

Did this change alter router numerics?
Did this change alter DP-LLM behavior?
Did this change require CUDA validation?
Did it accidentally use random labels or fake routing outputs?
Did it add a large artifact?
Did it mix generation stats with sanity-check stats?
Did it report latency without CUDA synchronization?
Did it change public CLI behavior?
Are docs updated if behavior changed?
Anti-patterns to avoid

Do not do these:

“Implemented router” when only a mock exists.
“Benchmark shows faster” without synchronized timing.
“Inference works” when only --help or compileall passed.
“Training works” when labels are random or cached from an unknown source.
Large rewrites that make existing QAQ files unrecognizable.
Adding a new abstraction without tests.
Moving files without updating imports, docs, and commands.
