# Debug Report

## Symptom

The focused QAQ tests and held-out evaluator CLI could not start because the
shell resolved Python outside the repository virtual environment.

## Reproduction Command

Working directory: `/nfs/home/s314511048/dpqaq`

Shell: `/bin/bash`

Runtime: system Python 3.12.3

Environment: `VIRTUAL_ENV=/nfs/home/s314511048/.venv`, but
`/nfs/home/s314511048/.venv/bin` was absent from `PATH`.

Relevant environment variables:

```text
VIRTUAL_ENV=/nfs/home/s314511048/.venv
CONDA_PREFIX=
PYTHONPATH=
CUDA_VISIBLE_DEVICES=
```

```bash
python3 -m pytest tests/router/test_evaluate_qaq_heldout.py -q
```

## Expected Behavior

Pytest should import the project ML dependencies and execute the CPU tests.

## Actual Behavior

The system Python was selected and did not contain pytest or PyTorch.

## Error Log

```text
/usr/bin/python3: No module named pytest
ModuleNotFoundError: No module named 'torch'
```

## Failure Layer Classification

Most likely layer:

* Command problem: yes
* Permission problem: no
* Shell/script invocation problem: yes
* Environment problem: yes
* Dependency problem: no
* Python/package/import problem: yes
* GPU/CUDA problem: no
* Distributed/torchrun problem: no
* Filesystem/path problem: no
* Data/checkpoint/model file problem: no
* Code logic problem: no
* Configuration problem: yes
* Resource problem: no
* Concurrency/race problem: no
* Unknown/insufficient evidence: no

Final classification: shell environment activation/PATH mismatch.

## Hypotheses

### Hypothesis 1: Project dependencies were not installed

Why it could explain the symptom: system Python reported missing torch and
pytest.

Evidence for: imports failed under `/usr/bin/python3`.

Evidence against: `/nfs/home/s314511048/.venv/bin/python` imports torch,
pytest, transformers, datasets, and accelerate successfully.

How to verify: invoke the virtual-environment Python directly.

### Hypothesis 2: The virtual environment was recorded but not activated

Why it could explain the symptom: `VIRTUAL_ENV` points to the correct
environment, but `which python3` resolves to `/usr/bin/python3`.

Evidence for: the virtual environment's `bin` directory is absent from
`PATH`.

Evidence against: none.

How to verify: compare imports from system Python and
`/nfs/home/s314511048/.venv/bin/python`.

## Most Likely Root Cause

The shell has a stale or partial virtual-environment state: `VIRTUAL_ENV` is
set, but its executable directory is not on `PATH`. The dependencies and
CUDA-enabled PyTorch are already installed; no package installation is needed.

## Minimal Fix

Invoke `/nfs/home/s314511048/.venv/bin/python` explicitly for verification and
GPU evaluation. Keep `CUDA_VISIBLE_DEVICES=0` explicit for the GPU job.

## Verification

```bash
/nfs/home/s314511048/.venv/bin/python -m pytest tests/router -q
CUDA_VISIBLE_DEVICES=0 /nfs/home/s314511048/.venv/bin/python -c \
  "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info(0))"
```

Expected verification result:

```text
Focused tests pass, and local CUDA device 0 is an RTX 3090 with sufficient free memory.
```
## Follow-up Dataset Failure

After the environment fix, the one-example GPU gate reproduced a second setup
failure before model loading:

```text
huggingface_hub.errors.HfUriError: Invalid HF URI
'hf://datasets/wikitext@.../.huggingface.yaml'
```

Datasets 5.0.0 and Hugging Face Hub 1.20.1 resolved the legacy `wikitext`
alias to `Salesforce/wikitext`, but the unnamespaced URI failed validation.
The local cache and Hub API both confirmed `Salesforce/wikitext` at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The minimal applied fix changes
the evaluator and validation documentation to the canonical namespaced
identifier. Verification is rerunning the exact one-example CUDA command.
## Follow-up Fixed-High Invariant

The first real five-mode forward completed but artifact writing was rejected
because the evaluator required fixed-high to have neither under-precision nor
over-precision. That invariant was incorrect: fixed-high must never be below
the route reference, but it is legitimately over-precise whenever a lower bit
meets the error threshold. The check now rejects only fixed-high
under-precision, with a CPU regression test covering both cases.

Final verification:

```text
38 router tests passed.
The full 16-example CUDA run completed on Device 0.
Artifact status: REAL_GPU_HELDOUT.
Artifact source hashes, five modes, 16 identical token windows per mode,
finite logits, and fixed-high baseline deltas were independently verified.
```

## Follow-up Shared-Profile CPU Fixture Failures

The required shared-profile CPU gate initially reproduced three test-fixture
failures, all outside production execution: the fake dequantized kernel was
multiplied by an all-zero input; a statistics assertion included zero-valued
candidate-bit buckets; and a route-specific fixture accidentally made bit 4
valid for a route whose test expected it to be rejected. The production
shared-profile path was not changed for these failures. The minimal test-only
corrections use a one-hot fake-kernel input, assert positive executed-bit
counts, and pass the route-specific valid-bit set explicitly. The targeted
CPU gate is rerun after these corrections.

## Follow-up Validation and GPU Allocation

The corrected targeted shared-profile CPU gate passed with 33 tests, and the
complete `tests/router` suite passed with 93 tests. Compileall, both required
script help checks, and `git diff --check` also passed using
`/nfs/home/s314511048/.venv/bin/python`.

The bounded real-CUDA comparison subsequently completed on GPU 4, an RTX 3090.
The larger three-repeat trace measured fixed-high at 905.337 ms p50 and 33.4677
generated tokens/s versus max-profile-sharing at 916.279 ms p50 and 33.0366
tokens/s; both executed at effective bit 6. The separate 241,920-decision
route-safety audit found zero underprecision violations. A full-process Nsight
capture included model load, so its H2D dominance is not treated as a
serving-step bottleneck. GPU 4 is now occupied by another user's process; no
process was interrupted.
