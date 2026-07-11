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
