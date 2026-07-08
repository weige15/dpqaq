# QAQ Experiment Log Template

Use one copy of this template per validation or experiment run. Do not fill
unknown values with guesses; write `UNVALIDATED`.

## Run Identity

- Date:
- Owner:
- Git commit:
- Host:
- GPU:
- CUDA:
- Python environment:
- Original model path:
- Any-Precision model path:
- Router checkpoint:
- Estimator results path:

## Goal

- Question being tested:
- Expected comparison:
- Paper table target, if any:

## Configuration

- Candidate bits:
- Reference bit:
- Label mode:
- Error threshold:
- Target bits:
- Lambda budget:
- Router hidden dim:
- Router layers:
- Layer embedding dim:
- Norm feature:
- Estimated-error feature:
- Confidence threshold:
- Fallback bits:
- Prefill policy:
- Batch policy:
- DP-LLM artifacts used:

## Data

- Training dataset:
- Training split/subset:
- Training context length:
- Training example count:
- Validation/eval dataset:
- Validation/eval split/subset:
- Prompt count:
- Max new tokens:

## Commands

Training command:

```bash
UNVALIDATED
```

Inference/evaluation command:

```bash
UNVALIDATED
```

Benchmark command, if CUDA-synchronized:

```bash
UNVALIDATED
```

## Correctness Results

- Router checkpoint reload passed:
- Route-map match passed:
- Label-count artifact:
- Validation accuracy:
- Unsafe-label or threshold-violation rate:
- Finite logits:
- Stats isolation checked:
- Effective-bits formula checked:
- Known failures:

## Metrics

| Mode | Quality metric | Avg bit | Effective bits | Fallback fraction | DP guard fraction | Latency p50 | Latency p95 | Tokens/sec | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_low | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| fixed_high | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| dp_threshold | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |
| mlp_multibit | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED | UNVALIDATED |

## Artifacts

- Router checkpoint:
- Router metadata JSON:
- Inference stats JSON:
- Per-layer histogram:
- Evaluation output:
- Benchmark log:

## Interpretation

- What is validated:
- What remains `UNVALIDATED`:
- Can this run be used in a paper table:
- Reason:
