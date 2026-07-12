# Three-Dataset Pre-Decode Predictor Run

## Collection

A real FineWeb-Edu extension was collected on CUDA device 0 using the frozen
Any-Precision model, QAQ router, DP estimator, and candidate bits 3/4/5/6.

- Source: HuggingFaceFW/fineweb-edu, sample-10BT, revision
  87f09149ef4734204d70ed1d046ddc9ca3f2b8f9.
- Input: first 50,000 rows of one cached parquet shard, selected
  deterministically by document hash.
- Requests: 256; 128 development, 32 calibration, 96 test.
- Documents: 256 unique source documents.
- Manifest hash:
  873b0648a5be734889d307ce7ce16fa7aa7f3a5fdcbedaa87aa18e0b536e9589.
- Records SHA-256:
  6af8216a3e9a6bb9964db184b6a975c8119ca2444f5fd8bc9625794e60998b71.

The output is kept outside the repository at
/tmp/dpqaq-fineweb-edu-v1. It contains v2 records without raw text or token
payloads.

## Predictor evaluation

The combined held-out run used 768 requests from WikiText-2, C4, and
FineWeb-Edu, three registered seeds (17, 29, 43), 300-tree random forests,
calibration-only uncertainty fitting, and 1,000 source-document cluster
bootstrap repetitions.

- Result SHA-256:
  70cbaea0ab4719a3efdb2db39c29e7e9ae6cca86ab8cfa22160134b4df2017d9.
- Development/calibration/test: 384/96/288 requests.
- Grouped development folds: zero source-document overlap.
- Calibration and test metrics are in
  /tmp/dpqaq-predecode-three-dataset.json.

Gate outcomes are (safe-bit, effective-bits, group-profile):

| Seed | WikiText-2 | C4 | FineWeb-Edu |
| ---: | --- | --- | --- |
| 17 | pass, fail, fail | pass, pass, pass | pass, pass, pass |
| 29 | pass, fail, fail | pass, pass, pass | pass, pass, pass |
| 43 | pass, fail, fail | pass, pass, pass | fail, pass, pass |

## Conclusion

The third source improves evidence about domain transfer: FineWeb-Edu passes
the regression gates for all seeds and the safe-bit gate for two seeds. The
overall preregistered verdict remains predictability_established: false
because WikiText-2 fails the effective-bit and profile gates for every seed,
and FineWeb-Edu fails safe-bit balanced accuracy for seed 43.

These results do not authorize serving decisions or imply latency/throughput
benefits. The conservative fixed-high uncertainty fallback remains required.

