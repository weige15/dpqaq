# Three-source LODO predictor evaluation

## Status

The requested leave-one-dataset-out evaluation ran on the three collected
sources before scheduler integration was considered.

Command:

    python scripts/predecode_predictors.py \
      --dataset_path artifacts/qaq-request-demand-preregistered-v1 /tmp/dpqaq-fineweb-edu-v1/records.jsonl \
      --datasets wikitext2 c4_new fineweb_edu \
      --evaluation_mode lodo \
      --output_json /tmp/dpqaq-predecode-lodo-three-dataset.json \
      --seeds 17 29 43 --trees 300 --alpha 0.10 \
      --bootstrap_repetitions 1000 \
      --model_dir /tmp/dpqaq-predecode-lodo-models

The result is real-data, prompt/prefill-only, calibrated evaluation output.
Its SHA-256 is
a4b57e80ca70138ab52175e61643397dc0b5326aa60c7685a9532950caa52239.

## Split integrity

The combined collection has 768 requests: 384 development, 96 calibration,
and 288 test. It has 285, 70, and 210 source documents in those partitions.
For every held-out source, the models use 256 development and 64 calibration
requests from the other two datasets, then evaluate 96 test requests from the
held-out dataset only. All three LODO split-integrity checks are true, and
the grouped development folds have no document overlap.

Each seed result contains document-cluster bootstrap 95% confidence intervals
for endpoint metrics and paired model-minus-baseline changes. Calibration
coverage was 0.8906 for safe-bit classification, 0.9219 for effective bits
and guard probability, and 0.9883--0.9902 for the group profile.

## Transfer results

Ranges below are across seeds 17, 29, and 43. Effective/profile baselines are
the mean learned from the two non-held-out training datasets.

| Held-out dataset | Safe-bit balanced accuracy (majority) | Effective-bit MAE / mean baseline | Profile MAE / mean baseline |
| --- | ---: | ---: | ---: |
| WikiText-2 | 0.245--0.263 / 0.250 | 0.02789--0.02819 / 0.02479 | 0.03139--0.03220 / 0.02784 |
| C4 | 0.280--0.332 / 0.250 | 0.02041--0.02056 / 0.02661 | 0.02703--0.02715 / 0.03260 |
| FineWeb-Edu | 0.227--0.288 / 0.250 | 0.01658--0.01684 / 0.01722 | 0.02144--0.02175 / 0.02367 |

The preregistered gate outcomes are:

| Held-out dataset | Safe bit | Effective bits | Group profile |
| --- | --- | --- | --- |
| WikiText-2 | 2/3 seeds pass | 0/3 pass | 0/3 pass |
| C4 | 3/3 pass | 3/3 pass | 3/3 pass |
| FineWeb-Edu | 2/3 pass | 0/3 pass | 0/3 pass |

The confidence intervals reinforce the transfer distinction: for example,
the seed-17 effective-bit MAE delta versus the training mean is
[-0.008419, -0.003899] on C4, [-0.003883, 0.002613] on FineWeb-Edu, and
[-0.000054, 0.005794] on WikiText-2. A positive MAE improvement alone is
therefore not treated as evidence of predictability.

## Decision

The overall LODO result is
predictability_established: false. Scheduler integration remains disabled;
the evaluated conservative fixed-high fallback is not an end-to-end serving
decision or latency/throughput claim.

## Expanded four-source rerun

A fourth source-separated dataset, HellaSwag, was added from cached
Rowan/hellaswag train parquet revision
218ec52e09a7e7462a5400043bb9a69a41d06b76. The collector uses 256 unique
source_id documents and only ctx_a/ctx_b text; endings and labels are
excluded. The records have 128/32/96 development/calibration/test requests.

Command:

    python scripts/predecode_predictors.py \
      --dataset_path artifacts/qaq-request-demand-preregistered-v1 /tmp/dpqaq-fineweb-edu-v1/records.jsonl /tmp/dpqaq-hellaswag-v1/records.jsonl \
      --datasets wikitext2 c4_new fineweb_edu hellaswag \
      --evaluation_mode lodo \
      --output_json /tmp/dpqaq-predecode-lodo-four-dataset.json \
      --seeds 17 29 43 --trees 300 --alpha 0.10 \
      --bootstrap_repetitions 1000 \
      --model_dir /tmp/dpqaq-predecode-lodo-four-dataset-models

The result contains 1,024 requests (512/128/384 development/calibration/test)
and has SHA-256
a123718e44d4a1d7367989b20b461bc8cf82ea4fd1f2b82925925c24250c2bc5.

| Held-out dataset | Safe bit | Effective bits | Group profile |
| --- | --- | --- | --- |
| WikiText-2 | 2/3 seeds pass | 0/3 pass | 0/3 pass |
| C4 | 3/3 pass | 3/3 pass | 0/3 pass |
| FineWeb-Edu | 2/3 pass | 3/3 pass | 3/3 pass |
| HellaSwag | 3/3 pass | 0/3 pass | 0/3 pass |

Calibration coverage remains approximately 0.8958 for safe-bit classification,
0.9167 for effective bits and guard probability, and 0.9883--0.9896 for the
profile. HellaSwag effective-bit MAE is lower than its mean baseline, but its
R2 is about -4.6 to -5.2; this is not evidence of predictability. All four
LODO split-integrity checks are true, and scheduler integration remains
disabled.

## Support-matched transfer rerun

Because HellaSwag uses 32/64-token windows while the original natural-text
collection began at 128 tokens, a real FineWeb supplement was collected from
an unused shard with the four HellaSwag length cells. The original FineWeb
documents were disjoint from the supplement.

The final input has 1,280 requests: 640 development, 160 calibration, and
480 test. The output is
/tmp/dpqaq-predecode-lodo-four-dataset-short-transfer.json with SHA-256
bd244b672db4a274e49c2a5c7fe13b458728914223c8142997bad7ad9fd53947.

| Held-out dataset | Safe bit | Effective bits | Group profile |
| --- | --- | --- | --- |
| WikiText-2 | 0/3 pass | 0/3 pass | 0/3 pass |
| C4 | 3/3 pass | 3/3 pass | 3/3 pass |
| FineWeb-Edu | 3/3 pass | 3/3 pass | 0/3 pass |
| HellaSwag | 3/3 pass | 0/3 pass | 0/3 pass |

HellaSwag transfer improved materially: mean effective-bit R2 changed from
-4.9105 to -1.5464, and profile R2 from -1.5719 to -0.4156. Its effective
MAE confidence intervals improved over the mean baseline, but R2 remains
negative, so the endpoint gate still fails. Calibration coverage remained
approximately 0.8984/0.9141/0.9893/0.9141 for safe-bit/effective/profile/
guard targets. All four LODO split-integrity checks remain true.

The overall result is still predictability_established: false. Scheduler
integration remains disabled and the conservative fixed-high fallback stays
in force.
