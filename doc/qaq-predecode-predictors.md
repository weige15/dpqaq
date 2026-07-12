# Prompt/Prefill-Only QAQ Predictors

scripts/predecode_predictors.py is the strict held-out predictor path for the
preregistered qaq_request_demand_v2 collection. It is separate from the
legacy pilot analyzer in scripts/analyze_qaq_request_demand.py.

## Data split

The script refuses v1 records and, by default, requires both registered
WikiText-2 and C4 datasets. Use --datasets to name an expanded source set,
such as WikiText-2, C4, and FineWeb-Edu. It uses:

- development: pooled WikiText-2 and C4 records for model fitting.
- calibration: untouched source documents for classifier temperature,
  conformal interval scales, and conservative fallback cutoffs.
- test: untouched source documents for final metrics.

The split unit is (dataset, source.document_id). Development diagnostics use
five grouped folds over document IDs. Test metrics are reported separately for
WikiText-2, C4, and the pooled test set.

Only prompt_features are passed to the models. The loader rejects feature
names containing continuation, generated, observed, route, profile, quality,
target, safe, fallback, guard, or delta terms. Continuation tokens, teacher-
forced quality, observed routes, and guard counts are targets only.

## Run

From the repository root:

    python scripts/predecode_predictors.py \
      --dataset_path artifacts/qaq-request-demand-preregistered-v1 \
      --output_json /tmp/qaq-predecode-predictors.json \
      --seeds 17 29 43 \
      --trees 300 \
      --alpha 0.10 \
      --bootstrap_repetitions 1000 \
      --model_dir /tmp/qaq-predecode-models

The collection and model checkpoints are real research inputs. Do not commit
the generated JSON, pickle bundles, or larger model artifacts.

For cross-dataset transfer, run explicit leave-one-dataset-out evaluation. Each
held-out source is excluded from both development fitting and calibration; only
its test documents are scored:

    python scripts/predecode_predictors.py \
      --dataset_path artifacts/qaq-request-demand-preregistered-v1 /tmp/dpqaq-fineweb-edu-v1/records.jsonl \
      --datasets wikitext2 c4_new fineweb_edu \
      --evaluation_mode lodo \
      --output_json /tmp/dpqaq-predecode-lodo-three-dataset.json \
      --seeds 17 29 43 --trees 300 --alpha 0.10 \
      --bootstrap_repetitions 1000 \
      --model_dir /tmp/dpqaq-predecode-lodo-models

## Targets and baselines

The primary endpoints are:

- minimum safe fixed bit: balanced accuracy, macro-F1, calibration error, and
  under-prediction rate;
- QAQ effective bits: MAE and R² against the development mean baseline;
- eight-component guarded-MLP group profile: MAE and variance-weighted R²
  against both the development mean-vector and a scalar-effective-bit model
  broadcast across groups;
- guard-trigger probability: regression of the per-request guard-trigger
  fraction against its development mean baseline.

The binary label “any DP guard trigger occurred” is also reported. If it is
single-class, its majority baseline is the only valid result; no classifier
claim is made from that target.

## Uncertainty and scheduler fallback

Random-forest ensemble spread is scaled on calibration residuals using
split-conformal quantiles. Classification probabilities are temperature-scaled
on calibration data. The calibration cutoff retains approximately 90% of
requests in the predictive lane; requests above the calibrated regression
uncertainty cutoffs or below the calibrated class-confidence cutoff use the
fixed-high lane.

predict_request_conservatively(bundle, features) exposes this decision for a
scheduler. It returns the predicted safe bit, scalar/profile demand, guard
probability, uncertainty fields, fallback reasons, and either predicted or
fixed_high lane. This is a conservative fallback policy, not a measured
throughput claim.

## Confidence intervals and interpretation

Reported 95% intervals are document-cluster bootstrap intervals, with
(dataset, source.document_id) as the resampling unit. The output includes
paired intervals for model-minus-baseline changes.

Predictability is not established by a positive MAE change alone. A result
must be judged using the preregistered endpoint gates: safe-bit balanced
accuracy and macro-F1 must beat the majority baseline; scalar and profile
regressors must meet both the MAE-improvement and R² requirements; the result
must hold per dataset and across all registered seeds. Coverage and fallback
rates are reported separately. This analysis does not establish CUDA latency,
throughput, or serving benefit.


For the three-source study, the LODO gate must pass for every held-out source
and registered seed before scheduler integration is considered. Until then,
the scheduler must remain on a conservative fixed-high fallback path.
