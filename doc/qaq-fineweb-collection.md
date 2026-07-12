# FineWeb-Edu Source Extension

scripts/collect_qaq_fineweb_request_demand.py extends the v2 request-demand
schema with a third, source-document-preserving dataset: the cached
HuggingFaceFW/fineweb-edu sample-10BT parquet data.

The collector:

- scans a bounded, explicitly recorded prefix of sorted parquet rows;
- hashes each normalized document into development, calibration, or test with
  the existing document partition function;
- assigns one non-overlapping request per document to the four registered
  prompt/continuation length cells;
- selects exact 32/8/24 quotas per cell for development/calibration/test;
- stores document IDs, row-ID hashes, parquet names, selection hashes, dataset
  revision, file hashes, and model provenance;
- reuses the real QAQ collection callback for fixed-bit NLL, QAQ profiles,
  prompt token features, fixed-high prefill features, and guard counts.

Raw text and token IDs are not written to the manifest or records.

Example GPU command:

    CUDA_VISIBLE_DEVICES=0 python scripts/collect_qaq_fineweb_request_demand.py \
      --ap_model_path <AP_MODEL_PATH> \
      --router_checkpoint <ROUTER_CHECKPOINT> \
      --estimator_results <ESTIMATOR_DIR> \
      --tokenizer_path <TOKENIZER_PATH> \
      --parquet_shards <FINEWEB_PARQUET> \
      --output_dir /tmp/dpqaq-fineweb-edu-v1 \
      --row_limit 50000 \
      --candidate_multiplier 8 \
      --bits 3 4 5 6 \
      --safe_nll_delta 0.02 \
      --profile_layer_group_size 4 \
      --confidence_threshold 0.6 \
      --fallback_bits 1 \
      --device cuda:0 \
      --local_files_only

The resulting records can be evaluated jointly with the existing collection:

    python scripts/predecode_predictors.py \
      --dataset_path artifacts/qaq-request-demand-preregistered-v1 \
        /tmp/dpqaq-fineweb-edu-v1/records.jsonl \
      --datasets wikitext2 c4_new fineweb_edu \
      --output_json /tmp/qaq-predecode-three-dataset.json \
      --seeds 17 29 43 \
      --trees 300 \
      --bootstrap_repetitions 1000

The third-dataset result is a generalization check, not evidence that a
scheduler is safe. Predictability still requires every registered dataset and
seed to pass the safe-bit, effective-bit, and profile gates.

