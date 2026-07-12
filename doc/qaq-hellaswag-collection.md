# HellaSwag Source Extension

scripts/collect_qaq_hellaswag_request_demand.py adds a fourth real source to
the request-demand collection using cached Rowan/hellaswag train parquet.

The collector:

- treats each HellaSwag source_id as one source document;
- concatenates only the real ctx_a and ctx_b fields in source-row order;
- excludes answer endings and labels from the input text;
- partitions source documents with the existing document hash;
- selects exact 32/8/24 development/calibration/test quotas in four short
  length cells: 32/16, 32/32, 64/16, and 64/32 prompt/continuation tokens;
- records source-ID hashes, row hashes, dataset revision, shard hash, and
  model provenance;
- reuses the shared real QAQ collection callback for all targets.

The cached source revision is
218ec52e09a7e7462a5400043bb9a69a41d06b76 and the train parquet SHA-256 is
cacb12587faa63d7f723a72d61d12bfa94b140446f5a6a0a2e1c6906ab88bf02.

Manifest preflight:

    CUDA_VISIBLE_DEVICES=0 python scripts/collect_qaq_hellaswag_request_demand.py \
      --ap_model_path <AP_MODEL_PATH> \
      --router_checkpoint <ROUTER_CHECKPOINT> \
      --estimator_results <ESTIMATOR_DIR> \
      --tokenizer_path <TOKENIZER_PATH> \
      --parquet_shards <HELLASWAG_TRAIN_PARQUET> \
      --output_dir /tmp/dpqaq-hellaswag-v1 \
      --row_limit 39905 --candidate_multiplier 8 \
      --bits 3 4 5 6 --device cuda:0 \
      --local_files_only --manifest_only

The real collection produced 256 validated v2 records with 256 unique source
documents and 128/32/96 development/calibration/test requests. The artifact
is outside the repository at /tmp/dpqaq-hellaswag-v1. Answer fields are not
used as prompt features or targets.
