# QAQ Profile-Aware Batching Benchmark

scripts/benchmark_qaq_profile_batching.py replays a frozen held-out request
stream through six policies:

- fcfs: arrival/deadline dynamic batching with per-row grouped execution.
- scalar_predicted: buckets requests by predicted scalar effective bits.
- oracle_profile: groups by observed QAQ profile from frozen request records.
- predicted_profile: groups by held-out predicted profile.
- uncertainty_fallback: routes calibrated uncertain requests to fixed-high
  and groups the remaining requests by predicted profile.
- fixed_high: fixed-high safety baseline with FCFS batching.

All policies use identical request IDs, prompt windows, native continuation
lengths, arrival timestamps, candidate bits, and model artifacts. The profile
policies use the existing QAQDPLLM_Linear.batch_policy="max" path for
multi-request batches. FCFS and fixed-high use grouped execution. The installed
Any-Precision CUDA kernel caps the batch dimension at 8, so the benchmark uses
--max_batch_size 8.

## Reproducible run

The completed experiment used:

- 192 requests per trace: 48 each for prompt/continuation cells
  (128,32), (128,128), (512,32), and (512,128).
- Native 32- and 128-token continuations; --max_new_tokens 128.
- Arrival rate 1000 with seeds 101, 202, and 303.
- Predictor seed 17, 27 deliberately uncertain held-out requests per trace.
- One warmup batch and ten measured repeats per policy/trace.
- Synchronized CUDA timing around every manual cached prefill/decode batch.
- Continuous nvidia-smi sampling to a per-run CSV.

Run from the repository root on a free lab GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_qaq_profile_batching.py \
      --collection_dir artifacts/qaq-request-demand-preregistered-v1 \
      --analysis_json artifacts/qaq-request-demand-preregistered-v1-analysis/analysis.json \
      --ap_model_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --router_checkpoint checkpoints/qaq_router_llama31_8b_th005.pt \
      --estimator_results estimator_private_values/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512/finetuned_max6.0_3b-6b_th_pb_train_0.01_1.0_1ep_targ4.5b_init_0-40_adam \
      --tokenizer_path cache/packed/anyprec-(Meta-Llama-3.1-8B)-w6_orig3-gc1-c4_s100_blk512 \
      --datasets wikitext2 c4_new --request_limit 0 --min_uncertain_requests 1 \
      --max_new_tokens 128 --arrival_rate 1000 --arrival_seed 101 \
      --predictor_seed 17 --policies fcfs scalar_predicted oracle_profile \
      predicted_profile uncertainty_fallback fixed_high --max_batch_size 8 \
      --max_wait_ms 50 --warmup_batches 1 --repeat 10 \
      --confidence_threshold 0.6 --device cuda:0 --local_files_only \
      --output_json /tmp/qaq-profile-native128-r10-b8-seed101-all.json

For isolated policy runs, pass one value to --policies. Replace
--arrival_seed with 202 and 303 for the other traces. The JSON records the
exact stream hash, request IDs, arrival timestamps, uncertain request IDs,
warmup/repeat settings, CUDA environment, and per-policy measurements.

The seed-303 fixed-high process was retried on free GPU 6 after an initial
attempt on GPU 5 hit an out-of-memory condition caused by an unrelated process
already using that GPU. The retry used the same command and stream.

## Metrics and quality definition

The JSON reports p50/p95/p99 end-to-end latency, TTFT, TPOT, queue delay,
requests/sec, generated tokens/sec, token-slot throughput, request and prompt
occupancy, predictor/scheduler overhead, profile padding, effective bits,
average selected bit, per-layer bit histograms, confidence fallbacks, DP guard
triggers, uncertainty fallback rate, and quality violations.

quality_violation_rate is the QAQPrecisionAuditor route-decision
underprecision rate: real low-bit/reference-bit output-error labels determine
the smallest safe candidate bit, and the metric counts decisions where the
executed bit is lower. It is not task accuracy or perplexity. Quality auditing
is a separate CUDA-backed replay and is excluded from timed latency and
throughput.

profile_padding_bits is the mean component-wise padding from each request's
scheduler signal to the batch maximum. predictor_overhead_ms measures
consuming the held-out predictor outputs and applying bucket/fallback
decisions; predictor training is not timed.

## Output files

The benchmark writes the JSON requested by --output_json. The experiment
also captured per-run sampler CSVs under /tmp with the columns emitted by
nvidia-smi: timestamp, physical GPU index, model, memory used/total, GPU and
memory utilization, and power. The 18 native experiment JSONs and 18 sampler
files are intentionally outside the repository.