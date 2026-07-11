# Autoregressive QAQ Decode Traces

`scripts/collect_qaq_decode_traces.py` collects real cached autoregressive
decodes for the QAQ ablations. Every prompt is run with the same tokenizer,
seed, and greedy decoding settings under every requested mode. The collector is
restricted to CUDA Device 0 so timing artifacts are comparable.

## Run

Set the physical device explicitly and run from the repository root:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 python scripts/collect_qaq_decode_traces.py \
  --ap_model_path /path/to/anyprec-llama3.1-8b \
  --router_checkpoint checkpoints/qaq_router_llama31_8b.pt \
  --estimator_results /path/to/estimator_results \
  --prompt "Explain mixed-precision inference in one sentence." \
  --bits 3 4 5 6 \
  --max_new_tokens 16 \
  --device cuda:0 \
  --output_jsonl artifacts/qaq_decode_trace.jsonl \
  --summary_json artifacts/qaq_decode_trace.summary.json
```

The default modes are `fixed_low`, `fixed_high`, `dp_threshold_only`,
`mlp_multibit`, and `mlp_multibit_dp_guard`. Use repeated `--prompt` or
`--prompt_file` to run a prompt set under every mode.

## Phase boundary and timing

The collector performs one cached prefill forward, selects the first output
token, records `prefill_router_stats`, and clears all router counters. It then
performs one cached forward per subsequent generated token and records only
those counters in `decode_router_stats`.

- `ttft_s` is synchronized prefill time through first-token selection.
- `decode_time_s` is synchronized time for decode forwards after prefill only.
- `tpot_s` is `decode_time_s / decode_token_count`.
- CUDA timing synchronizes before and after each measured region.
- `output_token_count` includes the first token selected from prefill;
  `decode_token_count` counts only later decode forwards.

Each record includes `per_token_route_profiles`, with selected bits by layer,
fallback counts, DP-guard counts, and per-layer event deltas. The aggregate
`decode_selected_bit_profile` is a coarse per-request profile derived from the
same per-token observations. Generated token IDs have both a sequence hash and
one hash per token.

## Quality boundary

This JSONL is a latency/routing trace, not a quality evaluation. It does not
run teacher forcing and its latency or generated-token hashes must not be used
to infer perplexity or other generation quality. Use the separate held-out
teacher-forced evaluation artifact from `scripts/evaluate_qaq_heldout.py` when
quality metrics are required.
