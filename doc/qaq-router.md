# QAQ-Style Router

This repo adds a trainable QAQ-style precision router on top of DP-LLM / Any-Precision weights.

The router is trained from real relative-error labels. For each captured linear input `x`, the training script dequantizes the Any-Precision weights at candidate bits and labels the sample with the smallest bit whose output error is below the threshold:

```text
rel_error_b = ||W_ref x - W_b x|| / (||W_ref x|| + eps)
label = min bit b where rel_error_b <= error_threshold
```

In binary mode, pass exactly two bits, for example `--bits 3 6`, and the label is low bit when the low-bit relative error is safe, otherwise high bit.

## Train

Install dependencies and the CUDA extension first:

```bash
cd /nfs/home/s314511048/dpqaq
pip install -r requirements.txt
pip install -e .
cd any_precision/modules/kernels
pip install .
```

Run `0_set_configs.py` once if the Any-Precision model config does not already contain `anyprec.size_d`:

```bash
cd /nfs/home/s314511048/dpqaq
python 0_set_configs.py /path/to/anyprec-llama3.1-8b
```

Train the router on a real calibration subset:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 python scripts/train_qaq_router.py \
  --model_path meta-llama/Llama-3.1-8B \
  --ap_model_path /path/to/anyprec-llama3.1-8b \
  --dataset c4 \
  --context_length 512 \
  --dataset_length 40 \
  --bits 3 4 5 6 \
  --error_threshold 0.01 \
  --target_bits 4.5 \
  --lambda_budget 0.1 \
  --router_hidden_dim 256 \
  --router_layers 2 \
  --save_path checkpoints/qaq_router_llama31_8b.pt \
  --device cuda
```

If you want the router to consume DP-style estimated-error features, first create DP-LLM estimator files, then add:

```bash
  --include_estimated_error \
  --estimator_results /path/to/estimator_results
```

## Inference

Load the QAQ model directly:

```python
from any_precision import QAQDPLLMForCausalLM

model = QAQDPLLMForCausalLM.from_quantized(
    "/path/to/anyprec-llama3.1-8b",
    router_checkpoint="checkpoints/qaq_router_llama31_8b.pt",
    precisions=[3, 4, 5, 6],
    confidence_threshold=0.6,
    fallback_bits=1,
)
```

During prefill, the QAQ path uses max valid precision by default. During decoding, it routes each row independently. If a batch has mixed selected bits, rows are grouped by bit and computed with separate `matmul_kbit` calls.

Run generation sanity checks:

```bash
cd /nfs/home/s314511048/dpqaq
CUDA_VISIBLE_DEVICES=0 python scripts/run_qaq_inference.py \
  --ap_model_path /path/to/anyprec-llama3.1-8b \
  --router_checkpoint checkpoints/qaq_router_llama31_8b.pt \
  --estimator_results /path/to/estimator_results \
  --bits 3 4 5 6 \
  --prompt "Explain mixed-precision inference in one sentence." \
  --max_new_tokens 16 \
  --confidence_threshold 0.6 \
  --device cuda \
  --output_json qaq_inference_stats.json
```

The output reports average selected bit-width, effective bits, per-layer bit counts, fallback fraction, latency, and whether logits are finite.

## Checkpoint Contents

Router checkpoints contain:

- `router_state_dict`
- candidate bits
- hidden size
- number of routing ids
- route map from decoder layer and linear name to route id
- training config
- label mode
- error threshold
- target bits
- label and training stats

## Limitations

This workspace does not include a local Any-Precision Llama 3.1 8B checkpoint, so full training and generation validation must run on the GPU server with the real `--ap_model_path`. The local static check is still useful, but it does not prove CUDA extension availability or model-quality behavior.
