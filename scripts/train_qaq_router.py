import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import random
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from any_precision import AnyPrecisionForCausalLM
from any_precision.modules.QAQRouter import QAQRouter, save_qaq_router_checkpoint
from dp_llm_utils.dataset_tokenize import getDataLoader, getPossibleDatasets
from dp_llm_utils.model_def import getModelInfoFromConfig
from dp_llm_utils.record_x import getLayer0Inputs, getX, clearX

try:
    from any_precision_ext import dequant_kbit
except:
    dequant_kbit = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a QAQ-style MLP router from DP-LLM-style relative-error labels."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Original HF model path.")
    parser.add_argument("--ap_model_path", type=str, required=True, help="Any-Precision quantized model path.")
    parser.add_argument("--dataset", type=str, default="c4", choices=getPossibleDatasets())
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--dataset_length", type=int, default=40)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--reference_bit", type=int, default=None)
    parser.add_argument("--label_mode", choices=["multibit", "binary"], default="multibit")
    parser.add_argument("--target_bits", type=float, default=None)
    parser.add_argument("--error_threshold", type=float, default=0.01)
    parser.add_argument("--lambda_budget", type=float, default=0.0)
    parser.add_argument("--router_hidden_dim", type=int, default=256)
    parser.add_argument("--router_layers", type=int, default=2)
    parser.add_argument("--layer_embedding_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no_norm_feature", action="store_true")
    parser.add_argument("--include_estimated_error", action="store_true")
    parser.add_argument("--estimator_results", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--label_batch_size", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--max_tokens_per_linear", type=int, default=0)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_training_data", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32", "auto"], default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", action="store_false", dest="trust_remote_code")
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_route_map(layer_count: int, module_list: list[tuple[str, str]]) -> list[dict[str, Any]]:
    route_map = []
    route_id = 0
    for layer in range(layer_count):
        for parent, name in module_list:
            route_map.append({
                "route_id": route_id,
                "layer": layer,
                "parent": parent,
                "name": name,
                "route_name": f"{layer}.{name}",
            })
            route_id += 1
    return route_map


def route_lookup(route_map: list[dict[str, Any]]) -> dict[tuple[int, str], int]:
    return {(item["layer"], item["name"]): item["route_id"] for item in route_map}


def load_estimator_results(path: str | None):
    if path is None:
        return {}, {}
    linear_reg_path = os.path.join(path, "linear_reg_d.pt")
    jl_path = os.path.join(path, "jl_d.pt")
    linear_reg_d = torch.load(linear_reg_path, map_location="cpu", weights_only=False)
    jl_d = torch.load(jl_path, map_location="cpu", weights_only=False)
    return linear_reg_d, jl_d


def flatten_xarr(xarr: list[torch.Tensor], max_tokens: int, seed: int) -> torch.Tensor:
    x = torch.cat(xarr, dim=0).reshape(-1, xarr[0].shape[-1]).contiguous()
    if max_tokens > 0 and x.shape[0] > max_tokens:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        idx = torch.randperm(x.shape[0], generator=generator)[:max_tokens]
        x = x[idx].contiguous()
    return x


def dequant_weight(linear, bit: int, device: torch.device) -> torch.Tensor:
    qweight = linear.qweight.to(device)
    lut = linear._buffers[f"lut{bit}"].to(device)
    return dequant_kbit(qweight, lut, bit)


@torch.no_grad()
def make_labels_for_linear(
        x_cpu: torch.Tensor,
        linear,
        bits: list[int],
        reference_bit: int,
        error_threshold: float,
        label_mode: str,
        batch_size: int,
        device: torch.device,
        eps: float = 1e-8,
) -> torch.Tensor:
    bit_to_label = {bit: idx for idx, bit in enumerate(bits)}
    labels = torch.full((x_cpu.shape[0],), bit_to_label[reference_bit], dtype=torch.long)

    if label_mode == "binary":
        eval_bits = [bits[0]]
    else:
        eval_bits = bits

    assigned = torch.zeros(x_cpu.shape[0], dtype=torch.bool)
    w_ref = dequant_weight(linear, reference_bit, device)

    for bit in eval_bits:
        if label_mode == "multibit" and assigned.all():
            break
        if bit == reference_bit:
            if label_mode == "multibit":
                labels[~assigned] = bit_to_label[bit]
                assigned[~assigned] = True
            continue

        w_bit = dequant_weight(linear, bit, device)
        for start in range(0, x_cpu.shape[0], batch_size):
            stop = min(start + batch_size, x_cpu.shape[0])
            x = x_cpu[start:stop].to(device=device, dtype=w_ref.dtype)

            y_ref = torch.matmul(x, w_ref.T).float()
            y_bit = torch.matmul(x, w_bit.T).float()
            rel_error = (y_ref - y_bit).norm(dim=-1) / (y_ref.norm(dim=-1) + eps)
            safe = (rel_error <= error_threshold).cpu()

            if label_mode == "binary":
                low_label = bit_to_label[bit]
                high_label = bit_to_label[reference_bit]
                labels[start:stop] = torch.where(
                    safe,
                    torch.full_like(labels[start:stop], low_label),
                    torch.full_like(labels[start:stop], high_label),
                )
            else:
                new_assignments = safe & ~assigned[start:stop]
                label_slice = labels[start:stop]
                assigned_slice = assigned[start:stop]
                label_slice[new_assignments] = bit_to_label[bit]
                assigned_slice[new_assignments] = True
                labels[start:stop] = label_slice
                assigned[start:stop] = assigned_slice

        del w_bit
        torch.cuda.empty_cache()

    del w_ref
    torch.cuda.empty_cache()
    return labels


def estimator_for_linear(layer: int, name: str, linear_reg_d: dict, jl_d: dict, device: torch.device):
    key = (layer, name)
    if key in linear_reg_d:
        params = linear_reg_d[key]
        return "linear", (float(params[0]), float(params[1]))
    if key in jl_d:
        return "jl", jl_d[key].to(device)
    return None, None


def estimated_error_feature(x: torch.Tensor, estimator_type: str | None, estimator_params):
    if estimator_type is None:
        return None
    if estimator_type == "linear":
        slope, intercept = estimator_params
        return x.norm(dim=-1) * slope + intercept
    if estimator_type == "jl":
        return (x @ estimator_params.T).norm(dim=-1)
    raise RuntimeError(f"Unknown estimator type: {estimator_type}")


def train_router_on_linear(
        router: QAQRouter,
        optimizer: torch.optim.Optimizer,
        x_cpu: torch.Tensor,
        labels_cpu: torch.Tensor,
        route_id: int,
        bits_tensor: torch.Tensor,
        target_bits: float | None,
        lambda_budget: float,
        train_batch_size: int,
        device: torch.device,
        estimator_type: str | None,
        estimator_params,
) -> dict[str, float]:
    router.train()
    order = torch.randperm(x_cpu.shape[0])
    total_loss = 0.0
    total_ce = 0.0
    total_budget = 0.0
    total_examples = 0
    expected_bits_sum = 0.0

    for start in range(0, x_cpu.shape[0], train_batch_size):
        idx = order[start:start + train_batch_size]
        x = x_cpu[idx].to(device)
        labels = labels_cpu[idx].to(device)
        est_error = estimated_error_feature(x, estimator_type, estimator_params)

        optimizer.zero_grad(set_to_none=True)
        logits = router(x, route_id, estimated_error=est_error)
        ce_loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        expected_bits = (probs * bits_tensor).sum(dim=-1)

        if target_bits is not None and lambda_budget > 0:
            budget_loss = (expected_bits.mean() - target_bits) ** 2
            loss = ce_loss + lambda_budget * budget_loss
        else:
            budget_loss = torch.zeros_like(ce_loss)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        batch_n = labels.shape[0]
        total_examples += batch_n
        total_loss += loss.item() * batch_n
        total_ce += ce_loss.item() * batch_n
        total_budget += budget_loss.item() * batch_n
        expected_bits_sum += expected_bits.detach().sum().item()

    return {
        "loss": total_loss / total_examples,
        "ce_loss": total_ce / total_examples,
        "budget_loss": total_budget / total_examples,
        "expected_bits": expected_bits_sum / total_examples,
    }


def save_training_data(save_dir: str | None, route_info: dict[str, Any], x_cpu: torch.Tensor, labels: torch.Tensor):
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    filename = f"route_{route_info['route_id']:04d}_{route_info['route_name'].replace('.', '_')}.pt"
    torch.save(
        {
            "route_info": route_info,
            "x": x_cpu,
            "labels": labels,
        },
        os.path.join(save_dir, filename),
    )


def main():
    args = parse_args()
    if dequant_kbit is None:
        raise ModuleNotFoundError("Please install any_precision_ext from any_precision/modules/kernels first.")

    if len(args.bits) != len(set(args.bits)):
        raise ValueError("--bits must be unique")
    bits = sorted(args.bits)
    if args.label_mode == "binary" and len(bits) != 2:
        raise ValueError("--label_mode binary requires exactly two --bits values, e.g. --bits 3 6")
    reference_bit = args.reference_bit if args.reference_bit is not None else max(bits)
    if reference_bit not in bits:
        raise ValueError("--reference_bit must be one of --bits")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = resolve_dtype(args.torch_dtype)

    config = AutoConfig.from_pretrained(args.ap_model_path, trust_remote_code=args.trust_remote_code)
    model_info = getModelInfoFromConfig(config)
    layer_count = model_info["layer_count"]
    module_list = model_info["module_list"]
    route_map = build_route_map(layer_count, module_list)
    route_ids = route_lookup(route_map)

    if args.include_estimated_error and args.estimator_results is None:
        raise ValueError("--include_estimated_error requires --estimator_results")
    linear_reg_d, jl_d = load_estimator_results(args.estimator_results)

    router = QAQRouter(
        hidden_size=config.hidden_size,
        num_layers=len(route_map),
        bits=bits,
        router_hidden_dim=args.router_hidden_dim,
        router_layers=args.router_layers,
        layer_embedding_dim=args.layer_embedding_dim,
        use_norm_feature=not args.no_norm_feature,
        use_estimated_error=args.include_estimated_error,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bits_tensor = torch.tensor(bits, dtype=torch.float32, device=device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    model_orig = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cpu",
        use_cache=False,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    model_orig.config.use_cache = False

    model_ap = AnyPrecisionForCausalLM.from_quantized(
        args.ap_model_path,
        precisions=bits,
        trust_remote_code=args.trust_remote_code,
    ).eval()
    ap_layers = model_ap.get_model_layers()

    dataloader = getDataLoader(args.dataset, tokenizer, args.context_length, args.dataset_length)

    label_counts = defaultdict(lambda: defaultdict(int))
    train_metrics = []

    for epoch in range(args.epochs):
        layerX = None
        misc_inputs = None
        with tqdm(range(layer_count), desc=f"QAQ router epoch {epoch + 1}/{args.epochs}", unit="layer") as tbar:
            for layer in tbar:
                if layer == 0:
                    layerX, misc_inputs = getLayer0Inputs(model_orig, dataloader, device=args.device)
                xarr_d, layer_out_arr = getX(
                    model_orig,
                    layer,
                    module_list,
                    layerX,
                    misc_inputs,
                    device=args.device,
                )

                for parent, name in module_list:
                    route_id = route_ids[(layer, name)]
                    route_info = route_map[route_id]
                    tbar.set_description(f"epoch {epoch + 1}: labels {route_info['route_name']}")

                    x_cpu = flatten_xarr(
                        xarr_d[name],
                        max_tokens=args.max_tokens_per_linear,
                        seed=args.seed + epoch * len(route_map) + route_id,
                    )
                    linear = ap_layers[layer]._modules[parent]._modules[name]
                    labels = make_labels_for_linear(
                        x_cpu=x_cpu,
                        linear=linear,
                        bits=bits,
                        reference_bit=reference_bit,
                        error_threshold=args.error_threshold,
                        label_mode=args.label_mode,
                        batch_size=args.label_batch_size,
                        device=device,
                    )
                    save_training_data(args.save_training_data, route_info, x_cpu, labels)

                    estimator_type, estimator_params = estimator_for_linear(
                        layer,
                        name,
                        linear_reg_d,
                        jl_d,
                        device,
                    ) if args.include_estimated_error else (None, None)

                    metrics = train_router_on_linear(
                        router=router,
                        optimizer=optimizer,
                        x_cpu=x_cpu,
                        labels_cpu=labels,
                        route_id=route_id,
                        bits_tensor=bits_tensor,
                        target_bits=args.target_bits,
                        lambda_budget=args.lambda_budget,
                        train_batch_size=args.train_batch_size,
                        device=device,
                        estimator_type=estimator_type,
                        estimator_params=estimator_params,
                    )

                    counts = torch.bincount(labels, minlength=len(bits))
                    for bit, count in zip(bits, counts.tolist()):
                        label_counts[route_info["route_name"]][str(bit)] += int(count)

                    metrics.update({
                        "epoch": epoch,
                        "route_id": route_id,
                        "route_name": route_info["route_name"],
                        "tokens": int(labels.numel()),
                    })
                    train_metrics.append(metrics)
                    tbar.set_postfix(loss=f"{metrics['loss']:.4f}", eb=f"{metrics['expected_bits']:.3f}")

                    del x_cpu, labels
                    torch.cuda.empty_cache()

                layerX = layer_out_arr
                del xarr_d
                clearX(model_orig, layer, module_list)

    stats = {
        "label_counts": {route: dict(counts) for route, counts in label_counts.items()},
        "train_metrics": train_metrics,
    }
    training_config = vars(args)
    training_config["bits"] = bits
    training_config["reference_bit"] = reference_bit

    save_qaq_router_checkpoint(
        path=args.save_path,
        router=router.cpu().eval(),
        training_config=training_config,
        label_mode=args.label_mode,
        error_threshold=args.error_threshold,
        target_bits=args.target_bits,
        route_map=route_map,
        stats=stats,
    )

    json_path = f"{args.save_path}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "training_config": training_config,
                "route_map": route_map,
                "stats": stats,
            },
            f,
            indent=2,
        )
    print(f"Saved QAQ router checkpoint to {args.save_path}")
    print(f"Saved QAQ router metadata to {json_path}")


if __name__ == "__main__":
    main()
