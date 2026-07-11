import gc
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate.big_modeling import init_empty_weights, load_checkpoint_and_dispatch

from .QAQDPLLM_Linear import QAQDPLLM_Linear
from .QAQRouter import QAQRouter, load_qaq_router_checkpoint
from any_precision.analyzer.analyzer import get_analyzer


def replace_module_by_name(layer, module_name, new_module):
    levels = module_name.split('.')
    module = layer
    for level in levels[:-1]:
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
    setattr(module, levels[-1], new_module)


class QAQDPLLMForCausalLM(nn.Module):
    def __init__(
            self,
            model_path,
            config,
            router=None,
            router_metadata=None,
            router_checkpoint=None,
            precisions=None,
            torch_dtype=torch.float16,
            fuse_layers=False,
            trust_remote_code=True,
            max_mem_dict=None,
            linear_reg_d=None,
            jl_d=None,
            T_d=None,
            router_mode="mlp_multibit",
            confidence_threshold=None,
            fallback_bits=1,
            prefill_by_router=False,
            batch_policy="group",
    ):
        super().__init__()

        self.config = config
        self.router_mode = router_mode
        self.confidence_threshold = confidence_threshold
        self.fallback_bits = fallback_bits
        self.prefill_by_router = prefill_by_router
        self.batch_policy = batch_policy

        max_mem_dict = max_mem_dict or {}
        linear_reg_d = linear_reg_d or {}
        jl_d = jl_d or {}
        T_d = T_d or {}

        if router is None and router_checkpoint is not None:
            router, router_metadata = load_qaq_router_checkpoint(router_checkpoint)
        if router is None and router_mode in {"mlp_binary", "mlp_multibit", "mlp_multibit_dp_guard"}:
            raise ValueError("router_checkpoint or router is required for MLP QAQ modes")
        if router is not None and not isinstance(router, QAQRouter):
            raise TypeError("router must be a QAQRouter")
        self.router = router
        self.router_metadata = router_metadata or {}

        self.supported_bits = list(range(self.config.anyprec['seed_precision'],
                                         self.config.anyprec['parent_precision'] + 1))
        if precisions is None:
            if self.router is not None:
                self.precisions = [bit for bit in self.router.bits if bit in self.supported_bits]
            else:
                self.precisions = self.supported_bits
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(bit in self.supported_bits for bit in precisions), \
                f"Supported bits {precisions} must be a subset of model supported bits {self.supported_bits}"
            self.precisions = [int(bit) for bit in precisions]

        if len(self.precisions) == 0:
            raise ValueError("No requested precision is supported by the quantized model")

        if self.router is not None:
            missing_bits = [bit for bit in self.precisions if bit not in self.router.bits]
            if missing_bits:
                raise ValueError(f"Router checkpoint does not support requested bits: {missing_bits}")

        self.precision = max(self.precisions)

        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                attn_implementation="flash_attention_2",
            )

        self.analyzer = get_analyzer(self.model)
        self.ap_linears = []
        self.route_map = []

        self._load_quantized_modules(
            dtype=torch_dtype,
            max_mem_dict=max_mem_dict,
            linear_reg_d=linear_reg_d,
            jl_d=jl_d,
            T_d=T_d,
        )

        if self.router is not None and self.router.num_layers != len(self.route_map):
            raise ValueError(
                f"Router checkpoint has {self.router.num_layers} routes, "
                f"but this model exposes {len(self.route_map)} quantized linear routes."
            )
        if self.router is not None:
            if "route_map" in self.router_metadata:
                self._validate_router_route_map(self.router_metadata["route_map"], self.route_map)
            elif router_checkpoint is not None:
                raise ValueError("Router checkpoint is missing route_map; cannot validate route order.")

        self.tie_weights()

        device_map = {key: 'cpu' for key in self.model.state_dict().keys()}

        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            dtype=torch_dtype,
        )

        if fuse_layers:
            self.fuse_layers()

        self.evaled = False
        self.prune_precisions()

    def forward(self, *args, **kwargs):
        prev_precision = self.precision
        prev_router_mode = self.router_mode
        precision = None
        router_mode = None

        if 'precision' in kwargs:
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        if 'router_mode' in kwargs:
            router_mode = kwargs.pop('router_mode')
            self.set_router_mode(router_mode)

        results = self.model.forward(*args, **kwargs)

        if precision is not None:
            self.set_precision(prev_precision)
        if router_mode is not None:
            self.set_router_mode(prev_router_mode)
        return results

    def generate(self, *args, **kwargs):
        precision = None
        router_mode = None
        prev_precision = self.precision
        prev_router_mode = self.router_mode

        if 'precision' in kwargs:
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        if 'router_mode' in kwargs:
            router_mode = kwargs.pop('router_mode')
            self.set_router_mode(router_mode)

        with torch.inference_mode():
            results = self.model.generate(*args, **kwargs)

        if precision is not None:
            self.set_precision(prev_precision)
        if router_mode is not None:
            self.set_router_mode(prev_router_mode)
        return results

    @staticmethod
    def _load_config(model_path, trust_remote_code=True):
        return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            router_checkpoint=None,
            router=None,
            router_metadata=None,
            trust_remote_code=True,
            fuse_layers=False,
            precisions=None,
            torch_dtype=torch.float16,
            max_mem_dict=None,
            linear_reg_d=None,
            jl_d=None,
            T_d=None,
            estimator_results=None,
            router_mode="mlp_multibit",
            confidence_threshold=None,
            fallback_bits=1,
            prefill_by_router=False,
            batch_policy="group",
    ):
        if estimator_results is not None:
            linear_reg_path = os.path.join(estimator_results, "linear_reg_d.pt")
            jl_path = os.path.join(estimator_results, "jl_d.pt")
            max_mem_path = os.path.join(estimator_results, "max_mem_dict.pt")
            T_path = os.path.join(estimator_results, "T_d.pt")
            if linear_reg_d is None and os.path.exists(linear_reg_path):
                linear_reg_d = torch.load(linear_reg_path, map_location="cpu", weights_only=False)
            if jl_d is None and os.path.exists(jl_path):
                jl_d = torch.load(jl_path, map_location="cpu", weights_only=False)
            if max_mem_dict is None and os.path.exists(max_mem_path):
                max_mem_dict = torch.load(max_mem_path, map_location="cpu", weights_only=False)
            if T_d is None and os.path.exists(T_path):
                T_d = torch.load(T_path, map_location="cpu", weights_only=False)

        config = cls._load_config(quant_model_path, trust_remote_code)
        return cls(
            model_path=quant_model_path,
            config=config,
            router_checkpoint=router_checkpoint,
            router=router,
            router_metadata=router_metadata,
            precisions=precisions,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            max_mem_dict=max_mem_dict,
            linear_reg_d=linear_reg_d,
            jl_d=jl_d,
            T_d=T_d,
            router_mode=router_mode,
            confidence_threshold=confidence_threshold,
            fallback_bits=fallback_bits,
            prefill_by_router=prefill_by_router,
            batch_policy=batch_policy,
        )

    def _load_quantized_modules(self, dtype=torch.float16, max_mem_dict=None, linear_reg_d=None, jl_d=None, T_d=None):
        max_mem_dict = max_mem_dict or {}
        linear_reg_d = linear_reg_d or {}
        jl_d = jl_d or {}
        T_d = T_d or {}
        requires_dp_threshold = self.router_mode in {"dp_threshold_only", "mlp_multibit_dp_guard"}
        layers = self.analyzer.get_layers()

        route_id = 0
        for layer_i, layer in enumerate(tqdm(layers, desc="Loading QAQ AP Layers")):
            named_linears = self.analyzer.get_modules(layer)

            for name, module in named_linears.items():
                parent_module, real_name = name.split(".")
                route_name = f"{layer_i}.{real_name}"

                if (layer_i, real_name) not in max_mem_dict:
                    max_mem_dict[(layer_i, real_name)] = max(self.precisions)

                est_linear, est_params = self._estimator_params(
                    layer_i,
                    real_name,
                    linear_reg_d,
                    jl_d,
                    required=requires_dp_threshold,
                )
                b_l, b_h, est_T = self._threshold_params(
                    layer_i,
                    real_name,
                    T_d,
                    required=requires_dp_threshold,
                )

                wqlinear = QAQDPLLM_Linear(
                    module.in_features,
                    module.out_features,
                    self.supported_bits,
                    router=self.router,
                    route_id=route_id,
                    route_name=route_name,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    dtype=dtype,
                    device=module.weight.device,
                    maxmem=max_mem_dict[(layer_i, real_name)],
                    router_mode=self.router_mode,
                    confidence_threshold=self.confidence_threshold,
                    fallback_bits=self.fallback_bits,
                    prefill_by_router=self.prefill_by_router,
                    batch_policy=self.batch_policy,
                    est_linear=est_linear,
                    est_params=est_params,
                    est_T=est_T,
                    b_l=b_l,
                    b_h=b_h,
                )
                self.ap_linears.append(wqlinear)
                self.route_map.append({
                    "route_id": route_id,
                    "layer": layer_i,
                    "parent": parent_module,
                    "name": real_name,
                    "route_name": route_name,
                })
                replace_module_by_name(layer, name, wqlinear)
                route_id += 1

            torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def _route_identity(route):
        required_fields = ("route_id", "layer", "parent", "name")
        missing = [field for field in required_fields if field not in route]
        if missing:
            raise ValueError(f"Router checkpoint route_map entry is missing fields: {missing}")

        layer = int(route["layer"])
        name = str(route["name"])
        return {
            "route_id": int(route["route_id"]),
            "layer": layer,
            "parent": str(route["parent"]),
            "name": name,
            "route_name": str(route.get("route_name", f"{layer}.{name}")),
        }

    @classmethod
    def _validate_router_route_map(cls, checkpoint_route_map, runtime_route_map):
        if not checkpoint_route_map:
            raise ValueError("Router checkpoint route_map is empty; cannot validate route order.")
        if len(checkpoint_route_map) != len(runtime_route_map):
            raise ValueError(
                f"Router checkpoint route_map has {len(checkpoint_route_map)} routes, "
                f"but this model exposes {len(runtime_route_map)} routes."
            )

        for idx, (checkpoint_route, runtime_route) in enumerate(zip(checkpoint_route_map, runtime_route_map)):
            checkpoint_identity = cls._route_identity(checkpoint_route)
            runtime_identity = cls._route_identity(runtime_route)
            if checkpoint_identity != runtime_identity:
                raise ValueError(
                    "Router checkpoint route_map mismatch at route index "
                    f"{idx}: checkpoint has {checkpoint_identity}, runtime has {runtime_identity}."
                )

    def _estimator_params(self, layer_i, real_name, linear_reg_d, jl_d, required=False):
        key = (layer_i, real_name)
        if key in linear_reg_d:
            lin_params = linear_reg_d[key]
            return True, (lin_params[0], lin_params[1])
        if key in jl_d:
            return False, jl_d[key]
        if required or (self.router is not None and self.router.use_estimated_error):
            raise RuntimeError(
                f"QAQ mode expects DP-style estimator parameters, but none were "
                f"provided for layer {layer_i} linear {real_name}."
            )
        return None, None

    def _threshold_params(self, layer_i, real_name, T_d, required=False):
        key = (layer_i, real_name)
        if key in T_d:
            b_l, b_h, est_T = T_d[key]
            return int(b_l), int(b_h), est_T
        if required:
            raise RuntimeError(
                f"DP threshold mode requires T_d values for layer {layer_i} linear {real_name}."
            )
        return None, None, None

    def get_effective_bits(self):
        total_weighted_bits = 0
        total_weighted_comps = 0
        for linear in self.ap_linears:
            param_count = linear.in_features * linear.out_features
            for bit, comp in linear.comp_count.items():
                total_weighted_bits += comp * bit * param_count
                total_weighted_comps += comp * param_count
        return total_weighted_bits / total_weighted_comps if total_weighted_comps > 0 else 0

    def get_router_stats(self):
        total_tokens = 0
        total_selected_bits = 0
        total_fallbacks = 0
        total_dp_guard_triggers = 0
        total_dp_threshold_tokens = 0
        total_dp_threshold_high = 0
        per_layer = {}
        phase_timing_totals = {}

        for linear in self.ap_linears:
            route_total = sum(linear.comp_count.values())
            bit_counts = {str(bit): int(count) for bit, count in linear.comp_count.items()}
            dp_guard_count = getattr(linear, "dp_guard_count", 0)
            dp_threshold_token_count = getattr(linear, "dp_threshold_token_count", 0)
            dp_threshold_high_count = getattr(linear, "dp_threshold_high_count", 0)
            layer_stats = {
                "bit_counts": bit_counts,
                "fallback_count": int(linear.fallback_count),
                "dp_guard_trigger_count": int(dp_guard_count),
                "dp_threshold_token_count": int(dp_threshold_token_count),
                "dp_threshold_high_count": int(dp_threshold_high_count),
                "routed_token_count": int(linear.routed_token_count),
            }

            if hasattr(linear, "get_phase_timing_stats"):
                phase_timing = linear.get_phase_timing_stats()
                if any(stats["count"] > 0 for stats in phase_timing.values()):
                    layer_stats["phase_timing"] = phase_timing
                    for phase, stats in phase_timing.items():
                        total = phase_timing_totals.setdefault(
                            phase, {"wall_time_s": 0.0, "cuda_time_s": 0.0, "count": 0}
                        )
                        total["wall_time_s"] += float(stats["wall_time_s"])
                        total["cuda_time_s"] += float(stats["cuda_time_s"])
                        total["count"] += int(stats["count"])

            per_layer[linear.route_name] = layer_stats
            total_tokens += route_total
            total_selected_bits += sum(bit * count for bit, count in linear.comp_count.items())
            total_fallbacks += linear.fallback_count
            total_dp_guard_triggers += dp_guard_count
            total_dp_threshold_tokens += dp_threshold_token_count
            total_dp_threshold_high += dp_threshold_high_count

        stats = {
            "average_selected_bit": total_selected_bits / total_tokens if total_tokens > 0 else 0,
            "effective_bits": self.get_effective_bits(),
            "fallback_fraction": total_fallbacks / total_tokens if total_tokens > 0 else 0,
            "dp_guard_trigger_fraction": total_dp_guard_triggers / total_tokens if total_tokens > 0 else 0,
            "dp_threshold_high_fraction": (
                total_dp_threshold_high / total_dp_threshold_tokens if total_dp_threshold_tokens > 0 else 0
            ),
            "total_tokens": int(total_tokens),
            "total_fallbacks": int(total_fallbacks),
            "total_dp_guard_triggers": int(total_dp_guard_triggers),
            "total_dp_threshold_tokens": int(total_dp_threshold_tokens),
            "total_dp_threshold_high": int(total_dp_threshold_high),
            "per_layer": per_layer,
        }
        if phase_timing_totals:
            stats["phase_timing"] = self._summarize_phase_timing(phase_timing_totals)
        return stats

    @staticmethod
    def _summarize_phase_timing(phase_timing_totals):
        return {
            phase: {
                "wall_time_s": float(values["wall_time_s"]),
                "cuda_time_s": float(values["cuda_time_s"]),
                "count": int(values["count"]),
                "mean_wall_ms": (
                    1000.0 * values["wall_time_s"] / values["count"]
                    if values["count"] > 0 else 0.0
                ),
                "mean_cuda_ms": (
                    1000.0 * values["cuda_time_s"] / values["count"]
                    if values["count"] > 0 else 0.0
                ),
            }
            for phase, values in phase_timing_totals.items()
        }

    def clear_comp_count(self):
        for linear in self.ap_linears:
            linear.clear_stats()

    def clear_router_stats(self):
        self.clear_comp_count()

    def set_phase_timing_enabled(self, enabled=True):
        for linear in self.ap_linears:
            if hasattr(linear, "set_phase_timing_enabled"):
                linear.set_phase_timing_enabled(enabled)

    def set_decision_observer(self, observer=None):
        for linear in self.ap_linears:
            linear.set_decision_observer(observer)

    def prune_precisions(self):
        for ap_linear in self.ap_linears:
            ap_linear.prune_precisions()
        torch.cuda.empty_cache()
        gc.collect()

    def set_precision(self, precision):
        for ap_linear in self.ap_linears:
            ap_linear.set_precision(precision)
        self.precision = precision

    def set_router_mode(self, router_mode):
        for ap_linear in self.ap_linears:
            ap_linear.set_router_mode(router_mode)
        self.router_mode = router_mode

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.anyprec['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.anyprec['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.model_config:
            raise NotImplementedError("This model does not support layer fusion")
        pass

    def eval(self):
        if not self.evaled:
            super().eval()
            self.evaled = True
        return self

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
