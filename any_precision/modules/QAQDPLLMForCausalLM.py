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

        if router is None and router_checkpoint is not None:
            router, router_metadata = load_qaq_router_checkpoint(router_checkpoint)
        if router is None and router_mode in {"mlp_binary", "mlp_multibit"}:
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
        )

        if self.router is not None and self.router.num_layers != len(self.route_map):
            raise ValueError(
                f"Router checkpoint has {self.router.num_layers} routes, "
                f"but this model exposes {len(self.route_map)} quantized linear routes."
            )

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
            if linear_reg_d is None and os.path.exists(linear_reg_path):
                linear_reg_d = torch.load(linear_reg_path, map_location="cpu", weights_only=False)
            if jl_d is None and os.path.exists(jl_path):
                jl_d = torch.load(jl_path, map_location="cpu", weights_only=False)
            if max_mem_dict is None and os.path.exists(max_mem_path):
                max_mem_dict = torch.load(max_mem_path, map_location="cpu", weights_only=False)

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
            router_mode=router_mode,
            confidence_threshold=confidence_threshold,
            fallback_bits=fallback_bits,
            prefill_by_router=prefill_by_router,
            batch_policy=batch_policy,
        )

    def _load_quantized_modules(self, dtype=torch.float16, max_mem_dict=None, linear_reg_d=None, jl_d=None):
        max_mem_dict = max_mem_dict or {}
        linear_reg_d = linear_reg_d or {}
        jl_d = jl_d or {}
        layers = self.analyzer.get_layers()

        route_id = 0
        for layer_i, layer in enumerate(tqdm(layers, desc="Loading QAQ AP Layers")):
            named_linears = self.analyzer.get_modules(layer)

            for name, module in named_linears.items():
                parent_module, real_name = name.split(".")
                route_name = f"{layer_i}.{real_name}"

                if (layer_i, real_name) not in max_mem_dict:
                    max_mem_dict[(layer_i, real_name)] = max(self.precisions)

                est_linear, est_params = self._estimator_params(layer_i, real_name, linear_reg_d, jl_d)

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

    def _estimator_params(self, layer_i, real_name, linear_reg_d, jl_d):
        if self.router is None or not self.router.use_estimated_error:
            return None, None
        if (layer_i, real_name) in linear_reg_d:
            lin_params = linear_reg_d[(layer_i, real_name)]
            return True, (lin_params[0], lin_params[1])
        if (layer_i, real_name) in jl_d:
            return False, jl_d[(layer_i, real_name)]
        raise RuntimeError(
            f"Router checkpoint expects estimated-error features, but no estimator "
            f"parameters were provided for layer {layer_i} linear {real_name}."
        )

    def get_effective_bits(self):
        total_bits = 0
        total_comps = 0
        for linear in self.ap_linears:
            total_comps_temp = 0
            for bits in linear.comp_count.keys():
                comp = linear.comp_count[bits]
                total_bits += (comp * bits) * (linear.in_features * linear.out_features)
                total_comps_temp += comp
            if total_comps == 0:
                total_comps = total_comps_temp
        total_params = 0
        for linear in self.ap_linears:
            total_params += (linear.in_features * linear.out_features)
        return total_bits / (total_params * total_comps) if total_comps > 0 else 0

    def get_router_stats(self):
        total_tokens = 0
        total_selected_bits = 0
        total_fallbacks = 0
        per_layer = {}

        for linear in self.ap_linears:
            route_total = sum(linear.comp_count.values())
            bit_counts = {str(bit): int(count) for bit, count in linear.comp_count.items()}
            per_layer[linear.route_name] = {
                "bit_counts": bit_counts,
                "fallback_count": int(linear.fallback_count),
                "routed_token_count": int(linear.routed_token_count),
            }
            total_tokens += route_total
            total_selected_bits += sum(bit * count for bit, count in linear.comp_count.items())
            total_fallbacks += linear.fallback_count

        return {
            "average_selected_bit": total_selected_bits / total_tokens if total_tokens > 0 else 0,
            "effective_bits": self.get_effective_bits(),
            "fallback_fraction": total_fallbacks / total_tokens if total_tokens > 0 else 0,
            "total_tokens": int(total_tokens),
            "total_fallbacks": int(total_fallbacks),
            "per_layer": per_layer,
        }

    def clear_comp_count(self):
        for linear in self.ap_linears:
            linear.clear_stats()

    def clear_router_stats(self):
        self.clear_comp_count()

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
