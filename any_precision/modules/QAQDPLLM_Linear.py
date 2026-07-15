import warnings
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    matmul_kbit, dequant_kbit = None, None


_PHASE_TIMING_KEYS = ("router", "estimator", "grouping", "dequant_matmul", "total")


class QAQDPLLM_Linear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            supported_bits,
            router,
            route_id,
            route_name,
            bias=True,
            precisions=None,
            device=None,
            dtype=None,
            maxmem=6,
            router_mode="mlp_multibit",
            confidence_threshold=None,
            fallback_bits=1,
            prefill_by_router=False,
            batch_policy="group",
            est_linear=None,
            est_params=None,
            est_T=None,
            b_l=None,
            b_h=None,
    ):
        super().__init__()
        if dequant_kbit is None or matmul_kbit is None:
            raise ModuleNotFoundError('Please install any precision CUDA kernel extension from modules/kernels.')
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            raise RuntimeError('supported_bits must be a list of integers.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = [int(bit) for bit in precisions]
        self.precision = max(self.precisions)
        self.supported_bits = [int(bit) for bit in supported_bits]
        self.maxmem = int(maxmem)
        self.route_id = int(route_id)
        self.route_name = route_name
        self.router_mode = router_mode
        self.confidence_threshold = confidence_threshold
        self.fallback_bits = int(fallback_bits)
        self.prefill_by_router = bool(prefill_by_router)
        self.batch_policy = batch_policy
        self.shared_precision = None
        self.est_linear = est_linear
        self.b_l = int(b_l) if b_l is not None else None
        self.b_h = int(b_h) if b_h is not None else None

        object.__setattr__(self, "_router", router)

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        for bit in supported_bits:
            self.register_buffer(
                f'lut{bit}',
                torch.empty((out_features, 2 ** bit), dtype=dtype, device=device)
            )

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        if est_linear is True:
            self.lin_slope, self.lin_inter = est_params
        elif est_linear is False and est_params is not None:
            self.jl = est_params.to(dtype=dtype) if dtype is not None else est_params
        else:
            self.jl = None

        self.est_T = torch.as_tensor(est_T, dtype=torch.float32) if est_T is not None else None

        self.comp_count = {bit: 0 for bit in self.precisions}
        self.fallback_count = 0
        self.dp_guard_count = 0
        self.dp_threshold_token_count = 0
        self.dp_threshold_high_count = 0
        self.routed_token_count = 0
        self.shared_profile_token_count = 0
        self.phase_timing_enabled = False
        self.decision_observer = None
        self.clear_phase_timing()

    @property
    def router(self):
        return object.__getattribute__(self, "_router")

    def prune_precisions(self):
        self.qweight = self.qweight[:min(max(self.precisions), self.maxmem)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                delattr(self, f'lut{bit}')
        self.precisions = [bit for bit in self.precisions if bit <= self.maxmem]
        self.comp_count = {bit: self.comp_count.get(bit, 0) for bit in self.precisions}

    def forward(self, x, **kwargs):
        if self._should_time_phases(x):
            with self._record_phase("total", x):
                y = self._forward_without_bias(x)
        else:
            y = self._forward_without_bias(x)

        if self.bias is not None:
            y += self.bias

        return y

    def _forward_without_bias(self, x):
        if self.router_mode == "shared_profile":
            return self._shared_profile_forward(x)
        if self._is_prefill(x) and not self.prefill_by_router:
            return self._fixed_precision_forward(x, self._max_valid_bit())
        return self._router_forward(x)

    def _is_prefill(self, x):
        if x.dim() >= 3:
            return x.shape[-2] > 1
        return x.numel() // x.shape[-1] > 1

    def _valid_bits(self):
        return [bit for bit in self.precisions if bit <= self.maxmem]

    def _max_valid_bit(self):
        valid_bits = self._valid_bits()
        if not valid_bits:
            raise RuntimeError(f"No valid precision remains for {self.route_name}")
        return max(valid_bits)

    def _min_valid_bit(self):
        valid_bits = self._valid_bits()
        if not valid_bits:
            raise RuntimeError(f"No valid precision remains for {self.route_name}")
        return min(valid_bits)

    def _requested_valid_bit(self):
        valid_bits = self._valid_bits()
        if not valid_bits:
            raise RuntimeError(f"No valid precision remains for {self.route_name}")
        lower_or_equal = [bit for bit in valid_bits if bit <= self.precision]
        return max(lower_or_equal) if lower_or_equal else min(valid_bits)


    def _fixed_precision_forward(self, x, bit):
        bit = int(bit)
        if bit not in self._valid_bits():
            raise RuntimeError(f"Precision {bit} is not valid for {self.route_name}: {self._valid_bits()}")
        flat_x = x.reshape(-1, x.shape[-1])
        self._notify_decision(
            flat_x,
            torch.full((flat_x.shape[0],), bit, dtype=torch.long, device=x.device),
        )
        if self._is_prefill(x):
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{bit}'], bit)
            y = torch.matmul(x, weight.T)
            self.comp_count[bit] += x.numel() // x.shape[-1]
            return y

        y = matmul_kbit(x, self.qweight, self._buffers[f'lut{bit}'], bit)
        self.comp_count[bit] += x.numel() // x.shape[-1]
        return y

    def _shared_profile_forward(self, x):
        if self.shared_precision is None:
            raise RuntimeError(
                f"shared_profile mode requires a supplied shared precision for {self.route_name}"
            )
        rows = x.numel() // x.shape[-1]
        y = self._fixed_precision_forward(x, self.shared_precision)
        self.shared_profile_token_count += rows
        return y

    def _router_forward(self, x):
        if self.router_mode == "fixed_low":
            return self._fixed_precision_forward(x, self._min_valid_bit())
        if self.router_mode == "fixed_high":
            return self._fixed_precision_forward(x, self._max_valid_bit())
        if self.router_mode == "fixed_precision":
            return self._fixed_precision_forward(x, self._requested_valid_bit())
        if self.router_mode not in {
            "mlp_binary",
            "mlp_multibit",
            "dp_threshold_only",
            "mlp_multibit_dp_guard",
        }:
            raise RuntimeError(f"Unsupported QAQ router mode: {self.router_mode}")
        if self.router_mode in {"mlp_binary", "mlp_multibit", "mlp_multibit_dp_guard"} and self.router is None:
            raise RuntimeError("MLP router mode requires a loaded QAQ router")

        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        chosen_bits = self._choose_mode_bits(flat_x)

        if self.batch_policy == "max" and chosen_bits.numel() > 1:
            max_bit = int(chosen_bits.max().item())
            warnings.warn(
                f"{self.route_name}: using max selected bit {max_bit} for the batch.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._fixed_precision_forward(x, max_bit)

        if self.batch_policy != "group":
            raise RuntimeError(f"Unsupported batch policy: {self.batch_policy}")

        y_flat = torch.empty(
            (flat_x.shape[0], self.out_features),
            dtype=x.dtype,
            device=x.device,
        )

        with self._record_phase("grouping", flat_x):
            selected_bits = sorted(set(chosen_bits.detach().cpu().tolist()))

        self._notify_decision(flat_x, chosen_bits)

        for bit in selected_bits:
            with self._record_phase("grouping", flat_x):
                mask = chosen_bits == bit
                rows = flat_x[mask].contiguous()
                row_count = int(mask.count_nonzero().item())
            if rows.numel() == 0:
                continue
            with self._record_phase("dequant_matmul", flat_x):
                if rows.shape[0] > 8:
                    weight = dequant_kbit(self.qweight, self._buffers[f'lut{bit}'], bit)
                    y_flat[mask] = torch.matmul(rows, weight.T)
                else:
                    y_flat[mask] = matmul_kbit(rows, self.qweight, self._buffers[f'lut{bit}'], bit)
            self.comp_count[bit] += row_count

        return y_flat.reshape(*original_shape, self.out_features)

    def _notify_decision(self, flat_x, chosen_bits):
        if self.decision_observer is not None:
            self.decision_observer(self, flat_x.detach(), chosen_bits.detach())

    def set_decision_observer(self, observer=None):
        """Attach an opt-in evaluator callback without changing routing decisions."""
        self.decision_observer = observer

    def _choose_mode_bits(self, flat_x):
        if self.router_mode == "dp_threshold_only":
            return self._choose_dp_threshold_bits(flat_x)

        if self.router_mode in {"mlp_binary", "mlp_multibit"}:
            chosen_bits, fallback_count = self._choose_router_bits(flat_x)
            self.fallback_count += int(fallback_count)
            self.routed_token_count += int(flat_x.shape[0])
            return chosen_bits

        if self.router_mode == "mlp_multibit_dp_guard":
            with self._record_phase("estimator", flat_x):
                estimated_error = self._estimated_error(flat_x)
            with self._record_phase("router", flat_x):
                router_bits, fallback_count = self._choose_router_bits(flat_x, estimated_error=estimated_error)
            dp_bits = self._choose_dp_threshold_bits(flat_x, estimated_error=estimated_error)
            chosen_bits = torch.maximum(router_bits, dp_bits)

            self.fallback_count += int(fallback_count)
            self.dp_guard_count += int((chosen_bits > router_bits).count_nonzero().item())
            self.routed_token_count += int(flat_x.shape[0])
            return chosen_bits

        raise RuntimeError(f"Unsupported QAQ router mode: {self.router_mode}")

    def _choose_router_bits(self, flat_x, estimated_error=None):
        self._ensure_router_device(flat_x.device)
        if self.router.use_estimated_error and estimated_error is None:
            estimated_error = self._estimated_error(flat_x)

        with torch.no_grad():
            logits = self.router(flat_x, self.route_id, estimated_error=estimated_error)
            probs = torch.softmax(logits, dim=-1)
            confidence, chosen_idx = probs.max(dim=-1)

            fallback_mask = torch.zeros_like(chosen_idx, dtype=torch.bool)
            if self.confidence_threshold is not None:
                fallback_mask = confidence < self.confidence_threshold
                if fallback_mask.any():
                    chosen_idx = chosen_idx.clone()
                    chosen_idx[fallback_mask] = torch.clamp(
                        chosen_idx[fallback_mask] + self.fallback_bits,
                        max=len(self.router.bits) - 1,
                    )

            chosen_bits = torch.tensor(self.router.bits, device=flat_x.device, dtype=torch.long)[chosen_idx]
            chosen_bits = self._clamp_to_valid_bits(chosen_bits)
            return chosen_bits, fallback_mask.count_nonzero().item()

    def _choose_dp_threshold_bits(self, flat_x, estimated_error=None):
        if self.b_l is None or self.b_h is None or self.est_T is None:
            raise RuntimeError(
                f"DP threshold mode for {self.route_name} requires b_l, b_h, and T_d threshold values."
            )
        if estimated_error is None:
            estimated_error = self._estimated_error(flat_x)

        threshold = self.est_T.to(device=flat_x.device, dtype=estimated_error.dtype)
        high_mask = estimated_error > threshold
        low = torch.full_like(high_mask, self.b_l, dtype=torch.long, device=flat_x.device)
        high = torch.full_like(high_mask, self.b_h, dtype=torch.long, device=flat_x.device)
        chosen_bits = torch.where(high_mask, high, low)
        chosen_bits = self._clamp_to_valid_bits(chosen_bits)

        self.dp_threshold_token_count += int(flat_x.shape[0])
        self.dp_threshold_high_count += int(high_mask.count_nonzero().item())
        return chosen_bits

    def _ensure_router_device(self, device):
        try:
            router_device = next(self.router.parameters()).device
        except StopIteration:
            router_device = device
        if router_device != device:
            self.router.to(device)

    def _estimated_error(self, flat_x):
        if self.est_linear is True:
            slope = self.lin_slope.to(flat_x.device) if isinstance(self.lin_slope, torch.Tensor) else self.lin_slope
            inter = self.lin_inter.to(flat_x.device) if isinstance(self.lin_inter, torch.Tensor) else self.lin_inter
            return flat_x.norm(dim=-1) * slope + inter
        if self.est_linear is False and self.jl is not None:
            if self.jl.device != flat_x.device or self.jl.dtype != flat_x.dtype:
                self.jl = self.jl.to(device=flat_x.device, dtype=flat_x.dtype)
            return (flat_x @ self.jl.T).norm(dim=-1)
        raise RuntimeError(
            f"Router for {self.route_name} expects estimated-error features, "
            "but no DP-style estimator parameters were provided."
        )

    def _clamp_to_valid_bits(self, chosen_bits):
        valid_bits = self._valid_bits()
        if not valid_bits:
            raise RuntimeError(f"No valid precision remains for {self.route_name}")

        clamped = chosen_bits.clone()
        for raw_bit in sorted(set(chosen_bits.detach().cpu().tolist())):
            if raw_bit in valid_bits:
                continue
            lower_or_equal = [bit for bit in valid_bits if bit <= raw_bit]
            replacement = max(lower_or_equal) if lower_or_equal else min(valid_bits)
            clamped[chosen_bits == raw_bit] = replacement
        return clamped

    def set_precision(self, precision):
        self.precision = precision

    def set_shared_precision(self, precision):
        if isinstance(precision, bool) or int(precision) != precision:
            raise ValueError(f"Shared precision for {self.route_name} must be an integer")
        precision = int(precision)
        if precision not in self._valid_bits():
            raise ValueError(
                f"Shared precision {precision} is not valid for {self.route_name}: {self._valid_bits()}"
            )
        self.shared_precision = precision

    def clear_shared_precision(self):
        self.shared_precision = None

    def set_router_mode(self, router_mode):
        self.router_mode = router_mode

    def clear_stats(self):
        for bit in self.comp_count.keys():
            self.comp_count[bit] = 0
        self.fallback_count = 0
        self.dp_guard_count = 0
        self.dp_threshold_token_count = 0
        self.dp_threshold_high_count = 0
        self.routed_token_count = 0
        self.shared_profile_token_count = 0
        self.clear_phase_timing()

    def set_phase_timing_enabled(self, enabled=True):
        self.phase_timing_enabled = bool(enabled)

    def clear_phase_timing(self):
        self._phase_wall_time_s = {key: 0.0 for key in _PHASE_TIMING_KEYS}
        self._phase_cuda_time_s = {key: 0.0 for key in _PHASE_TIMING_KEYS}
        self._phase_count = {key: 0 for key in _PHASE_TIMING_KEYS}
        self._phase_cuda_events = []

    def get_phase_timing_stats(self):
        self._flush_phase_timing()
        return {
            key: {
                "wall_time_s": float(self._phase_wall_time_s[key]),
                "cuda_time_s": float(self._phase_cuda_time_s[key]),
                "count": int(self._phase_count[key]),
                "mean_wall_ms": (
                    1000.0 * self._phase_wall_time_s[key] / self._phase_count[key]
                    if self._phase_count[key] > 0 else 0.0
                ),
                "mean_cuda_ms": (
                    1000.0 * self._phase_cuda_time_s[key] / self._phase_count[key]
                    if self._phase_count[key] > 0 else 0.0
                ),
            }
            for key in _PHASE_TIMING_KEYS
        }

    def _should_time_phases(self, x):
        return (
            self.phase_timing_enabled
            and self.router_mode == "mlp_multibit_dp_guard"
            and not (self._is_prefill(x) and not self.prefill_by_router)
        )

    @contextmanager
    def _record_phase(self, phase, reference_tensor):
        should_record = self._should_time_phases(reference_tensor)
        if not should_record:
            yield
            return

        wall_start = time.perf_counter()
        profiler_range = torch.profiler.record_function(f"qaq.{phase}")
        use_cuda_events = torch.cuda.is_available() and reference_tensor.device.type == "cuda"
        start_event = None
        end_event = None
        if use_cuda_events:
            stream = torch.cuda.current_stream(reference_tensor.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)

        try:
            with profiler_range:
                yield
        finally:
            if use_cuda_events:
                end_event.record(torch.cuda.current_stream(reference_tensor.device))
                self._phase_cuda_events.append((phase, start_event, end_event))
            self._phase_wall_time_s[phase] += time.perf_counter() - wall_start
            self._phase_count[phase] += 1

    def _flush_phase_timing(self):
        if not self._phase_cuda_events:
            return
        for phase, start_event, end_event in self._phase_cuda_events:
            end_event.synchronize()
            self._phase_cuda_time_s[phase] += start_event.elapsed_time(end_event) / 1000.0
        self._phase_cuda_events = []

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'route_id={self.route_id}, route_name={self.route_name}, bias={self.bias is not None}'
        )
