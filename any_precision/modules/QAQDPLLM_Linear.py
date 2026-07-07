import warnings

import torch
import torch.nn as nn

try:
    from any_precision_ext import matmul_kbit, dequant_kbit
except:
    matmul_kbit, dequant_kbit = None, None


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
        self.est_linear = est_linear

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
            self.register_buffer("jl", est_params.to(dtype).to(device))
        else:
            self.jl = None

        self.comp_count = {bit: 0 for bit in self.precisions}
        self.fallback_count = 0
        self.routed_token_count = 0

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
        if self._is_prefill(x) and not self.prefill_by_router:
            y = self._fixed_precision_forward(x, self._max_valid_bit())
        else:
            y = self._router_forward(x)

        if self.bias is not None:
            y += self.bias

        return y

    def _is_prefill(self, x):
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

    def _fixed_precision_forward(self, x, bit):
        if self._is_prefill(x):
            weight = dequant_kbit(self.qweight, self._buffers[f'lut{bit}'], bit)
            y = torch.matmul(x, weight.T)
            self.comp_count[bit] += x.numel() // x.shape[-1]
            return y

        y = matmul_kbit(x, self.qweight, self._buffers[f'lut{bit}'], bit)
        self.comp_count[bit] += x.numel() // x.shape[-1]
        return y

    def _router_forward(self, x):
        if self.router_mode == "fixed_low":
            return self._fixed_precision_forward(x, self._min_valid_bit())
        if self.router_mode == "fixed_high":
            return self._fixed_precision_forward(x, self._max_valid_bit())
        if self.router_mode not in {"mlp_binary", "mlp_multibit"}:
            raise RuntimeError(f"Unsupported QAQ router mode: {self.router_mode}")
        if self.router is None:
            raise RuntimeError("MLP router mode requires a loaded QAQ router")

        original_shape = x.shape[:-1]
        flat_x = x.reshape(-1, x.shape[-1])
        chosen_bits, fallback_count = self._choose_bits(flat_x)
        self.fallback_count += int(fallback_count)
        self.routed_token_count += int(flat_x.shape[0])

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

        for bit in sorted(set(chosen_bits.detach().cpu().tolist())):
            mask = chosen_bits == bit
            rows = flat_x[mask].contiguous()
            if rows.numel() == 0:
                continue
            if rows.shape[0] > 8:
                weight = dequant_kbit(self.qweight, self._buffers[f'lut{bit}'], bit)
                y_flat[mask] = torch.matmul(rows, weight.T)
            else:
                y_flat[mask] = matmul_kbit(rows, self.qweight, self._buffers[f'lut{bit}'], bit)
            self.comp_count[bit] += int(mask.count_nonzero().item())

        return y_flat.reshape(*original_shape, self.out_features)

    def _choose_bits(self, flat_x):
        self._ensure_router_device(flat_x.device)
        estimated_error = self._estimated_error(flat_x) if self.router.use_estimated_error else None

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

    def _ensure_router_device(self, device):
        try:
            router_device = next(self.router.parameters()).device
        except StopIteration:
            router_device = device
        if router_device != device:
            self.router.to(device)

    def _estimated_error(self, flat_x):
        if self.est_linear is True:
            return flat_x.norm(dim=-1) * self.lin_slope + self.lin_inter
        if self.est_linear is False and self.jl is not None:
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

    def set_router_mode(self, router_mode):
        self.router_mode = router_mode

    def clear_stats(self):
        for bit in self.comp_count.keys():
            self.comp_count[bit] = 0
        self.fallback_count = 0
        self.routed_token_count = 0

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'route_id={self.route_id}, route_name={self.route_name}, bias={self.bias is not None}'
        )
