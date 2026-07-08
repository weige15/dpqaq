import os
from typing import Any

import torch
import torch.nn as nn


class QAQRouter(nn.Module):
    """MLP router for query-adaptive precision selection.

    The layer id is a routing id. In this codebase it is normally assigned to a
    concrete decoder-layer/linear-module pair, not only to a transformer block.
    """

    def __init__(
            self,
            hidden_size: int,
            num_layers: int,
            bits: list[int],
            input_feature_dim: int | None = None,
            router_hidden_dim: int = 256,
            router_layers: int = 2,
            layer_embedding_dim: int = 32,
            use_norm_feature: bool = True,
            use_estimated_error: bool = False,
            dropout: float = 0.0,
    ):
        super().__init__()
        if router_layers < 1:
            raise ValueError("router_layers must be >= 1")
        if len(bits) < 2:
            raise ValueError("QAQRouter needs at least two candidate bit-widths")
        if len(bits) != len(set(bits)):
            raise ValueError("bits must be unique")

        self.hidden_size = int(hidden_size)
        self.input_feature_dim = int(input_feature_dim) if input_feature_dim is not None else int(hidden_size)
        self.num_layers = int(num_layers)
        self.bits = [int(bit) for bit in bits]
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_layers = int(router_layers)
        self.layer_embedding_dim = int(layer_embedding_dim)
        self.use_norm_feature = bool(use_norm_feature)
        self.use_estimated_error = bool(use_estimated_error)
        self.dropout = float(dropout)

        self.layer_embedding = nn.Embedding(self.num_layers, self.layer_embedding_dim)

        scalar_dim = int(self.use_norm_feature) + int(self.use_estimated_error)
        input_dim = self.input_feature_dim + self.layer_embedding_dim + scalar_dim

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(self.router_layers - 1):
            layers.append(nn.Linear(current_dim, self.router_hidden_dim))
            layers.append(nn.GELU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            current_dim = self.router_hidden_dim
        layers.append(nn.Linear(current_dim, len(self.bits)))
        self.mlp = nn.Sequential(*layers)

    def _flat_hidden(self, x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        if feature_dim > self.input_feature_dim:
            raise ValueError(
                f"expected input feature dim <= {self.input_feature_dim}, got {feature_dim}"
            )
        flat_x = x.reshape(-1, feature_dim)
        if feature_dim < self.input_feature_dim:
            padded = flat_x.new_zeros(flat_x.shape[0], self.input_feature_dim)
            padded[:, :feature_dim] = flat_x
            flat_x = padded
        return flat_x

    def _flat_layer_ids(self, layer_ids: int | torch.Tensor, count: int, device: torch.device) -> torch.Tensor:
        if isinstance(layer_ids, int):
            layer_ids = torch.full((count,), layer_ids, dtype=torch.long, device=device)
        else:
            layer_ids = layer_ids.to(device=device, dtype=torch.long).reshape(-1)
            if layer_ids.numel() == 1:
                layer_ids = layer_ids.expand(count)
        if layer_ids.numel() != count:
            raise ValueError(f"expected {count} layer ids, got {layer_ids.numel()}")
        return layer_ids

    def _flat_scalar(self, scalar: torch.Tensor, count: int, name: str, device: torch.device) -> torch.Tensor:
        scalar = scalar.to(device=device, dtype=torch.float32).reshape(-1, 1)
        if scalar.shape[0] != count:
            raise ValueError(f"expected {count} values for {name}, got {scalar.shape[0]}")
        return scalar

    def forward(
            self,
            x: torch.Tensor,
            layer_ids: int | torch.Tensor,
            estimated_error: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat_x = self._flat_hidden(x).to(dtype=torch.float32)
        count = flat_x.shape[0]
        flat_layer_ids = self._flat_layer_ids(layer_ids, count, flat_x.device)

        features = [flat_x, self.layer_embedding(flat_layer_ids).to(dtype=torch.float32)]

        if self.use_norm_feature:
            norm_feature = torch.log1p(flat_x.norm(dim=-1, keepdim=True))
            features.append(norm_feature)

        if self.use_estimated_error:
            if estimated_error is None:
                raise ValueError("estimated_error is required for this router checkpoint")
            features.append(self._flat_scalar(estimated_error, count, "estimated_error", flat_x.device))

        router_input = torch.cat(features, dim=-1)
        return self.mlp(router_input)

    def config_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "input_feature_dim": self.input_feature_dim,
            "num_layers": self.num_layers,
            "bits": list(self.bits),
            "router_hidden_dim": self.router_hidden_dim,
            "router_layers": self.router_layers,
            "layer_embedding_dim": self.layer_embedding_dim,
            "use_norm_feature": self.use_norm_feature,
            "use_estimated_error": self.use_estimated_error,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QAQRouter":
        return cls(
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            bits=[int(bit) for bit in config["bits"]],
            input_feature_dim=int(config.get("input_feature_dim", config["hidden_size"])),
            router_hidden_dim=int(config.get("router_hidden_dim", 256)),
            router_layers=int(config.get("router_layers", 2)),
            layer_embedding_dim=int(config.get("layer_embedding_dim", 32)),
            use_norm_feature=bool(config.get("use_norm_feature", True)),
            use_estimated_error=bool(config.get("use_estimated_error", False)),
            dropout=float(config.get("dropout", 0.0)),
        )


def build_qaq_router_checkpoint(
        router: QAQRouter,
        training_config: dict[str, Any] | None = None,
        label_mode: str | None = None,
        error_threshold: float | None = None,
        target_bits: float | None = None,
        route_map: list[dict[str, Any]] | None = None,
        stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": "qaq_router_v1",
        "router_config": router.config_dict(),
        "router_state_dict": router.state_dict(),
        "candidate_bits": list(router.bits),
        "hidden_size": router.hidden_size,
        "input_feature_dim": router.input_feature_dim,
        "num_layers": router.num_layers,
        "training_config": training_config or {},
        "label_mode": label_mode,
        "error_threshold": error_threshold,
        "target_bits": target_bits,
        "route_map": route_map or [],
        "stats": stats or {},
    }


def save_qaq_router_checkpoint(
        path: str,
        router: QAQRouter,
        training_config: dict[str, Any] | None = None,
        label_mode: str | None = None,
        error_threshold: float | None = None,
        target_bits: float | None = None,
        route_map: list[dict[str, Any]] | None = None,
        stats: dict[str, Any] | None = None,
) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    checkpoint = build_qaq_router_checkpoint(
        router=router,
        training_config=training_config,
        label_mode=label_mode,
        error_threshold=error_threshold,
        target_bits=target_bits,
        route_map=route_map,
        stats=stats,
    )
    torch.save(checkpoint, path)


def load_qaq_router_checkpoint(path: str, map_location: str | torch.device = "cpu") -> tuple[QAQRouter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "qaq_router_v1":
        raise ValueError(f"Unsupported QAQ router checkpoint format in {path}")
    router = QAQRouter.from_config(checkpoint["router_config"])
    router.load_state_dict(checkpoint["router_state_dict"])
    router.eval()
    return router, checkpoint
