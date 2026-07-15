"""Validation and accounting helpers for scheduler-supplied QAQ profiles."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


PROJECTION_RULE = (
    "smallest route-valid bit >= demand; minimum valid bit below minimum; "
    "maximum valid bit above maximum"
)


def _profile_items(profile: Mapping[str, Any] | Sequence[tuple[str, Any]], name: str):
    if isinstance(profile, Mapping):
        return [(str(key), value) for key, value in profile.items()]
    try:
        items = list(profile)
    except TypeError as error:
        raise ValueError(f"{name} must be a mapping or a sequence of route/value pairs") from error
    normalized = []
    for item in items:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError(f"{name} must contain route/value pairs")
        normalized.append((str(item[0]), item[1]))
    return normalized


def validate_group_profile(
    profile: Sequence[float],
    *,
    expected_dimension: int | None = None,
    layer_group_size: int | None = None,
) -> tuple[float, ...]:
    """Validate one continuous group-demand profile without changing values."""
    if layer_group_size is not None and int(layer_group_size) < 1:
        raise ValueError("layer_group_size must be positive")
    values = tuple(float(value) for value in profile)
    if not values:
        raise ValueError("group profile must be non-empty")
    if expected_dimension is not None and len(values) != int(expected_dimension):
        raise ValueError(
            f"group profile dimension {len(values)} disagrees with expected dimension {expected_dimension}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("group profile values must be finite")
    return values


def _request_profiles(
    request_group_profiles: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    expected_dimension: int | None = None,
    layer_group_size: int | None = None,
) -> list[tuple[str, tuple[float, ...]]]:
    items = _profile_items(request_group_profiles, "request_group_profiles")
    if not items:
        raise ValueError("request_group_profiles must be non-empty")
    result = []
    seen = set()
    for request_id, profile in items:
        if request_id in seen:
            raise ValueError(f"duplicate request profile: {request_id}")
        seen.add(request_id)
        result.append((request_id, validate_group_profile(
            profile,
            expected_dimension=expected_dimension,
            layer_group_size=layer_group_size,
        )))
    dimensions = {len(profile) for _, profile in result}
    if len(dimensions) != 1:
        raise ValueError("request group profiles have inconsistent dimensions")
    return result


def compose_max_group_profile(
    request_group_profiles: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    *,
    expected_dimension: int | None = None,
    layer_group_size: int | None = None,
) -> tuple[float, ...]:
    """Compose one continuous component-wise maximum across the actual batch."""
    profiles = _request_profiles(
        request_group_profiles,
        expected_dimension=expected_dimension,
        layer_group_size=layer_group_size,
    )
    return tuple(max(profile[index] for _, profile in profiles) for index in range(len(profiles[0][1])))


def validate_route_map(
    route_map: Sequence[Mapping[str, Any]],
    *,
    layer_group_size: int | None = None,
    profile_dimension: int | None = None,
) -> list[dict[str, Any]]:
    """Validate structured route identities and optional layer-group coverage."""
    if not route_map:
        raise ValueError("route_map must be non-empty")
    normalized = []
    route_ids = set()
    route_names = set()
    for route in route_map:
        if not isinstance(route, Mapping):
            raise ValueError("route_map entries must be mappings")
        missing = [field for field in ("route_id", "layer") if field not in route]
        if missing:
            raise ValueError(f"route_map entry is missing fields: {missing}")
        route_id = int(route["route_id"])
        layer = int(route["layer"])
        route_name = str(route.get("route_name", f"{layer}.{route.get('name', '')}"))
        if route_id in route_ids or route_name in route_names:
            raise ValueError(f"duplicate route in route_map: {route_name}")
        if layer < 0:
            raise ValueError(f"route layer must be non-negative: {route_name}")
        route_ids.add(route_id)
        route_names.add(route_name)
        normalized.append({**dict(route), "route_id": route_id, "layer": layer, "route_name": route_name})

    if layer_group_size is not None:
        layer_group_size = int(layer_group_size)
        if layer_group_size < 1:
            raise ValueError("layer_group_size must be positive")
        groups = {route["layer"] // layer_group_size for route in normalized}
        if profile_dimension is None:
            profile_dimension = max(groups) + 1
        profile_dimension = int(profile_dimension)
        if profile_dimension < 1:
            raise ValueError("profile dimension must be positive")
        invalid = sorted(group for group in groups if group >= profile_dimension)
        if invalid:
            raise ValueError(f"route group indices outside profile: {invalid}")
        expected_groups = set(range(profile_dimension))
        if groups != expected_groups:
            raise ValueError(
                f"route groups {sorted(groups)} do not provide complete profile groups "
                f"0..{profile_dimension - 1}"
            )
    return normalized


def validate_route_profile(
    route_profile: Mapping[str, int] | Sequence[tuple[str, int]],
    route_map: Sequence[Mapping[str, Any]],
    route_valid_bits: Mapping[str, Sequence[int]],
) -> dict[str, int]:
    """Validate a complete route profile against each route's actual valid bits."""
    routes = validate_route_map(route_map)
    items = _profile_items(route_profile, "route_profile")
    values = {}
    for route_name, raw_bit in items:
        if route_name in values:
            raise ValueError(f"duplicate route in route_profile: {route_name}")
        values[route_name] = raw_bit
    expected = [route["route_name"] for route in routes]
    missing = [route for route in expected if route not in values]
    unknown = [route for route in values if route not in set(expected)]
    if missing:
        raise ValueError(f"route_profile is missing routes: {missing}")
    if unknown:
        raise ValueError(f"route_profile contains unknown routes: {unknown}")
    result = {}
    for route_name in expected:
        try:
            raw_value = values[route_name]
            if isinstance(raw_value, bool) or int(raw_value) != raw_value:
                raise ValueError
            bit = int(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"route bit for {route_name} must be an integer") from error
        valid = sorted({int(candidate) for candidate in route_valid_bits.get(route_name, ())})
        if not valid:
            raise ValueError(f"route {route_name} has no valid bits")
        if bit not in valid:
            raise ValueError(f"route bit {bit} for {route_name} is not in valid bits {valid}")
        result[route_name] = bit
    extra_valid = set(route_valid_bits) - set(expected)
    if extra_valid:
        raise ValueError(f"route_valid_bits contains unknown routes: {sorted(extra_valid)}")
    return result


def project_demand_to_valid_bit(demand: float, valid_bits: Sequence[int]) -> tuple[int, bool]:
    """Project a continuous demand upward onto one route's valid-bit set."""
    demand = float(demand)
    if not math.isfinite(demand):
        raise ValueError("profile demand must be finite")
    valid = sorted({int(bit) for bit in valid_bits})
    if not valid:
        raise ValueError("route valid bits must be non-empty")
    if demand <= valid[0]:
        return valid[0], False
    for bit in valid:
        if bit >= demand:
            return bit, False
    return valid[-1], True


def project_group_profile_to_routes(
    group_profile: Sequence[float],
    route_map: Sequence[Mapping[str, Any]],
    route_valid_bits: Mapping[str, Sequence[int]],
    *,
    layer_group_size: int,
) -> dict[str, Any]:
    """Project group demands to a complete route-level profile and report ceilings."""
    group_profile = validate_group_profile(group_profile, layer_group_size=layer_group_size)
    routes = validate_route_map(
        route_map,
        layer_group_size=layer_group_size,
        profile_dimension=len(group_profile),
    )
    route_profile = {}
    route_group_indices = {}
    capped_routes = {}
    known_routes = {route["route_name"] for route in routes}
    extra_valid = set(route_valid_bits) - known_routes
    if extra_valid:
        raise ValueError(f"route_valid_bits contains unknown routes: {sorted(extra_valid)}")
    for route in routes:
        route_name = route["route_name"]
        group_index = route["layer"] // int(layer_group_size)
        route_group_indices[route_name] = group_index
        bit, capped = project_demand_to_valid_bit(group_profile[group_index], route_valid_bits.get(route_name, ()))
        route_profile[route_name] = bit
        if capped:
            capped_routes[route_name] = {
                "demand": float(group_profile[group_index]),
                "max_valid_bit": max(int(value) for value in route_valid_bits[route_name]),
            }
    return {
        "route_profile": route_profile,
        "route_group_indices": route_group_indices,
        "capped_routes": capped_routes,
        "capped_route_count": len(capped_routes),
        "projection_rule": PROJECTION_RULE,
    }


def build_max_shared_profile(
    request_group_profiles: Mapping[str, Sequence[float]] | Sequence[tuple[str, Sequence[float]]],
    layer_group_size: int,
    route_map: Sequence[Mapping[str, Any]],
    route_valid_bits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Build the immutable batch profile used by max-profile sharing."""
    profiles = _request_profiles(request_group_profiles, layer_group_size=layer_group_size)
    dimension = len(profiles[0][1])
    shared_group = compose_max_group_profile(profiles, expected_dimension=dimension, layer_group_size=layer_group_size)
    projection = project_group_profile_to_routes(
        shared_group, route_map, route_valid_bits, layer_group_size=layer_group_size
    )
    request_projected = {}
    request_groups = {}
    for request_id, profile in profiles:
        request_groups[request_id] = list(profile)
        request_projected[request_id] = project_group_profile_to_routes(
            profile, route_map, route_valid_bits, layer_group_size=layer_group_size
        )["route_profile"]
    return {
        "composition_policy": "max_profile_sharing",
        "profile_source": "predicted",
        "layer_group_size": int(layer_group_size),
        "profile_dimension": dimension,
        "request_group_profiles": request_groups,
        "shared_group_profile": list(shared_group),
        "shared_route_profile": projection["route_profile"],
        "request_projected_route_profiles": request_projected,
        "route_group_indices": projection["route_group_indices"],
        "capped_routes": projection["capped_routes"],
        "capped_route_count": projection["capped_route_count"],
        "projection_rule": projection["projection_rule"],
    }


def account_profile_execution(
    shared_profile: Mapping[str, Any],
    executed_route_profile: Mapping[str, int],
    route_map: Sequence[Mapping[str, Any]],
    route_valid_bits: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Compare actual executed route bits with each request's projected target."""
    executed = validate_route_profile(executed_route_profile, route_map, route_valid_bits)
    request_targets = shared_profile.get("request_projected_route_profiles")
    if not isinstance(request_targets, Mapping) or not request_targets:
        raise ValueError("shared profile has no request projected route profiles")
    under = exact = over = 0
    signed_sum = absolute_sum = 0
    per_request = {}
    for request_id, target_profile in request_targets.items():
        target = validate_route_profile(target_profile, route_map, route_valid_bits)
        request_under = request_exact = request_over = 0
        request_signed = request_absolute = 0
        for route in (item["route_name"] for item in validate_route_map(route_map)):
            signed_gap = executed[route] - target[route]
            absolute_gap = abs(signed_gap)
            signed_sum += signed_gap
            absolute_sum += absolute_gap
            request_signed += signed_gap
            request_absolute += absolute_gap
            if signed_gap < 0:
                under += 1
                request_under += 1
            elif signed_gap > 0:
                over += 1
                request_over += 1
            else:
                exact += 1
                request_exact += 1
        per_request[str(request_id)] = _accounting_summary(
            request_under, request_exact, request_over, request_signed, request_absolute
        )
    return _accounting_summary(under, exact, over, signed_sum, absolute_sum, per_request=per_request)


def _accounting_summary(
    under: int,
    exact: int,
    over: int,
    signed_sum: int,
    absolute_sum: int,
    *,
    per_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    total = under + exact + over
    result = {
        "decision_count": total,
        "profile_under_precision_count": under,
        "profile_exact_precision_count": exact,
        "profile_over_precision_count": over,
        "profile_under_precision_rate": under / total if total else 0.0,
        "profile_exact_precision_rate": exact / total if total else 0.0,
        "profile_over_precision_rate": over / total if total else 0.0,
        "signed_bit_gap_sum": signed_sum,
        "absolute_bit_gap_sum": absolute_sum,
        "mean_signed_bit_gap": signed_sum / total if total else 0.0,
        "mean_absolute_bit_gap": absolute_sum / total if total else 0.0,
    }
    if per_request is not None:
        result["per_request"] = dict(per_request)
    return result


def aggregate_profile_accounting(accounts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate batch-level scheduler-profile accounting without reusing auditor names."""
    keys = (
        "decision_count",
        "profile_under_precision_count",
        "profile_exact_precision_count",
        "profile_over_precision_count",
        "signed_bit_gap_sum",
        "absolute_bit_gap_sum",
    )
    totals = {key: sum(int(account.get(key, 0)) for account in accounts) for key in keys}
    return _accounting_summary(
        totals["profile_under_precision_count"],
        totals["profile_exact_precision_count"],
        totals["profile_over_precision_count"],
        totals["signed_bit_gap_sum"],
        totals["absolute_bit_gap_sum"],
    )


def route_profile_from_stats(router_stats: Mapping[str, Any]) -> dict[str, int]:
    """Extract one actual bit per route from shared-profile execution statistics."""
    result = {}
    for route, stats in router_stats.get("per_layer", {}).items():
        active = [int(bit) for bit, count in stats.get("bit_counts", {}).items() if int(count) > 0]
        if len(active) != 1:
            raise ValueError(f"shared execution must use exactly one bit for route {route}, got {active}")
        result[str(route)] = active[0]
    if not result:
        raise ValueError("router statistics contain no executed routes")
    return result
