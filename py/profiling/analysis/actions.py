from __future__ import annotations

from typing import Any


def action_id(action: dict[str, Any]) -> str:
    return str(action.get("id", ""))


def action_payload(action: dict[str, Any]) -> dict[str, Any]:
    payload = action.get("payload")
    return payload if isinstance(payload, dict) else {}


def action_type(action: dict[str, Any]) -> str:
    payload = action_payload(action)
    return str(action.get("type") or payload.get("type") or "UNKNOWN").upper()


def action_field(action: dict[str, Any], *names: str) -> Any:
    payload = action_payload(action)
    for name in names:
        if name in action:
            return action.get(name)
        if name in payload:
            return payload.get(name)
    return None


def action_fingerprint(action: dict[str, Any]) -> str:
    typ = action_type(action)
    unit_id = action_field(action, "unit_id", "unitId", "u")
    city_id = action_field(action, "city_id", "cityId", "c")
    x = action_field(action, "x")
    y = action_field(action, "y")
    target = action_field(action, "target_unit_id", "targetUnitId", "target_id", "targetId", "tu")
    tech = action_field(action, "tech_id", "techId", "tech")
    unit_type = action_field(action, "unit_type", "unitType", "ut")
    building_type = action_field(action, "building_type", "buildingType", "bt")
    resource_type = action_field(action, "resource_type", "resourceType", "rt")
    parts = [typ]
    if unit_id is not None:
        parts.append(f"u={unit_id}")
    if city_id is not None:
        parts.append(f"c={city_id}")
    if target is not None:
        parts.append(f"target={target}")
    if x is not None:
        parts.append(f"x={x}")
    if y is not None:
        parts.append(f"y={y}")
    if tech is not None:
        parts.append(f"tech={tech}")
    if unit_type is not None:
        parts.append(f"unit={unit_type}")
    if building_type is not None:
        parts.append(f"bt={building_type}")
    if resource_type is not None:
        parts.append(f"rt={resource_type}")
    return ":".join(parts)


def top_mass_keys(scores: dict[str, float], threshold: float = 0.95) -> set[str]:
    total = sum(max(0.0, float(value)) for value in scores.values())
    if total <= 0.0:
        return set(scores)
    selected: set[str] = set()
    mass = 0.0
    for key, value in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True):
        selected.add(str(key))
        mass += max(0.0, float(value)) / total
        if mass >= threshold:
            break
    return selected

