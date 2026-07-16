from __future__ import annotations

import math


def normalize(values: list[float], epsilon: float = 1e-12) -> list[float]:
    clipped = [max(0.0, float(value)) + epsilon for value in values]
    total = sum(clipped)
    return [value / total for value in clipped] if total > 0.0 else []


def kl_bits(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0.0 and qi > 0.0)


def js_bits(p: list[float], q: list[float]) -> float:
    midpoint = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
    return 0.5 * kl_bits(p, midpoint) + 0.5 * kl_bits(q, midpoint)


def rank_map(scores: list[float]) -> dict[int, int]:
    ordered = sorted(range(len(scores)), key=lambda index: (scores[index], -index), reverse=True)
    return {index: rank for rank, index in enumerate(ordered, start=1)}


def top_overlap(a: list[float], b: list[float], k: int) -> float:
    limit = min(max(0, int(k)), len(a), len(b))
    if limit <= 0:
        return 0.0
    a_top = set(sorted(range(len(a)), key=lambda index: a[index], reverse=True)[:limit])
    b_top = set(sorted(range(len(b)), key=lambda index: b[index], reverse=True)[:limit])
    return len(a_top & b_top) / float(limit)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / float(len(a | b))
