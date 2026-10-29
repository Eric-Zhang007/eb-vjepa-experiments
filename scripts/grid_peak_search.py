"""Discrete peak search with observed-shape guards; no unimodality guarantee."""
import math

GRID = tuple(range(1, 21))
TOLERANCE = 0.01


def decision(scores, maximum=20):
    if not 2 <= maximum <= 20:
        raise ValueError("Invalid grid maximum")
    grid = tuple(range(1, maximum + 1))
    if any(k not in grid or not math.isfinite(v) or not 0 <= v <= 1 for k, v in scores.items()):
        raise ValueError("Invalid search observation")
    for anchor in ((1 + maximum) // 2, 1, maximum):
        if anchor not in scores:
            return {"next": anchor, "reason": "anchor"}
    keys = sorted(scores)
    valley = any(scores[a] > scores[b] + TOLERANCE and scores[c] > scores[b] + TOLERANCE
                 for i, a in enumerate(keys) for j, b in enumerate(keys[i + 1:], i + 1)
                 for c in keys[j + 1:])
    left, right = 1, maximum
    fallback = "observed_nonunimodality" if valley else None
    while left < right and fallback is None:
        middle = (left + right) // 2
        for point in (middle, middle + 1):
            if point not in scores:
                return {"next": point, "reason": "binary_comparison", "interval": [left, right]}
        delta = scores[middle + 1] - scores[middle]
        if abs(delta) <= TOLERANCE:
            fallback = "adjacent_scores_within_1pp_do_not_discard_interval"
        elif delta > 0:
            left = middle + 1
        else:
            right = middle
    best = max(keys, key=lambda k: (scores[k], -k))
    if fallback:
        missing = sorted(set(grid) - scores.keys(), key=lambda k: (abs(k - best), k))
        if missing:
            return {"next": missing[0], "reason": fallback, "mode": "dense_guard"}
    for neighbor in (best - 1, best + 1):
        if neighbor in grid and neighbor not in scores:
            return {"next": neighbor, "reason": "verify_1k_neighbors"}
    return {"next": None, "best": best, "best_score": scores[best],
            "tested": keys, "grid_complete": len(scores) == len(grid),
            "interpretation": "Best observed on exploratory subsets; binary result conditional on unimodality; not a statistical/global optimum guarantee"}
