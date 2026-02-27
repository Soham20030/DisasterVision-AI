"""Priority score computation for damage tiles."""

# Severity weights: higher = more urgent response needed
SEVERITY_WEIGHTS = {
    "destroyed": 1.0,
    "major":     0.7,
    "minor":     0.4,
    "no-damage": 0.0,
}


def compute_priority(damage_class: str, confidence: float) -> float:
    """
    Priority = severity_weight × confidence
    Range: 0.0 (no damage, low confidence) → 1.0 (destroyed, 100% confident)
    """
    weight = SEVERITY_WEIGHTS.get(damage_class, 0.0)
    return round(weight * confidence, 4)


def compute_aggregate_stats(tiles: list) -> dict:
    """Compute region-level aggregate statistics from a list of TileResult dicts."""
    total = len(tiles)
    if total == 0:
        return {}

    class_counts = {"no-damage": 0, "minor": 0, "major": 0, "destroyed": 0}
    confidence_list = []
    priority_list = []

    for t in tiles:
        if t["status"] != "ok":
            continue
        cls = t["damage_class"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        confidence_list.append(t["confidence"])
        priority_list.append(t["priority_score"])

    ok_count = sum(class_counts.values())

    class_pct = {
        cls: round(count / ok_count * 100, 1) if ok_count > 0 else 0.0
        for cls, count in class_counts.items()
    }

    avg_confidence = round(sum(confidence_list) / len(confidence_list), 4) if confidence_list else 0.0
    avg_priority   = round(sum(priority_list) / len(priority_list), 4) if priority_list else 0.0

    # Overall severity: weighted average of severity weights
    weight_map = SEVERITY_WEIGHTS
    severity_scores = [weight_map.get(t["damage_class"], 0) for t in tiles if t["status"] == "ok"]
    avg_severity = round(sum(severity_scores) / len(severity_scores), 4) if severity_scores else 0.0

    return {
        "total_tiles": total,
        "ok_tiles": ok_count,
        "failed_tiles": total - ok_count,
        "class_counts": class_counts,
        "class_percentages": class_pct,
        "avg_confidence": avg_confidence,
        "avg_priority": avg_priority,
        "avg_severity_score": avg_severity,
    }
