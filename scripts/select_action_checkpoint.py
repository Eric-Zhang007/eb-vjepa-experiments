"""Predeclared training-loss selector, not validation or control performance."""
import json
import math


def select_checkpoint(run, window=1000):
    rows = [json.loads(line) for line in (run / "train_metrics.jsonl").read_text().splitlines()]
    metrics = {r["step"]: r["train/action_loss"] for r in rows}
    candidates = []
    for checkpoint in sorted((run / "checkpoints/weights").glob("step_*.pt")):
        step = int(checkpoint.stem.split("_")[1])
        expected = list(range(step - window + 10, step + 1, 10))
        values = [metrics.get(s, float("nan")) for s in expected]
        if len(values) != 100 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"Incomplete/nonfinite 1000-step log window for {checkpoint}")
        candidates.append({"step": step, "mean_logged_action_loss": sum(values) / len(values),
                           "logged_samples": len(values), "checkpoint": str(checkpoint)})
    if not candidates:
        raise ValueError("No saved checkpoints")
    selected = min(candidates, key=lambda c: (c["mean_logged_action_loss"], c["step"]))
    (run / "checkpoint_selection.json").write_text(json.dumps({
        "rule": "lowest mean globally averaged logged train/action_loss in preceding 1000 optimizer steps; logs every10 steps; earlier step breaks ties",
        "not_validation": True, "candidates": candidates, "selected": selected,
        "fixed_comparator_step": 20000}, indent=2))
    return selected["step"]
