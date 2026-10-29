"""Small controlled semantic probe for the existing instruction cache.

This is deliberately a representation probe, not a claim about closed-loop
task understanding.  It reports whether semantic slots are linearly decodable
from frozen vectors and whether paraphrase consistency differs from semantic
slot changes.  The old nearest-canonical score is retained only as a baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
OUT = ROOT / "experiments/instruction_shift_140k_20261026"


# Hand-audited labels for the 20 canonical task families in pairs.json.
# These are semantic slots, not scene labels.
SLOTS = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate":
        dict(action="pick", object="black_bowl", source="between_plate_ramekin", destination="plate", state="open"),
    "pick up the black bowl on the cookie box and place it on the plate":
        dict(action="pick", object="black_bowl", source="on_cookie_box", destination="plate", state="open"),
    "pick up the black bowl on the ramekin and place it on the plate":
        dict(action="pick", object="black_bowl", source="on_ramekin", destination="plate", state="open"),
    "pick up the black bowl next to the cookie box and place it on the plate":
        dict(action="pick", object="black_bowl", source="next_cookie_box", destination="plate", state="open"),
    "pick up the black bowl next to the plate and place it on the plate":
        dict(action="pick", object="black_bowl", source="next_plate", destination="plate", state="open"),
    "pick up the black bowl on the wooden cabinet and place it on the plate":
        dict(action="pick", object="black_bowl", source="on_cabinet", destination="plate", state="open"),
    "pick up the alphabet soup and place it in the basket":
        dict(action="pick", object="alphabet_soup", source="none", destination="basket", state="open"),
    "pick up the cream cheese and place it in the basket":
        dict(action="pick", object="cream_cheese", source="none", destination="basket", state="open"),
    "pick up the tomato sauce and place it in the basket":
        dict(action="pick", object="tomato_sauce", source="none", destination="basket", state="open"),
    "pick up the milk and place it in the basket":
        dict(action="pick", object="milk", source="none", destination="basket", state="open"),
    "pick up the orange juice and place it in the basket":
        dict(action="pick", object="orange_juice", source="none", destination="basket", state="open"),
    "put the bowl on the stove":
        dict(action="put", object="bowl", source="none", destination="stove", state="open"),
    "put the wine bottle on top of the cabinet":
        dict(action="put", object="wine_bottle", source="none", destination="cabinet_top", state="open"),
    "put the bowl on top of the cabinet":
        dict(action="put", object="bowl", source="none", destination="cabinet_top", state="open"),
    "push the plate to the front of the stove":
        dict(action="push", object="plate", source="none", destination="stove_front", state="open"),
    "put the cream cheese in the bowl":
        dict(action="put", object="cream_cheese", source="none", destination="bowl", state="open"),
    "turn on the stove and put the moka pot on it":
        dict(action="turn_on_put", object="moka_pot", source="none", destination="stove", state="stove_on"),
    "put the black bowl in the bottom drawer of the cabinet and close it":
        dict(action="put_close", object="black_bowl", source="none", destination="bottom_drawer", state="closed"),
    "put the white mug on the left plate and put the yellow and white mug on the right plate":
        dict(action="put_two", object="two_mugs", source="none", destination="left_right_plates", state="open"),
    "put the white mug on the plate and put the chocolate pudding to the right of the plate":
        dict(action="put_two", object="mug_pudding", source="none", destination="plate_right", state="open"),
}


def normalize(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return (x - x.mean(0, keepdim=True)) / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)


def ridge_predict(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Dual ridge classifier; avoids inverting a 3k-dimensional matrix."""
    x_train = normalize(x_train)
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    x_test = (x_test.double() - mean) / std
    gram = x_train @ x_train.T
    alpha = torch.linalg.solve(gram + lam * torch.eye(len(x_train), dtype=gram.dtype), y_train.double())
    scores = (x_test @ x_train.T) @ alpha
    return scores.argmax(-1)


def task_groups(pairs):
    out = {}
    for p in pairs:
        out.setdefault(p["canonical"], []).append(p["paraphrase"])
    return out


def main():
    pairs = json.loads((OUT / "pairs.json").read_text())
    groups = task_groups(pairs)
    assert set(groups) == set(SLOTS), "semantic slot table is incomplete"
    eval_groups = {g: v for g, v in groups.items() if len(v) >= 2}
    dream = torch.load(OUT / "dream_features.pt", map_location="cpu", weights_only=True)["features"]
    umt5 = torch.load(OUT / "umt5_pooled.pt", map_location="cpu", weights_only=True)

    reps = {"umt5_instruction_only": umt5}
    reps["dream_input_embedding"] = dream["input_embedding"]
    for key in sorted(k for k in dream if k.startswith("scene")):
        reps["dream_" + key] = dream[key]

    # Each task has one canonical and two paraphrases.  Train on canonical +
    # paraphrase A, test on paraphrase B; then swap A/B.  This measures whether
    # a semantic slot survives a wording change, rather than raw cosine shape.
    report = {"protocol": {
        "description": "semantic-slot ridge probe with leave-one-paraphrase-out evaluation",
        "n_tasks": len(groups), "n_eval_tasks": len(eval_groups), "n_paraphrases": sum(map(len, groups.values())),
        "slots": ["action", "object", "source", "destination", "state"],
        "behavioral_probe": False,
    }, "representations": {}}
    canon = list(eval_groups)
    labels = {slot: {v: i for i, v in enumerate(sorted({SLOTS[g][slot] for g in canon}))} for slot in report["protocol"]["slots"]}
    for name, vectors in reps.items():
        result = {"slot_accuracy": {}, "full_task_accuracy": []}
        for held in (0, 1):
            train_text = canon + [eval_groups[g][1 - held] for g in canon]
            test_text = [eval_groups[g][held] for g in canon]
            x_train = torch.stack([vectors[t] for t in train_text])
            x_test = torch.stack([vectors[t] for t in test_text])
            y_task = torch.eye(len(canon), dtype=torch.float64).repeat(2, 1)
            result["full_task_accuracy"].append(float((ridge_predict(x_train, y_task, x_test) == torch.arange(len(canon))).double().mean()))
            for slot in report["protocol"]["slots"]:
                y = torch.zeros(len(train_text), len(labels[slot]), dtype=torch.float64)
                for i, text in enumerate(train_text):
                    canonical = text if text in SLOTS else next(g for g in canon if text in groups[g])
                    y[i, labels[slot][SLOTS[canonical][slot]]] = 1.0
                pred = ridge_predict(x_train, y, x_test)
                truth = torch.tensor([labels[slot][SLOTS[g][slot]] for g in canon])
                result["slot_accuracy"].setdefault(slot, []).append(float((pred == truth).double().mean()))
        result["full_task_accuracy_mean"] = sum(result["full_task_accuracy"]) / 2
        result["slot_accuracy_mean"] = {k: sum(v) / len(v) for k, v in result["slot_accuracy"].items()}
        report["representations"][name] = result

    # Keep the old retrieval number only as a descriptive baseline.
    old = json.loads((OUT / "results.json").read_text())
    report["old_retrieval_baseline"] = {k: old[k]["retrieval_accuracy"] for k in old if isinstance(old[k], dict) and "retrieval_accuracy" in old[k]}
    (OUT / "semantic_probe_results.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Instruction semantic probe",
        "",
        "This probe tests semantic-slot decodability under paraphrase holdout. It does not measure closed-loop task success.",
        "",
        "| representation | full task acc | action | object | source | destination | state |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in report["representations"].items():
        s = r["slot_accuracy_mean"]
        lines.append(f"| {name} | {r['full_task_accuracy_mean']:.3f} | {s['action']:.3f} | {s['object']:.3f} | {s['source']:.3f} | {s['destination']:.3f} | {s['state']:.3f} |")
    lines += [
        "", "The prior nearest-canonical score is retained as a descriptive baseline only. A behavioral probe with matched scenes and minimal semantic switches is still required before making a claim about task discrimination.",
    ]
    (OUT / "semantic_probe_report.md").write_text("\n".join(lines))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
