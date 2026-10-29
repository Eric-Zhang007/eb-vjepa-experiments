"""Fixed-observation paraphrase behavior probe for Dream-JEPA.

It uses recorded LIBERO observations whose original instruction is known, keeps
the image history/proprio/noise seed identical, and swaps only the wording.
This measures semantic-preserving instruction stability. It deliberately does
not call the result task discrimination; a valid semantic-switch probe needs a
scene containing both competing targets.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from elaf.evaluation.dream_jepa import DreamJEPAPolicy


ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
OUT = ROOT / "experiments/instruction_shift_140k_20261026"
VIDEO = ROOT / "experiments/dream_jepa_failures_100k_20261025/videos"
POLICY = ROOT / "experiments/dream_jepa_eval_140k_20261026/policy.json"

CASES = {
    "libero_object_task7_trial0": "pick up the milk and place it in the basket",
    "libero_10_task3_trial2": "put the black bowl in the bottom drawer of the cabinet and close it",
    "libero_goal_task5_trial1": "push the plate to the front of the stove",
    "libero_goal_task6_trial2": "put the cream cheese in the bowl",
    "libero_spatial_task9_trial0": "pick up the black bowl on the wooden cabinet and place it on the plate",
}


def split_frame(frame):
    mid = frame.shape[1] // 2
    return np.ascontiguousarray(frame[:, :mid]), np.ascontiguousarray(frame[:, mid:])


def load_pairs():
    pairs = json.loads((OUT / "pairs.json").read_text())
    out = {}
    for x in pairs:
        out.setdefault(x["canonical"], []).append(x["paraphrase"])
    return out


def main():
    policy_cfg = json.loads(POLICY.read_text())
    policy_cfg["device"] = "cuda:0"
    policy_cfg["num_steps"] = 4
    policy_cfg["language_subgoals"] = False
    policy = DreamJEPAPolicy(policy_cfg)
    groups = load_pairs()
    result = {"protocol": {
        "type": "fixed_observation_paraphrase_stability",
        "checkpoint": policy_cfg["checkpoint"],
        "flow_steps": 4,
        "frames": [40, 80],
        "same_proprio": True,
        "same_noise_seed": True,
        "semantic_switch_probe": False,
    }, "cases": {}}

    for stem, canonical in CASES.items():
        jsonl = VIDEO / f"{stem}.jsonl"
        mp4 = VIDEO / f"{stem}.mp4"
        rows = [json.loads(x) for x in jsonl.read_text().splitlines()]
        spec = rows[0]["spec"]
        trace = {int(x["step"]): x for x in rows[1:]}
        frames = imageio.mimread(mp4, memtest=False)
        texts = [canonical] + groups[canonical]
        case = {"suite": spec["suite"], "task_id": spec["task_id"], "texts": texts, "frames": {}}
        for frame_idx in (40, 80):
            history = []
            for k in range(max(0, frame_idx - 3), frame_idx + 1):
                image, wrist = split_frame(frames[k])
                history.append({"image": image, "wrist_image": wrist})
            # The recorded frame index is one-based in the trace.
            tr = trace[max(1, frame_idx)]
            actions = []
            # Run one text at a time.  The model plus frozen JEPA encoder is
            # close to the 48 GB limit; the identical seed keeps the noise
            # draw matched without requiring a large inference batch.
            for text in texts:
                observation = {
                    "image": history[-1]["image"], "wrist_image": history[-1]["wrist_image"],
                    "history": history, "proprio": np.asarray(tr["proprio"], dtype=np.float32),
                    "instruction": text, "noise_seed": 100000 + frame_idx,
                    "suite": spec["suite"],
                }
                actions.append(policy.predict([observation])[0])
            base = actions[0].reshape(-1)
            rows_out = []
            for text, action in zip(texts[1:], actions[1:]):
                x = action.reshape(-1)
                denom = np.linalg.norm(base) + 1e-8
                rows_out.append({
                    "text": text,
                    "relative_l2_to_canonical": float(np.linalg.norm(x - base) / denom),
                    "cosine_to_canonical": float(np.dot(x, base) / ((np.linalg.norm(x) + 1e-8) * denom)),
                })
            case["frames"][str(frame_idx)] = rows_out
        result["cases"][stem] = case

    (OUT / "behavior_paraphrase_probe.json").write_text(json.dumps(result, indent=2))
    vals = [x["relative_l2_to_canonical"] for c in result["cases"].values() for f in c["frames"].values() for x in f]
    cos = [x["cosine_to_canonical"] for c in result["cases"].values() for f in c["frames"].values() for x in f]
    report = [
        "# Fixed-observation paraphrase behavior probe",
        "",
        "This tests whether meaning-preserving rewrites change the predicted action under identical observations and noise. It is not a task-discrimination test.",
        "",
        f"- Cases: {len(result['cases'])}; paraphrase comparisons: {len(vals)}",
        f"- Mean relative action change: {np.mean(vals):.4f}",
        f"- Median relative action change: {np.median(vals):.4f}",
        f"- Mean action cosine: {np.mean(cos):.4f}",
        f"- Minimum action cosine: {np.min(cos):.4f}",
        "",
        "A semantic-switch probe remains necessary: it must use the same scene with two valid competing targets and measure target/object/destination compliance.",
    ]
    (OUT / "behavior_paraphrase_probe.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
