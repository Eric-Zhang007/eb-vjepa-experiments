"""Paired real-instruction permutation of the original 40 Plus language trials."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
BASE = ROOT / "experiments/action_only/libero_20k_plus_280_shared_run"
OUT = ROOT / "experiments/action_only/libero_20k_plus_language_swapped_40"


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def permute(entries):
    """Perfect matching preserves each suite's instruction multiset."""
    result = []
    rng = random.Random(239)
    for suite in sorted({e["suite"] for e in entries}):
        group = sorted((e for e in entries if e["suite"] == suite), key=lambda e: e["index"])
        assert all("_language_" in e["name"] for e in group)
        choices = {}
        for i, e in enumerate(group):
            choices[i] = [j for j, d in enumerate(group) if
                          d["name"].split("_language_")[0] != e["name"].split("_language_")[0]
                          and d["language"] != e["language"]]
            rng.shuffle(choices[i])
        matched = {}

        def assign(i, seen):
            for j in choices[i]:
                if j in seen:
                    continue
                seen.add(j)
                if j not in matched or assign(matched[j], seen):
                    matched[j] = i
                    return True
            return False

        for i in range(len(group)):
            if not assign(i, set()):
                raise RuntimeError(f"No different-task instruction permutation exists for {suite}")
        donors = {i: j for j, i in matched.items()}
        for i, e in enumerate(group):
            d = group[donors[i]]
            result.append({**e, "donor_index": d["index"], "donor_name": d["name"],
                           "swapped_language": d["language"]})
    return sorted(result, key=lambda e: e["index"])


def prepare():
    import torch
    from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    baseline = json.loads((BASE / "summary.json").read_text())
    parent = json.loads((BASE / "manifest.json").read_text())
    assert baseline["total"] == 280 and baseline["checkpoint_step"] == 20000
    assert baseline["manifest_sha256"] == hashlib.sha256((BASE / "manifest.json").read_bytes()).hexdigest()
    original = [e for e in baseline["episodes"] if e["category"] == "Language Instructions"]
    entries = permute([e for e in parent["entries"] if e["category"] == "Language Instructions"])
    assert len(entries) == len(original) == 40 and sum(e["success"] for e in original) == 17
    assert all(e["trials"] == 1 for e in entries)
    assert {(e["index"], 0) for e in entries} == {(e["index"], e["trial"]) for e in original}
    assert all(e["seed"] == 239 + e["index"] * 5 + e["trial"] for e in original)
    checkpoint = ROOT / "experiments/action_only/train_final/checkpoints/weights/step_020000.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == baseline["checkpoint_sha256"]
    assert subprocess.check_output(["git", "-C", parent["source"], "rev-parse", "HEAD"],
                                   text=True).strip() == parent["source_commit"]
    (OUT / "text_cache").symlink_to((BASE / "text_cache").resolve(), target_is_directory=True)
    for e in entries:
        prompt = DEFAULT_PROMPT.format(task=e["swapped_language"])
        path = OUT / "text_cache" / (hashlib.sha256(prompt.encode()).hexdigest() +
                                   ".t5_len128.wan22ti2v5b.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["prompt"] == prompt
        assert payload["context"].shape == (128, 4096)
        assert payload["mask"].any() and torch.isfinite(payload["context"]).all()
        e["text_cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for suite in {e["suite"] for e in entries}:
        group = [e for e in entries if e["suite"] == suite]
        assert Counter(e["language"] for e in group) == Counter(e["swapped_language"] for e in group)
    manifest = {**parent, "entries": entries, "episodes": 40, "instruction_mode": "swapped",
                "parent_manifest_sha256": baseline["manifest_sha256"],
                "checkpoint_sha256": baseline["checkpoint_sha256"],
                "protocol": "Seed-239 within-suite different-task perfect matching; original task success predicate"}
    atomic(OUT / "manifest.json", manifest)
    atomic(OUT / "baseline_language_40.json", original)
    atomic(OUT / "preflight.json", {
        "status": "PASS", "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
        "parent_preflight": str(BASE / "preflight.json"), "reused_original_gpu_bf16_text_cache": True,
        "scope": "Original tasks, states, seeds and inference settings; different-task real instructions only"})
    print("SWAPPED_TEXT_PREPARATION_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    OUT.mkdir(exist_ok=True)
    if args.detach:
        with (OUT / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__], start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        print(json.dumps({"supervisor_pid": child.pid}))
        return
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / "status.json").exists():
        raise RuntimeError("Existing ablation; refusing duplicate launch")
    state = {"phase": "preparing", "pid": os.getpid(), "started_unix": time.time(),
             "planned_episodes": 40, "checkpoint_step": 20000, "instruction_mode": "swapped",
             "workers": []}

    def save(**extra):
        state.update(extra)
        atomic(OUT / "status.json", state)

    save()
    active, pending, done = {}, [0, 1], set()
    prep = None
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
               "TMPDIR": str(ROOT / "cache/tmp"),
               "PYTHONPATH": f"{ROOT}/repos/RoboActualizer/src:{ROOT}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"}
        with (OUT / "prepare.log").open("a") as log:
            prep = subprocess.Popen([sys.executable, __file__, "--prepare"], env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            save(prepare_pid=prep.pid)
            if prep.wait(timeout=300):
                raise RuntimeError("Preparation failed; see prepare.log")
        manifest = json.loads((OUT / "manifest.json").read_text())
        baseline = {(e["index"], e["trial"]): e for e in json.loads(
            (OUT / "baseline_language_40.json").read_text())}
        while pending or active:
            free = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
            free = {int(s.split(",")[0]): int(s.split(",")[1]) for s in free.splitlines()}
            for shard, (p, gpu) in list(active.items()):
                code = p.poll()
                if code is not None:
                    report = json.loads((OUT / f"shard_{shard}/report.json").read_text())
                    if code or report["phase"] != "complete":
                        raise RuntimeError(f"Shard {shard} failed")
                    done.add(shard)
                    del active[shard]
            occupied = {gpu for p, gpu in active.values()}
            for gpu in (7, 5):
                if not pending or gpu in occupied or free[gpu] < 3650:
                    continue
                shard = pending.pop(0)
                command = ["bash", str(ROOT / "scripts/run_10k_libero_probe.sh"),
                           "--benchmark", "plus", "--step", "20000", "--shards", "2",
                           "--shard", str(shard), "--memory-fraction", "0.042",
                           "--instruction-mode", "swapped", "--output", str(OUT)]
                worker_env = {**os.environ, "EVAL_GPU": str(gpu), "EVAL_SCRIPT": "eval_10k_robust.py",
                              "LIBERO_SOURCE": manifest["source"],
                              "LIBERO_CONFIG_PATH": str(ROOT / "experiments/action_only/plus_config")}
                with (OUT / f"shard_{shard}.log").open("a") as log:
                    p = subprocess.Popen(command, env=worker_env, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
                active[shard] = p, gpu
                state["workers"].append({"pid": p.pid, "gpu": gpu, "shard": shard, "command": command})
            count = sum(len(json.loads(p.read_text()).get("episodes", [])) for p in OUT.glob("shard_*/report.json"))
            save(phase="running" if active else "waiting_memory", completed_episodes=count,
                 completed_shards=sorted(done), pending_shards=pending)
            if time.time() - state["started_unix"] > 3 * 3600:
                raise TimeoutError("Three-hour ablation budget exhausted")
            if pending or active:
                time.sleep(10)
        episodes = []
        for shard in range(2):
            r = json.loads((OUT / f"shard_{shard}/report.json").read_text())
            assert r["phase"] == "complete" and r["checkpoint_sha256"] == manifest["checkpoint_sha256"]
            assert r["instruction_mode"] == "swapped" and len(r["episodes"]) == r["planned_episodes"] == 20
            episodes.extend(r["episodes"])
        assert len(episodes) == 40 and {(e["index"], e["trial"]) for e in episodes} == set(baseline)
        for e in episodes:
            original = baseline[e["index"], e["trial"]]
            for field in ("seed", "name", "language", "suite", "task_id"):
                assert original[field] == e[field]
            assert e["input_instruction"] == e["swapped_language"] != e["language"]
            e["original_success"] = original["success"]
        successes = sum(e["success"] for e in episodes)
        summary = {"total": 40, "successes": successes, "success_rate": successes / 40,
                   "baseline_successes": 17, "baseline_success_rate": 17 / 40,
                   "delta_percentage_points": (successes - 17) * 2.5, "checkpoint_step": 20000,
                   "checkpoint_sha256": manifest["checkpoint_sha256"], "instruction_mode": "swapped",
                   "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
                   "episodes": episodes, "paired_counts": {f"original_{a}_swapped_{b}": sum(
                       e["original_success"] == bool(a) and e["success"] == bool(b) for e in episodes)
                       for a in (0, 1) for b in (0, 1)}}
        atomic(OUT / "summary.json", summary)
        save(phase="complete", completed_episodes=40, successes=successes,
             success_rate=successes / 40, finished_unix=time.time())
    except Exception as exc:
        children = [p for p, _ in active.values()] + ([prep] if prep else [])
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        save(phase="failed", error=repr(exc), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
