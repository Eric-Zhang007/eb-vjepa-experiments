"""Paired empty-text ablation of exactly the original 40 Plus language trials."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
BASE = ROOT / "experiments/action_only/libero_20k_plus_280_shared_run"
OUT = ROOT / "experiments/action_only/libero_20k_plus_language_empty_40"


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare():
    import torch
    from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components
    torch.set_num_threads(8)
    baseline = json.loads((BASE / "summary.json").read_text())
    assert baseline["total"] == 280 and baseline["checkpoint_step"] == 20000
    parent = json.loads((BASE / "manifest.json").read_text())
    assert baseline["manifest_sha256"] == hashlib.sha256((BASE / "manifest.json").read_bytes()).hexdigest()
    manifest = {**parent, "entries": [e for e in parent["entries"]
                                    if e["category"] == "Language Instructions"],
                "episodes": 40, "instruction_mode": "empty", "parent_manifest_sha256": baseline["manifest_sha256"],
                "protocol": "Paired empty-string ablation of original 40 language trials"}
    assert len(manifest["entries"]) == 40 and all(e["trials"] == 1 for e in manifest["entries"])
    original = [e for e in baseline["episodes"] if e["category"] == "Language Instructions"]
    assert len(original) == 40 and sum(e["success"] for e in original) == 17
    assert {(e["index"], 0) for e in manifest["entries"]} == {(e["index"], e["trial"]) for e in original}
    assert all(e["seed"] == 239 + e["index"] * 5 + e["trial"] for e in original)
    checkpoint = ROOT / "experiments/action_only/train_final/checkpoints/weights/step_020000.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == baseline["checkpoint_sha256"]
    assert subprocess.check_output(["git", "-C", parent["source"], "rev-parse", "HEAD"],
                                   text=True).strip() == parent["source_commit"]
    atomic(OUT / "manifest.json", manifest)
    atomic(OUT / "baseline_language_40.json", original)
    cache = OUT / "text_cache"
    cache.mkdir()
    encoder, tokenizer = load_text_encoder_components(
        model_id="Wan-AI/Wan2.2-TI2V-5B", tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        context_len=128, device="cpu", dtype=torch.bfloat16)
    assert not torch.cuda.is_available(), "Preparation must be CPU-only"
    with torch.inference_mode():
        ids, mask = tokenizer([""], return_mask=True, add_special_tokens=True)
        mask = mask.bool()
        context = encoder(ids, mask)
        context[~mask] = 0
        assert context.shape == (1, 128, 4096) and mask.any() and torch.isfinite(context).all()
        filename = cache / (hashlib.sha256(b"").hexdigest() + ".t5_len128.wan22ti2v5b.pt")
        torch.save({"context": context[0].cpu(), "mask": mask[0].cpu(), "prompt": ""}, filename)
    atomic(OUT / "empty_text_report.json", {
        "prompt": "", "device": "cpu", "dtype": str(context.dtype),
        "active_token_ids": ids[mask].tolist(), "mask_tokens": int(mask.sum()),
        "context_norm": float(context.float().norm()),
        "cache_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        "special_tokens_preserved": True, "zero_embedding_substitution": False,
        "checkpoint_sha256": baseline["checkpoint_sha256"],
    })
    atomic(OUT / "preflight.json", {
        "status": "PASS",
        "manifest_sha256": hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest(),
        "parent_preflight": str(BASE / "preflight.json"),
        "scope": "Same validated tasks/assets; only text input changed; original indices/seeds retained",
    })
    print("EMPTY_TEXT_PREPARATION_PASS", flush=True)


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
    status_path = OUT / "status.json"
    if status_path.exists():
        raise RuntimeError("Existing ablation; refusing duplicate launch")
    state = {"phase": "preparing_cpu_text", "pid": os.getpid(), "started_unix": time.time(),
             "planned_episodes": 40, "checkpoint_step": 20000, "instruction_mode": "empty",
             "workers": []}
    def save(**extra):
        state.update(extra)
        atomic(status_path, state)
    save()
    active, pending, done = {}, [0, 1], set()
    prep = None
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
               "TMPDIR": str(ROOT / "cache/tmp"), "HF_HOME": str(ROOT / "cache/huggingface"),
               "DIFFSYNTH_MODEL_BASE_PATH": str(ROOT / "checkpoints/text"),
               "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
               "PYTHONPATH": f"{ROOT}/repos/RoboActualizer/src:{ROOT}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"}
        with (OUT / "prepare.log").open("a") as log:
            prep = subprocess.Popen([sys.executable, __file__, "--prepare"], env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            save(prepare_pid=prep.pid)
            if prep.wait(timeout=1800):
                raise RuntimeError("Empty-text preparation failed; see prepare.log")
        manifest = json.loads((OUT / "manifest.json").read_text())
        baseline = {(e["index"], e["trial"]): e for e in json.loads(
            (OUT / "baseline_language_40.json").read_text())}
        checkpoint_hash = json.loads((OUT / "empty_text_report.json").read_text())["checkpoint_sha256"]
        while pending or active:
            free = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
            free = {int(s.split(",")[0]): int(s.split(",")[1]) for s in free.splitlines()}
            for shard, (p, gpu) in list(active.items()):
                code = p.poll()
                if code is not None:
                    report = json.loads((OUT / f"shard_{shard}/report.json").read_text())
                    if code or report["phase"] != "complete":
                        raise RuntimeError(f"Evaluation shard {shard} failed")
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
                           "--instruction-mode", "empty", "--output", str(OUT)]
                worker_env = {**os.environ, "EVAL_GPU": str(gpu), "EVAL_SCRIPT": "eval_10k_robust.py",
                              "LIBERO_SOURCE": manifest["source"],
                              "LIBERO_CONFIG_PATH": str(ROOT / "experiments/action_only/plus_config")}
                with (OUT / f"shard_{shard}.log").open("a") as log:
                    p = subprocess.Popen(command, env=worker_env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=log)
                active[shard] = p, gpu
                state["workers"].append({"pid": p.pid, "gpu": gpu, "shard": shard, "command": command})
            count = sum(len(json.loads(p.read_text()).get("episodes", [])) for p in OUT.glob("shard_*/report.json"))
            save(phase="running" if active else "waiting_memory", completed_episodes=count,
                 completed_shards=sorted(done), pending_shards=pending)
            if time.time() - state["started_unix"] > 3 * 3600:
                raise TimeoutError("Three-hour ablation budget exhausted")
            time.sleep(10)
        episodes = []
        for shard in range(2):
            r = json.loads((OUT / f"shard_{shard}/report.json").read_text())
            assert r["phase"] == "complete" and r["checkpoint_sha256"] == checkpoint_hash
            assert r["instruction_mode"] == "empty" and r["text_encoder_input"] == ""
            assert len(r["episodes"]) == r["planned_episodes"] == 20
            episodes.extend(r["episodes"])
        assert len(episodes) == 40 and {(e["index"], e["trial"]) for e in episodes} == set(baseline)
        paired = []
        for e in episodes:
            original = baseline[e["index"], e["trial"]]
            for field in ("seed", "name", "language", "suite", "task_id"):
                assert original[field] == e[field]
            assert e["input_instruction"] == ""
            paired.append({**e, "original_success": original["success"]})
        success = sum(e["success"] for e in paired)
        summary = {"total": 40, "successes": success, "success_rate": success / 40,
                   "baseline_successes": 17, "baseline_success_rate": 17 / 40,
                   "delta_percentage_points": (success - 17) * 2.5,
                   "checkpoint_step": 20000, "checkpoint_sha256": checkpoint_hash,
                   "instruction_mode": "empty_string_encoded_by_real_umt5", "episodes": paired,
                   "paired_counts": {f"original_{a}_empty_{b}": sum(
                       e["original_success"] == bool(a) and e["success"] == bool(b) for e in paired)
                       for a in (0, 1) for b in (0, 1)}}
        atomic(OUT / "summary.json", summary)
        save(phase="complete", completed_episodes=40, success_rate=summary["success_rate"],
             successes=success, finished_unix=time.time())
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
