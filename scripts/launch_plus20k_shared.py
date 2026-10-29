"""Run fixed Plus-280 on spare GPUs only; never signal pretraining processes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/data/users/jiachangzhang/eb_vjepa")
OUT = ROOT / "experiments/action_only/libero_20k_plus_280_shared_run"
GPUS = (7, 5, 4, 3)
FRACTION = 0.042
# Measured first rollout: 1.54 GiB peak allocated / 1.56 GiB reserved.
# Cap torch at ~2.0 GiB, allowing 0.5 GiB driver/EGL + 1 GiB headroom.
MIN_FREE_MIB = 3650


def prepare_cfg_text():
    import torch
    from Roboactualizer.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from Roboactualizer.models.roboactualizer.runtime import load_text_encoder_components
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), "Text preparation must be CPU-only"
    prompt = DEFAULT_PROMPT.format(task="")
    encoder, tokenizer = load_text_encoder_components(
        model_id="Wan-AI/Wan2.2-TI2V-5B", tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        context_len=128, device="cpu", dtype=torch.bfloat16)
    with torch.inference_mode():
        ids, mask = tokenizer([prompt], return_mask=True, add_special_tokens=True)
        mask = mask.bool()
        context = encoder(ids, mask)
        context[~mask] = 0
        assert context.shape == (1, 128, 4096) and mask.any() and torch.isfinite(context).all()
        path = OUT / "text_cache" / (hashlib.sha256(prompt.encode()).hexdigest() +
                                   ".t5_len128.wan22ti2v5b.pt")
        assert not path.exists(), "Refusing cache overwrite"
        torch.save({"prompt": prompt, "context": context[0], "mask": mask[0]}, path)
    report = {"prompt": prompt, "device": "cpu", "dtype": str(context.dtype),
              "mask_tokens": int(mask.sum()), "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "context_norm": float(context.float().norm())}
    (OUT / "cfg_text_report.json").write_text(json.dumps(report, indent=2))
    print("CFG_TEXT_PASS", flush=True)


def gpu_free():
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in text.splitlines()}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--text-cfg-scale", type=float, choices=(1.0, 1.5), default=1.0)
    parser.add_argument("--prepare-cfg-text", action="store_true")
    args = parser.parse_args()
    if args.text_cfg_scale == 1.5:
        OUT = ROOT / "experiments/action_only/libero_20k_plus_280_cfg15"
    if args.prepare_cfg_text:
        assert args.text_cfg_scale == 1.5
        prepare_cfg_text()
        return
    OUT.mkdir(exist_ok=True)
    if args.detach:
        with (OUT / "supervisor.log").open("a") as log:
            child = subprocess.Popen([sys.executable, __file__, "--text-cfg-scale", str(args.text_cfg_scale)], start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        print(json.dumps({"supervisor_pid": child.pid}))
        return
    lock = (OUT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = OUT / "status.json"
    if path.exists():
        raise RuntimeError("Existing evaluation; refusing duplicate or silent retry")
    source = ROOT / "experiments/action_only/libero_10k_plus"
    manifest = json.loads((source / "manifest.json").read_text())
    preflight = json.loads((source / "preflight.json").read_text())
    digest = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    assert preflight["status"] == "PASS" and preflight["manifest_sha256"] == digest
    assert manifest["benchmark"] == "plus" and manifest["episodes"] == 280
    commit = subprocess.check_output(
        ["git", "-C", manifest["source"], "rev-parse", "HEAD"], text=True).strip()
    assert commit == manifest["source_commit"]
    ckpt = ROOT / "experiments/action_only/train_final/checkpoints/weights/step_020000.pt"
    checkpoint_hash = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    baseline = None
    if args.text_cfg_scale != 1.0:
        baseline = json.loads((ROOT / "experiments/action_only/libero_20k_plus_280_shared_run/summary.json").read_text())
        assert baseline["total"] == 280 and baseline["checkpoint_sha256"] == checkpoint_hash
        assert baseline["manifest_sha256"] == digest
    for name in ("manifest.json", "preflight.json"):
        shutil.copy2(source / name, OUT / name)
    if args.text_cfg_scale == 1.0:
        (OUT / "text_cache").symlink_to(source / "text_cache", target_is_directory=True)
    else:
        (OUT / "text_cache").mkdir()
        for cached in (source / "text_cache").glob("*.pt"):
            (OUT / "text_cache" / cached.name).symlink_to(cached.resolve())
    state = {"phase": "waiting_memory", "pid": os.getpid(), "started_unix": time.time(),
             "checkpoint_step": 20000, "checkpoint_sha256": checkpoint_hash,
             "manifest_sha256": digest, "allowed_gpus": list(GPUS), "shards": 4,
             "memory_fraction": FRACTION, "minimum_free_mib": MIN_FREE_MIB,
             "planned_episodes": 280, "workers": [], "text_cfg_scale": args.text_cfg_scale}
    def save(**extra):
        state.update(extra)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)
    save()
    active, pending, done = {}, list(range(4)), set()
    validated = False
    prep = None
    try:
        if args.text_cfg_scale != 1.0:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
                   "TMPDIR": str(ROOT / "cache/tmp"),
                   "DIFFSYNTH_MODEL_BASE_PATH": str(ROOT / "checkpoints/text"),
                   "DIFFSYNTH_DOWNLOAD_SOURCE": "modelscope",
                   "PYTHONPATH": f"{ROOT}/repos/RoboActualizer/src:{ROOT}/envs/eb_vjepa_pretrain/lib/python3.11/site-packages"}
            with (OUT / "prepare.log").open("a") as log:
                prep = subprocess.Popen([sys.executable, __file__, "--text-cfg-scale", "1.5",
                                         "--prepare-cfg-text"], env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=log)
                save(phase="preparing_cfg_text_cpu", prepare_pid=prep.pid)
                if prep.wait(timeout=1800):
                    raise RuntimeError("CFG text encoding failed; see prepare.log")
        while pending or active:
            free = gpu_free()
            for shard, (process, gpu) in list(active.items()):
                code = process.poll()
                report_path = OUT / f"shard_{shard}/report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                if report.get("episodes"):
                    validated = True
                if code is not None:
                    if code or report.get("phase") != "complete":
                        raise RuntimeError(f"Evaluation shard {shard} failed; pretraining untouched")
                    assert report["checkpoint_sha256"] == checkpoint_hash
                    done.add(shard)
                    del active[shard]
            occupied = {gpu for _, gpu in active.values()}
            for gpu in GPUS:
                if not pending or gpu in occupied or free[gpu] < MIN_FREE_MIB:
                    continue
                # First real rollout is the memory-fit gate, not a synthetic test.
                if not validated and (active or gpu != 7):
                    continue
                shard = pending.pop(0)
                command = ["bash", str(ROOT / "scripts/run_10k_libero_probe.sh"),
                           "--benchmark", "plus", "--step", "20000", "--shards", "4",
                           "--shard", str(shard), "--memory-fraction", str(FRACTION),
                           "--text-cfg-scale", str(args.text_cfg_scale),
                           "--output", str(OUT)]
                env = {**os.environ, "EVAL_GPU": str(gpu), "EVAL_SCRIPT": "eval_10k_robust.py",
                       "LIBERO_SOURCE": manifest["source"],
                       "LIBERO_CONFIG_PATH": str(ROOT / "experiments/action_only/plus_config")}
                with (OUT / f"shard_{shard}.log").open("a") as log:
                    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=log)
                active[shard] = process, gpu
                state["workers"].append({"shard": shard, "gpu": gpu, "pid": process.pid,
                                         "command": command, "free_mib_at_launch": free[gpu]})
                save(phase="running" if validated else "memory_fit_rollout")
                time.sleep(2)
            completed_episodes = 0
            for report_path in OUT.glob("shard_*/report.json"):
                completed_episodes += len(json.loads(report_path.read_text()).get("episodes", []))
            save(completed_shards=sorted(done), pending_shards=pending,
                 completed_episodes=completed_episodes, free_mib=free,
                 first_rollout_passed=validated)
            if time.time() - state["started_unix"] > 6 * 3600:
                raise TimeoutError("Six-hour evaluation budget; preserve partial results")
            time.sleep(10)
        episodes = []
        for shard in range(4):
            report = json.loads((OUT / f"shard_{shard}/report.json").read_text())
            assert report["phase"] == "complete" and report["checkpoint_sha256"] == checkpoint_hash
            assert len(report["episodes"]) == report["planned_episodes"]
            assert report["text_cfg_scale"] == args.text_cfg_scale
            episodes.extend(report["episodes"])
        expected = {(e["index"], trial) for e in manifest["entries"] for trial in range(e["trials"])}
        assert {(e["index"], e["trial"]) for e in episodes} == expected
        assert len(episodes) == len(expected) == 280
        summary = {"episodes": episodes, "total": 280, "successes": sum(e["success"] for e in episodes),
                   "checkpoint_step": 20000, "checkpoint_sha256": checkpoint_hash,
                   "manifest_sha256": digest, "protocol": manifest["protocol"],
                   "text_cfg_scale": args.text_cfg_scale}
        if baseline is not None:
            reference = {(e["index"], e["trial"]): e for e in baseline["episodes"]}
            for e in episodes:
                original = reference[e["index"], e["trial"]]
                for field in ("seed", "name", "language", "suite", "task_id"):
                    assert original[field] == e[field]
                assert e["input_instruction"] == e["language"] and e["text_cfg_scale"] == 1.5
                e["baseline_success"] = original["success"]
            summary["baseline_successes"] = baseline["successes"]
            summary["paired_counts"] = {f"cfg1_{a}_cfg15_{b}": sum(
                e["baseline_success"] == bool(a) and e["success"] == bool(b) for e in episodes)
                for a in (0, 1) for b in (0, 1)}
        summary["success_rate"] = summary["successes"] / 280
        for field in ("category", "suite"):
            summary[field + "_breakdown"] = {
                key: {"episodes": sum(e[field] == key for e in episodes),
                      "successes": sum(e[field] == key and e["success"] for e in episodes)}
                for key in sorted({e[field] for e in episodes})}
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        save(phase="complete", successes=summary["successes"],
             success_rate=summary["success_rate"], finished_unix=time.time())
    except Exception as exc:
        if prep is not None and prep.poll() is None:
            prep.terminate()
            try:
                prep.wait(timeout=30)
            except subprocess.TimeoutExpired:
                prep.kill()
                prep.wait()
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, _ in active.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        save(phase="failed", error=repr(exc), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
