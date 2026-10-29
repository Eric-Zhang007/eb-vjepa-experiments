"""Spawn independent simulators, batch ready observations, record valid episodes."""
import argparse
from collections import deque
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import time
import signal

from elaf.evaluation.rpc import PolicyClient


def episode_key(spec):
    return json.dumps([spec[k] for k in ("benchmark", "suite", "task_id", "trial")], separators=(",", ":"))


def evaluate(specs, worker_config, client, output, environments, batch_size, batch_wait_ms=2, timeout=180):
    from elaf.evaluation.libero_env import run_env_worker
    if environments < 1 or batch_size < 1 or timeout <= 0:
        raise ValueError("Positive environment/batch/timeout required")
    if len({episode_key(s) for s in specs}) != len(specs) or not specs:
        raise ValueError("Manifest must contain nonempty unique episode specifications")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"specs": specs, "worker_config": worker_config, "policy": client.metadata}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    ctx = mp.get_context("spawn")
    pending = deque(specs)
    workers, active, ready, dispatched = {}, {}, deque(), {}
    records = []
    started = time.perf_counter()
    first_observation = None
    inference_seconds, inference_calls, inferred_observations, control_steps = 0.0, 0, 0, 0
    max_batch = 0
    simulator_seconds = 0.0
    last_progress = started
    try:
        for _ in range(min(environments, len(specs))):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=run_env_worker, args=(child, worker_config))
            process.start()
            child.close()
            workers[parent] = process
            spec = pending.popleft()
            active[parent] = {"spec": spec, "started": time.perf_counter(), "steps": 0}
            parent.send({"command": "reset", "spec": spec})
            dispatched[parent] = time.perf_counter()
        while active:
            busy = [pipe for pipe in active if not any(pipe is p for p, _ in ready)]
            wait_time = max(0, batch_wait_ms / 1000) if ready else 1.0
            completed = wait(busy, timeout=wait_time) if busy else []
            # After the first ready observation, collect others for a bounded
            # window. A mere wait() returns immediately on the first worker.
            deadline = time.perf_counter() + batch_wait_ms / 1000
            while completed and len(completed) + len(ready) < batch_size:
                remaining = [p for p in busy if p not in completed]
                budget = deadline - time.perf_counter()
                if not remaining or budget <= 0:
                    break
                extra = wait(remaining, timeout=budget)
                if not extra:
                    break
                completed.extend(extra)
            for pipe in completed:
                result = pipe.recv()
                dispatched.pop(pipe)
                last_progress = time.perf_counter()
                if not result.get("ok"):
                    raise RuntimeError(result.get("traceback", result.get("error")))
                state = active[pipe]
                simulator_seconds += result.get("info", {}).get("env_seconds", 0.0)
                if "settle_steps" in result.get("info", {}):
                    state["environment_protocol"] = {k: result["info"][k] for k in ("horizon", "settle_steps", "image_rotation_degrees", "gripper_convention")}
                    state["environment_protocol"]["execute_steps"] = worker_config["execute_steps"]
                count = result.get("steps", 0)
                control_steps += count
                state["steps"] += count
                if result.get("done", False):
                    row = {"spec": state["spec"], "success": bool(result["success"]),
                           "steps": state["steps"], "seconds": time.perf_counter() - state["started"],
                           "environment_protocol": state.get("environment_protocol", {})}
                    records.append(row)
                    with (output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    if pending:
                        spec = pending.popleft()
                        active[pipe] = {"spec": spec, "started": time.perf_counter(), "steps": 0}
                        pipe.send({"command": "reset", "spec": spec})
                        dispatched[pipe] = time.perf_counter()
                    else:
                        del active[pipe]
                    continue
                if first_observation is None:
                    first_observation = time.perf_counter()
                ready.append((pipe, result["observation"]))
            if ready:
                group = [ready.popleft() for _ in range(min(batch_size, len(ready)))]
                result = client.request({"command": "predict", "observations": [o for _, o in group],
                    "episode_keys": [episode_key(active[p]["spec"]) for p, _ in group]})
                actions = result["actions"]
                if len(actions) != len(group):
                    raise ValueError("Policy batch dimension mismatch")
                inference_seconds += result["inference_seconds"]
                inference_calls += 1
                inferred_observations += len(group)
                max_batch = max(max_batch, len(group))
                for (pipe, _), action in zip(group, actions):
                    pipe.send({"command": "step", "action_chunk": action})
                    dispatched[pipe] = time.perf_counter()
                last_progress = time.perf_counter()
            for pipe, process in workers.items():
                if pipe in active and not process.is_alive():
                    raise RuntimeError(f"Simulator worker exited: {process.exitcode}")
            if any(time.perf_counter() - sent > timeout for sent in dispatched.values()):
                raise TimeoutError("An individual simulator request timed out")
            if time.perf_counter() - last_progress > timeout:
                raise TimeoutError("Simulator made no progress before timeout")
        elapsed = time.perf_counter() - started
        if len(records) != len(specs):
            raise RuntimeError("Incomplete episode coverage")
        summary = {"complete": True, "episodes": len(records),
            "successes": sum(r["success"] for r in records),
            "success_rate": sum(r["success"] for r in records) / len(records),
            "wall_seconds": elapsed, "episodes_per_hour": len(records) * 3600 / elapsed,
            "control_steps": control_steps, "control_steps_per_second": control_steps / elapsed,
            "environment_startup_seconds": (first_observation - started) if first_observation else elapsed,
            "inference_seconds": inference_seconds, "inference_calls": inference_calls,
            "summed_simulator_seconds": simulator_seconds,
            "mean_inference_batch": inferred_observations / inference_calls if inference_calls else 0,
            "max_inference_batch": max_batch, "environments": environments, "batch_size": batch_size,
            "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    except Exception as error:
        (output / "error.json").write_text(json.dumps({"error": str(error), "completed_episodes": len(records)}))
        raise
    finally:
        for pipe, process in workers.items():
            if process.is_alive():
                try:
                    pipe.send({"command": "close"})
                except (BrokenPipeError, OSError):
                    pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join()
            pipe.close()


def main():
    def stop(signum, frame):
        raise KeyboardInterrupt("Evaluation terminated")
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--environments", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-ms", type=float, default=2)
    args = parser.parse_args()
    client = PolicyClient(("127.0.0.1", args.port), args.auth_file.read_bytes())
    try:
        summary = evaluate(json.loads(args.manifest.read_text()), json.loads(args.worker_config.read_text()),
                           client, args.output, args.environments, args.batch_size, args.batch_wait_ms)
        final_metadata = client.request({"command": "hello"})
        (args.output / "policy_final.json").write_text(json.dumps(final_metadata, indent=2))
        print(json.dumps(summary, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
