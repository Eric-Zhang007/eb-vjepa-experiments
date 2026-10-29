"""Authenticated loopback-only policy transport; one model per service process."""
import argparse
import json
import multiprocessing.connection as connection
from multiprocessing import AuthenticationError
import os
from pathlib import Path
import threading
import queue
import time
import traceback


class PolicyClient:
    def __init__(self, address, authkey, timeout=180):
        if address[0] not in ("127.0.0.1", "localhost"):
            raise ValueError("Only trusted loopback clients supported")
        self.connection = connection.Client(address, authkey=authkey)
        self.timeout = timeout
        self.metadata = self.request({"command": "hello"})["metadata"]

    def request(self, request):
        self.connection.send(request)
        if not self.connection.poll(self.timeout):
            raise TimeoutError("Policy service response timed out")
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("traceback", response.get("error")))
        return response

    def predict(self, observations, episode_keys=None):
        return self.request({"command": "predict", "observations": observations,
                             "episode_keys": episode_keys})["actions"]

    def close(self):
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--auth-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from elaf.evaluation.policy import build_policy
    cfg = json.loads(args.config.read_text())
    if int(cfg.get("max_batch_size", 32)) < 1 or float(cfg.get("batch_wait_ms", 2)) < 0:
        raise ValueError("Invalid service batching configuration")
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    policy = build_policy(cfg)
    metadata = policy.fingerprint()
    requests = queue.Queue(maxsize=256)
    listener = connection.Listener(("127.0.0.1", args.port), authkey=args.auth_file.read_bytes())
    args.ready_file.write_text(json.dumps({"pid": os.getpid(), "metadata": metadata}, indent=2))

    def infer():
        pending = None
        limit = int(cfg.get("max_batch_size", 32))
        window = float(cfg.get("batch_wait_ms", 2)) / 1000
        while True:
            first = pending if pending is not None else requests.get()
            pending = None
            jobs = [first]
            total = len(first[0]["observations"])
            deadline = time.perf_counter() + window
            while total < limit:
                try:
                    item = requests.get(timeout=max(0, deadline - time.perf_counter()))
                except queue.Empty:
                    break
                size = len(item[0]["observations"])
                if total + size > limit:
                    pending = item
                    break
                jobs.append(item)
                total += size
            try:
                observations = [obs for request, _, _ in jobs for obs in request["observations"]]
                start = time.perf_counter()
                with torch.inference_mode():
                    actions = policy.predict(observations)
                elapsed = time.perf_counter() - start
                offset = 0
                for request, event, result in jobs:
                    size = len(request["observations"])
                    result.update(ok=True, actions=actions[offset:offset+size],
                                  inference_seconds=elapsed * size / total, service_batch_size=total)
                    offset += size
                    event.set()
            except Exception as error:
                failure = {"ok": False, "error": str(error), "traceback": traceback.format_exc()}
                for _, event, result in jobs:
                    result.update(failure)
                    event.set()

    threading.Thread(target=infer, daemon=True).start()

    def serve(client):
        try:
            while True:
                try:
                    request = client.recv()
                except EOFError:
                    client.close()
                    return
                if request.get("command") == "hello":
                    current = policy.fingerprint()
                    client.send({"ok": True, "metadata": current,
                                 "peak_model_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
                    continue
                if request.get("command", "predict") != "predict":
                    raise ValueError("Unknown RPC command")
                size = len(request["observations"])
                if not 1 <= size <= int(cfg.get("max_batch_size", 32)):
                    raise ValueError("Request batch exceeds configured service capacity")
                event, result = threading.Event(), {}
                requests.put((request, event, result), timeout=180)
                if not event.wait(180):
                    raise TimeoutError("Model inference deadline exceeded")
                client.send(result)
        except Exception as error:
            try:
                client.send({"ok": False, "error": str(error), "traceback": traceback.format_exc()})
            finally:
                client.close()

    while True:
        try:
            client = listener.accept()
        except AuthenticationError:
            continue
        threading.Thread(target=serve, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
