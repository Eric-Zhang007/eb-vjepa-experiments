"""Real tiny-policy RPC batching test; set ELAF_RPC_TEST_CONFIG explicitly.

Run with CUDA_VISIBLE_DEVICES=1 to reserve physical GPU 1. This verifies
transport and inference plumbing, not trained policy performance.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing.connection as connection
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

from elaf.evaluation.rpc import PolicyClient


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class RPCTests(unittest.TestCase):
    def test_response_timeout(self):
        key = os.urandom(32)
        listener = connection.Listener(("127.0.0.1", 0), authkey=key)
        stop = threading.Event()
        def silent_server():
            client = listener.accept()
            try:
                client.recv()
                stop.wait(2)
            finally:
                client.close()
        thread = threading.Thread(target=silent_server, daemon=True)
        thread.start()
        client = PolicyClient.__new__(PolicyClient)
        client.connection = connection.Client(listener.address, authkey=key)
        client.timeout = 0.02
        try:
            with self.assertRaises(TimeoutError):
                client.request({"command": "hello"})
        finally:
            client.close()
            stop.set()
            thread.join(timeout=3)
            listener.close()
        self.assertFalse(thread.is_alive())

    @unittest.skipUnless(os.getenv("ELAF_RPC_TEST_CONFIG"), "requires explicit tiny checkpoint config")
    def test_real_concurrent_batch(self):
        cfg = json.loads(Path(os.environ["ELAF_RPC_TEST_CONFIG"]).read_text())
        cfg.update(max_batch_size=4, batch_wait_ms=50, device="cuda:0")
        with tempfile.TemporaryDirectory(prefix="elaf-rpc-test-") as directory:
            directory = Path(directory)
            config, auth, ready = [directory / name for name in ("config.json", "auth", "ready.json")]
            config.write_text(json.dumps(cfg))
            auth.write_bytes(os.urandom(32))
            auth.chmod(0o600)
            port = unused_port()
            with (directory / "service.log").open("w+") as log:
                process = subprocess.Popen([sys.executable, "-m", "elaf.evaluation.rpc", "--config", str(config),
                    "--port", str(port), "--auth-file", str(auth), "--ready-file", str(ready)], stdout=log, stderr=subprocess.STDOUT)
                clients = []
                try:
                    deadline = time.monotonic() + 90
                    while not ready.exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            log.flush(); log.seek(0)
                            self.fail("service startup failed: " + log.read())
                        time.sleep(0.05)
                    clients = [PolicyClient(("127.0.0.1", port), auth.read_bytes(), timeout=30) for _ in range(2)]
                    rng = np.random.default_rng(3)
                    observations = [dict(image=rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
                        proprio=np.zeros(8, np.float32), instruction="put the mug on the plate", noise_seed=100 + i) for i in range(2)]
                    # Warm native kernels before checking batching and determinism.
                    clients[0].predict([observations[0]])
                    barrier = threading.Barrier(2)
                    def call(index):
                        barrier.wait(timeout=5)
                        return clients[index].request({"command": "predict", "observations": [observations[index]]})
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        replies = list(pool.map(call, range(2)))
                    for reply in replies:
                        self.assertEqual(reply["service_batch_size"], 2)
                        self.assertEqual(reply["actions"].shape, (1, 8, 7))
                    serial = [clients[i].predict([observations[i]]) for i in range(2)]
                    for i in range(2):
                        np.testing.assert_allclose(replies[i]["actions"], serial[i], rtol=0.03, atol=0.02)
                    with self.assertRaisesRegex(RuntimeError, "Unknown RPC command"):
                        clients[0].request({"command": "invalid-command"})
                    print("RPC_BATCH_VERIFIED service_batch_size=2 replies=[1,8,7]", flush=True)
                finally:
                    for client in clients:
                        client.close()
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
