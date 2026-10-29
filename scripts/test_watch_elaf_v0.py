"""CPU assertions for idle scheduling and provenance gates."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
import types

import watch_elaf_v0 as watcher


class WatcherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.plan = {"gpus": list(range(6)), "idle_seconds": 30, "poll_seconds": 5,
                     "cwd": str(root), "launch_argv": ["python", "train.py"],
                     "validation_argv": ["python", "validate.py"]}
        for key in ["config_path", "cache_manifest", "cache_receipt", "validation_receipt",
                    "status_path", "claim_path", "lock_path", "training_log", "validation_log"]:
            self.plan[key] = str(root / key)
        Path(self.plan["config_path"]).write_text("{}")
        Path(self.plan["cache_manifest"]).write_text("manifest")
        source = root / "train.py"
        source.write_text("pass")
        self.plan["config_sha256"] = watcher.sha256(self.plan["config_path"])
        self.plan["source_sha256"] = {str(source): watcher.sha256(source)}
        watcher.atomic_json(self.plan["cache_receipt"], {"status": "complete",
                            "manifest_sha256": watcher.sha256(self.plan["cache_manifest"])})
        watcher.atomic_json(self.plan["validation_receipt"], {
            "status": "passed", "config_sha256": self.plan["config_sha256"],
            "source_sha256": self.plan["source_sha256"],
            "cache_manifest_sha256": watcher.sha256(self.plan["cache_manifest"])})

    def snapshot(self):
        return {i: {"uuid": f"GPU-{i}", "memory_mib": 0, "utilization": 0, "pids": []}
                for i in range(6)}

    def test_explicitly_disabled_training_validation_needs_no_receipt(self):
        Path(self.plan["validation_receipt"]).unlink()
        self.plan.pop("validation_argv")
        self.plan["require_training_validation"] = False
        self.assertTrue(watcher.prerequisites(self.plan))
        Path(self.plan["cache_receipt"]).unlink()
        self.assertFalse(watcher.prerequisites(self.plan))

    def test_busy_resets_and_requires_full_thirty_seconds(self):
        timer = watcher.IdleTimer()
        self.assertFalse(timer.observe(self.snapshot(), 0))
        self.assertFalse(timer.observe(self.snapshot(), 29))
        busy = self.snapshot()
        busy[5]["pids"] = [123]
        self.assertFalse(timer.observe(busy, 30))
        self.assertFalse(timer.observe(self.snapshot(), 31))
        self.assertFalse(timer.observe(self.snapshot(), 60))
        self.assertTrue(timer.observe(self.snapshot(), 61))
        timer.reset()  # Query failure or process restart cannot inherit elapsed time.
        self.assertFalse(timer.observe(self.snapshot(), 62))

    def test_memory_and_utilization_fail_closed(self):
        for field, value in [("memory_mib", 512), ("utilization", 6)]:
            sample = self.snapshot()
            sample[0][field] = value
            timer = watcher.IdleTimer()
            self.assertFalse(timer.observe(sample, 0))
            self.assertFalse(timer.observe(sample, 100))

    def test_parse_processes_and_malformed(self):
        gpu = "\n".join(f"{i}, GPU-{i}, 0, 0" for i in range(8))
        sample = watcher.parse_gpu_snapshot(gpu, "GPU-0, 12\nGPU-7, 99\n", list(range(6)))
        self.assertEqual(sample[0]["pids"], [12])
        self.assertNotIn(7, sample)
        for g, p in [("0, GPU-0, N/A, 0", ""), (gpu, "GPU-unknown, 12"),
                     (gpu, "No running processes found"), (gpu + "\n0, GPU-0, 0, 0", "")]:
            with self.assertRaises(ValueError):
                watcher.parse_gpu_snapshot(g, p, list(range(6)))

    def test_cache_and_validation_gate(self):
        self.assertTrue(watcher.prerequisites(self.plan))
        Path(self.plan["cache_receipt"]).unlink()
        self.assertFalse(watcher.prerequisites(self.plan))
        watcher.atomic_json(self.plan["cache_receipt"], {"status": "building"})
        self.assertFalse(watcher.prerequisites(self.plan))
        watcher.atomic_json(self.plan["cache_receipt"], {"status": "complete", "manifest_sha256": "wrong"})
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            watcher.prerequisites(self.plan)

    def test_mismatched_validation_source_and_config(self):
        receipt_path = Path(self.plan["validation_receipt"])
        receipt = json.loads(receipt_path.read_text())
        receipt["source_sha256"] = {}
        watcher.atomic_json(receipt_path, receipt)
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            watcher.prerequisites(self.plan)
        Path(self.plan["config_path"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            watcher.prerequisites(self.plan)

    def test_claim_is_exclusive(self):
        watcher.claim_once(self.plan["claim_path"], {"first": True})
        with self.assertRaises(FileExistsError):
            watcher.claim_once(self.plan["claim_path"], {"first": False})
        self.assertEqual(json.loads(Path(self.plan["claim_path"]).read_text()), {"first": True})

    def test_preflight_failure_never_spawns_training(self):
        Path(self.plan["validation_receipt"]).unlink()
        fake_lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        failed = types.SimpleNamespace(pid=123, poll=lambda: 1)
        with patch.dict(sys.modules, {"fcntl": fake_lock}), \
                patch.object(watcher, "query_gpus", return_value=self.snapshot()), \
                patch.object(watcher.time, "monotonic", side_effect=iter(range(0, 1000, 10))), \
                patch.object(watcher.time, "sleep"), \
                patch.object(watcher, "process_identity", return_value={"pid": 123}), \
                patch.object(watcher, "spawn", return_value=failed) as spawn:
            with self.assertRaisesRegex(RuntimeError, "Validation failed"):
                watcher.run(self.plan)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(spawn.call_args.args[1], "validation_argv")
        self.assertFalse(Path(self.plan["claim_path"]).exists())
        self.assertEqual(json.loads(Path(self.plan["status_path"]).read_text())["phase"], "blocked")

    def test_success_and_restart_launch_only_once(self):
        fake_lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        process = types.SimpleNamespace(pid=456, poll=lambda: 0)
        with patch.dict(sys.modules, {"fcntl": fake_lock}), \
                patch.object(watcher, "query_gpus", return_value=self.snapshot()), \
                patch.object(watcher.time, "monotonic", side_effect=iter(range(0, 1000, 10))), \
                patch.object(watcher.time, "sleep"), \
                patch.object(watcher, "process_identity", return_value={"pid": 456, "start_ticks": "123"}), \
                patch.object(watcher, "spawn", return_value=process) as spawn:
            watcher.run(self.plan)
            watcher.run(self.plan)
        self.assertEqual(spawn.call_count, 1)
        self.assertEqual(spawn.call_args.args[1], "launch_argv")
        self.assertEqual(json.loads(Path(self.plan["status_path"]).read_text())["phase"], "already_claimed")

    def test_failed_training_is_reported_without_relaunch(self):
        fake_lock = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        exits = iter([None, None, 7])
        process = types.SimpleNamespace(pid=456, poll=lambda: next(exits))
        with patch.dict(sys.modules, {"fcntl": fake_lock}), \
                patch.object(watcher, "query_gpus", return_value=self.snapshot()), \
                patch.object(watcher.time, "monotonic", side_effect=iter(range(0, 1000, 10))), \
                patch.object(watcher.time, "sleep"), \
                patch.object(watcher, "process_identity", return_value={"pid": 456}), \
                patch.object(watcher, "spawn", return_value=process) as spawn:
            watcher.run(self.plan)
        self.assertEqual(spawn.call_count, 1)
        status = json.loads(Path(self.plan["status_path"]).read_text())
        self.assertEqual(status["phase"], "blocked")
        self.assertEqual(status["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
