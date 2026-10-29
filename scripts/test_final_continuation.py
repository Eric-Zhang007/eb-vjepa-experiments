"""Isolated CPU tests: mocked subprocesses, no real evaluation results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class ContinuationTest(unittest.TestCase):
    def test_complete_chain_uses_new_run_and_eight_workers_per_benchmark(self):
        self.check_chain(False)

    def test_four_gpu_chain_keeps_eight_shards_inside_owned_devices(self):
        self.check_chain(True)

    def check_chain(self, four_gpu):
        spec = importlib.util.spec_from_file_location(
            "continuation", Path(__file__).with_name("continue_final_action.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mod.__file__ = str(root / "scripts/continue_final_action.py")
            out = root / "experiments/action_only/ctp_full_handoff"
            cache = out / "cache_retry1"
            cache.mkdir(parents=True)
            (cache / "full_cache_status.json").write_text('{"phase":"complete"}')
            run = out / "train_action_only_103233"
            weights = run / "checkpoints/weights"
            weights.mkdir(parents=True)
            ckpt = weights / "step_020000.pt"
            ckpt.write_bytes(b"unit-test-only-not-model-weights")
            sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            source = root / "experiments/action_only/libero_20k_plus_280_shared_run"
            (source / "text_cache").mkdir(parents=True)
            (source / "manifest.json").write_text('{"episodes":280}')
            (source / "preflight.json").write_text('{"status":"PASS"}')
            commands = []
            def worker(command, **kwargs):
                commands.append(command)
                target = Path(command[command.index("--output") + 1])
                shard = command[command.index("--shard") + 1]
                self.assertEqual(kwargs["env"]["EVAL_GPU"], str(int(shard) % (4 if four_gpu else 8)))
                self.assertEqual(command[command.index("--run") + 1], str(run))
                plus = "--benchmark" in command
                local = target / f"shard_{shard}"
                local.mkdir()
                key, n = ("episodes", 35) if plus else ("tasks", 25)
                (local / "report.json").write_text(json.dumps({
                    "phase": "complete", "checkpoint_sha256": sha,
                    key: [{"success": False} for _ in range(n)]}))
                return Mock(pid=1, returncode=0, poll=Mock(return_value=0))
            argv = ["continue_final_action.py"] + (["--gpus", "0,1,2,3", "--batch-size", "64"] if four_gpu else [])
            with patch("sys.argv", argv), patch.object(
                mod.subprocess, "run") as train, patch.object(mod.subprocess, "Popen", side_effect=worker):
                mod.main()
                train.assert_called_once()
                command = train.call_args.args[0]
                self.assertEqual(command[command.index("--batch-size") + 1], "64" if four_gpu else "32")
                self.assertEqual(command[command.index("--gpus") + 1], "0,1,2,3" if four_gpu else "0,1,2,3,4,5,6,7")
            self.assertEqual(len(commands), 16)
            self.assertEqual(json.loads((out / "continuation_status.json").read_text())["phase"], "complete")
            with patch("sys.argv", ["continue_final_action.py"]):
                with self.assertRaisesRegex(RuntimeError, "duplicate"):
                    mod.main()


if __name__ == "__main__":
    unittest.main()
