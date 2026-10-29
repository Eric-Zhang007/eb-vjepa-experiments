"""CPU-only launcher contract tests; no training or GPU work is launched."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


class LaunchContractTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "launcher", Path(__file__).with_name("launch_action_final.py"))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module.__file__ = str(self.root / "scripts/launch_action_final.py")
        self.cache = self.root / "new_cache"
        self.cache.mkdir()
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        (self.root / "experiments").mkdir()
        manifest = self.root / "experiments/libero_episode_selection.json"
        manifest.write_text("{}")
        selection = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.encoder = self.root / "encoder.pt"
        self.encoder.write_bytes(b"unit-test-identity-only")
        digest = hashlib.sha256(self.encoder.read_bytes()).hexdigest()
        (self.prepared / "full_cache_status.json").write_text(json.dumps({
            "phase": "complete", "cache": str(self.cache), "encoder_sha256": digest}))
        stats = {"episode_selection_sha256": selection, "num_files": 1692,
                 "encoder_checkpoint_sha256": digest}
        (self.cache / "latent_norm_stats.json").write_text(json.dumps(stats))
        (self.cache / "suite").mkdir()
        (self.cache / "suite/latents_meta.json").write_text(json.dumps({
            **stats, "episodes": {"0": 273336}}))
        self.output = self.root / "handoff/train"
        self.argv = ["launch_action_final.py", "--encoder", str(self.encoder),
                     "--step", "103233", "--cache-output", str(self.prepared),
                     "--train-output", str(self.output)]

    def run_launcher(self, returncode=0):
        child = Mock(pid=123)
        child.wait.return_value = returncode
        with patch("sys.argv", self.argv), patch.object(
            self.module.subprocess, "check_output",
            return_value="\n".join(f"{gpu}, 0" for gpu in range(8))), patch.object(
            self.module.subprocess, "Popen", return_value=child) as popen:
            self.module.main()
            return popen.call_args.args[0]

    def test_new_encoder_cache_stats_and_recipe(self):
        command = self.run_launcher()
        for expected in [f"model.backbone.ckpt_path={self.encoder}",
                         f"data.train.rgb_latent_cache_dir={self.cache}",
                         f"model.latent_norm_stats_path={self.cache}/latent_norm_stats.json",
                         "encoder_tag=ctp_final_103233", "num_epochs=20",
                         "batch_size=32", "--nproc_per_node=8"]:
            self.assertIn(expected, command)

    def test_encoder_mismatch_fails_before_launch(self):
        self.encoder.write_bytes(b"wrong")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.run_launcher()

    def test_official_original_run_identity(self):
        self.argv[self.argv.index("--step") + 1] = "0"
        command = self.run_launcher()
        self.assertIn("encoder_tag=official_original", command)
        self.assertIn("wandb.name=libero_action_only_official_original", command)
        self.assertIn(f"model.backbone.ckpt_path={self.encoder}", command)

    def test_child_failure_propagates(self):
        with self.assertRaisesRegex(RuntimeError, "status 7"):
            self.run_launcher(returncode=7)
        state = json.loads((self.output.parent / "formal_train_status.json").read_text())
        self.assertEqual(state["phase"], "failed")

    def test_duplicate_refused(self):
        self.run_launcher()
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.run_launcher()

    def test_four_gpu_batch64_keeps_global256(self):
        self.argv += ["--gpus", "0,1,2,3", "--batch-size", "64"]
        child = Mock(pid=123)
        child.wait.return_value = 0
        with patch("sys.argv", self.argv), patch.object(
            self.module.subprocess, "check_output",
            return_value="\n".join(f"{gpu}, {0 if gpu < 4 else 45000}" for gpu in range(8))), patch.object(
            self.module.subprocess, "Popen", return_value=child) as popen:
            self.module.main()
        command = popen.call_args.args[0]
        self.assertIn("--nproc_per_node=4", command)
        self.assertIn("batch_size=64", command)
        self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        state = json.loads((self.output.parent / "formal_train_status.json").read_text())
        self.assertEqual(state["world_size"], 4)
        self.assertEqual(state["global_batch"], 256)


if __name__ == "__main__":
    unittest.main()
