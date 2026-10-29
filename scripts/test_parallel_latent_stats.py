"""Unit-only synthetic cache verifies serial and parallel full-pass statistics."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    script = root / "repos/RoboActualizer/scripts/compute_latent_norm_stats.py"
    with tempfile.TemporaryDirectory() as temporary:
        cache = Path(temporary)
        suite = cache / "unit_only"
        suite.mkdir()
        tensors = [torch.randn(9000, 2, 8).half() for _ in range(4)]
        meta = {"encoder_checkpoint_sha256": "unit-only", "episode_selection_sha256": "unit-only",
                "embed_dim": 8, "clip_len": 4, "frame_interval": 2, "latent_dtype": "float16",
                "backbone_model_name": "unit-only", "episodes": {str(i): 9000 for i in range(4)}}
        (suite / "latents_meta.json").write_text(json.dumps(meta))
        for i, tensor in enumerate(tensors):
            torch.save({"latents": tensor, "encoder_checkpoint_sha256": "unit-only"},
                       suite / f"episode_{i:06d}.pt")
        results = []
        for workers in [1, 4]:
            output = cache / f"stats_{workers}.json"
            subprocess.run([sys.executable, str(script), str(cache), "--out", str(output),
                            "--workers", str(workers)], check=True)
            results.append(json.loads(output.read_text()))
        reference = torch.cat(tensors).reshape(-1, 8).double()
        for result in results:
            torch.testing.assert_close(torch.tensor(result["mean"], dtype=torch.float64), reference.mean(0), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(torch.tensor(result["std"], dtype=torch.float64), reference.std(0, correction=0), rtol=1e-12, atol=1e-12)
        assert results[0]["mean"] == results[1]["mean"]
        assert results[0]["std"] == results[1]["std"]
        tensors[0][0, 0, 0] = float("nan")
        torch.save({"latents": tensors[0], "encoder_checkpoint_sha256": "unit-only"}, suite / "episode_000000.pt")
        result = subprocess.run([sys.executable, str(script), str(cache), "--out", str(cache / "bad.json"), "--workers", "4"], capture_output=True)
        assert result.returncode and not (cache / "bad.json").exists()
        print("PASS serial/parallel/reference agreement and nonfinite refusal")


if __name__ == "__main__":
    main()
