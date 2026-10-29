"""CPU unit fixtures only; no training data or model weights are substituted."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch

from Roboactualizer.utils.encoder_identity import checkpoint_sha256, require_encoder_identity
from Roboactualizer.models.roboactualizer.policy import _load_latent_norm_stats


def rejected(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected identity rejection")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        encoder = root / "unit_fixture.bin"
        encoder.write_bytes(b"unit-test-only")
        sha = checkpoint_sha256(encoder)
        require_encoder_identity({"encoder_checkpoint_sha256": sha}, sha, "unit")
        rejected(lambda: require_encoder_identity({}, sha, "missing"))
        rejected(lambda: require_encoder_identity({"encoder_checkpoint_sha256": "wrong"}, sha, "mixed"))
        suite = root / "cache" / "suite"
        suite.mkdir(parents=True)
        meta = {"encoder_checkpoint_sha256": sha, "embed_dim": 2,
                "clip_len": 4, "frame_interval": 2, "latent_dtype": "float32",
                "backbone_model_name": "unit_fixture", "episodes": {"0": 2}}
        (suite / "latents_meta.json").write_text(json.dumps(meta))
        payload = {"encoder_checkpoint_sha256": sha,
                   "latents": torch.tensor([[[1., 2.]], [[3., 6.]]])}
        torch.save(payload, suite / "episode_000000.pt")
        # An unlisted stale file must not contribute to the statistics.
        torch.save({"latents": torch.zeros(2, 1, 2)}, suite / "episode_999999.pt")
        output = root / "stats.json"
        command = [sys.executable, str(Path(__file__).with_name("compute_latent_norm_stats.py")),
                   str(root / "cache"), "--out", str(output)]
        subprocess.run(command, check=True)
        mean, std, identity = _load_latent_norm_stats(str(output), 2, "unit_fixture", str(encoder))
        torch.testing.assert_close(mean, torch.tensor([2., 4.]))
        torch.testing.assert_close(std, torch.tensor([1., 2.]))
        assert identity["encoder_checkpoint_sha256"] == sha
        encoder.write_bytes(b"different-unit-fixture")
        rejected(lambda: _load_latent_norm_stats(str(output), 2, "unit_fixture", str(encoder)))
        output.unlink()
        payload["encoder_checkpoint_sha256"] = "wrong"
        torch.save(payload, suite / "episode_000000.pt")
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode != 0 and "Encoder identity mismatch" in failed.stderr
    print(json.dumps({"encoder_identity_unit": "PASS", "real_model_test": False}))


if __name__ == "__main__":
    main()
