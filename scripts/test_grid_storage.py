"""Temporary filesystem fixtures; never allocates production RAM caches."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import grid_cache_storage as storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.run = self.root / "experiments/action_only/ctp_grid20k_search/encoder_010000"
        self.cache = self.run / "cache_retry1/cache_final_010000_exclude_goal82"
        reference = self.root / "experiments/action_only/ctp_full_050000_handoff/cache_retry1/cache_final_050000_exclude_goal82"
        for suite, frames in {"spatial": 52970, "object": 66984, "goal": 51913, "10": 101469}.items():
            directory = reference / f"libero_{suite}_no_noops_1.0.0_lerobot"
            directory.mkdir(parents=True)
            (directory / "latents_meta.json").write_text(json.dumps({"episodes": {"unit-only": frames}}))
        self.meminfo = self.root / "unit_meminfo"
        self.meminfo.write_text("MemAvailable: 1000000000 kB\n")
        def local_path(value):
            value = str(value)
            if value.startswith("/dev/shm/eb_vjepa_grid_"):
                return self.root / "unit_ram" / Path(value).name
            if value == "/proc/meminfo":
                return self.meminfo
            return Path(value)
        self.path_patch = patch.object(storage, "Path", side_effect=local_path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.disk = patch.object(storage.shutil, "disk_usage", return_value=Mock(free=10**12))
        self.disk_mock = self.disk.start()
        self.addCleanup(self.disk.stop)

    def test_hybrid_payload_discovery_and_retirement_preserves_models_metadata(self):
        ram_bytes = storage.prepare_hybrid_storage(self.root, self.cache, 10000)
        self.assertGreater(ram_bytes, 130_000_000_000)
        self.assertLess(ram_bytes, 145_000_000_000)
        for suite in ("spatial", "object", "goal", "10"):
            directory = self.cache / f"libero_{suite}_no_noops_1.0.0_lerobot"
            directory.mkdir(exist_ok=True)
            self.assertEqual(directory.is_symlink(), suite != "10")
            (directory / "episode_000000.pt").write_bytes(b"unit-only")
            (directory / "encoder_identity.json").write_text('{"unit_fixture":true}')
        self.assertEqual(len(list(self.cache.glob("*/episode_*.pt"))), 4)
        (self.run / "continuation_status.json").write_text('{"phase":"complete"}')
        model = self.run / "preserve_model.pt"
        model.write_bytes(b"preserve-unit-model")
        storage.retire_grid_payloads(self.run, 10000)
        self.assertEqual(len(list(self.cache.glob("*/episode_*.pt"))), 0)
        self.assertEqual(len(list((self.run / "preserved_ram_cache_metadata").glob("*/encoder_identity.json"))), 3)
        self.assertTrue(model.exists())
        before = (self.run / "cache_retirement.json").read_text()
        storage.retire_grid_payloads(self.run, 10000)
        self.assertEqual(before, (self.run / "cache_retirement.json").read_text())

    def test_low_memory_refuses_before_creating_cache(self):
        self.disk_mock.return_value.free = 1
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            storage.prepare_hybrid_storage(self.root, self.cache, 10000)
        self.assertFalse(self.cache.exists())

    def test_symlink_to_unrelated_payload_cannot_be_retired(self):
        storage.prepare_hybrid_storage(self.root, self.cache, 10000)
        outside = self.root / "preserve_outside.pt"
        outside.write_bytes(b"preserve")
        suite = self.cache / "libero_spatial_no_noops_1.0.0_lerobot"
        (suite / "episode_000000.pt").symlink_to(outside)
        (self.run / "continuation_status.json").write_text('{"phase":"complete"}')
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            storage.retire_grid_payloads(self.run, 10000)
        self.assertEqual(outside.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()
