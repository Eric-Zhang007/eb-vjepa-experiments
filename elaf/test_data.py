"""Contract tests; synthetic tensors here test collation, never teacher training data."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
import torch
from elaf.data import ELAFDataset, collate_fn
from elaf.prepare_data import episodes


class DataTests(unittest.TestCase):
    def test_collate_preserves_native_patch_order_and_padding(self):
        def sample(n, value):
            return {"input_ids": torch.arange(n), "attention_mask": torch.ones(n,dtype=torch.long),
                "pixel_values":torch.full((4,1176),value), "image_grid_thw":torch.tensor([[1,2,2]]),
                "proprio":torch.zeros(8),"actions":torch.zeros(8,7),"action_valid":torch.ones(8,dtype=torch.bool),
                "future_latents":torch.zeros(16,1024),"provenance":{"frame_index":value}}
        result = collate_fn([sample(3,1),sample(5,2)],pad_token_id=99)
        self.assertEqual(result["input_ids"][0].tolist(),[0,1,2,99,99])
        self.assertEqual(result["attention_mask"][0].tolist(),[1,1,1,0,0])
        self.assertEqual(result["pixel_values"][:,0].tolist(),[1]*4+[2]*4)
        self.assertEqual(result["future_latents"].shape,(2,16,1024))

    def test_goal82_and_nontraining_episodes_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for suite in ("spatial","object","goal","10"):
                meta=root/f"libero_{suite}_no_noops_1.0.0_lerobot"/"meta"
                meta.mkdir(parents=True)
                (meta/"info.json").write_text(json.dumps({"splits":{"train":"0:83"},"chunks_size":1000,
                    "fps":20,"data_path":"{episode_index}.parquet","video_path":"{video_key}/{episode_index}.mp4"}))
                (meta/"episodes.jsonl").write_text("\n".join(json.dumps({"episode_index":i,"tasks":["task"],"length":10}) for i in (81,82,83)))
            selected=episodes(root)
            self.assertEqual(len(selected),7)
            self.assertFalse(any(ep["suite"]=="goal" and ep["episode_index"]==82 for ep in selected))
            self.assertFalse(any(ep["episode_index"]==83 for ep in selected))

    def test_empty_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            (Path(tmp)/"stats.json").write_text('{}')
            digest=hashlib.sha256(b'{}').hexdigest()
            path.write_text(json.dumps({"samples":[],"stats_path":"stats.json","source_selection":"stats.json",
                                      "stats_sha256":digest,"source_selection_sha256":digest}))
            with self.assertRaises(ValueError):
                ELAFDataset(path)

    def test_changed_normalization_statistics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            (Path(tmp)/"stats.json").write_text('{"changed":true}')
            digest=hashlib.sha256(b'{}').hexdigest()
            path.write_text(json.dumps({"samples":[{"path":"unused.pt"}],"stats_path":"stats.json",
                "source_selection":"stats.json","stats_sha256":digest,"source_selection_sha256":digest}))
            with self.assertRaisesRegex(ValueError,"identity changed"):
                ELAFDataset(path)


if __name__ == "__main__":
    unittest.main()
