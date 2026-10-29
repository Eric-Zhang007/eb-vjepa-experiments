"""Native baseline tests; DREAM_BASELINE_REAL=1 enables 8.3B single-step smoke."""
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dream_baseline.model import DreamBaselineConfig, DreamBaselinePolicy


ROOT = Path(os.getenv("ELAF_ROOT", "/data/users/jiachangzhang/eb_vjepa"))


class BaselineTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv("DREAM_BASELINE_REAL") == "1", "requires real checkpoint and CUDA")
    def test_real_backward_predict_save(self):
        torch.set_num_threads(2)
        cfg = DreamBaselineConfig(str(ROOT / "checkpoints/dream_vla/Dream-VLA-7B"), str(ROOT / "repos/Dream-VLX"))
        model = DreamBaselinePolicy.from_pretrained(cfg)
        processor = model.processor
        image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
            {"type": "text", "text": "What action should the robot take to put the mug on the plate?"}]}]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image, image], return_tensors="pt")
        device = torch.device("cuda:0")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prefix = inputs["input_ids"].shape[1]
        tokens = torch.tensor([[151667] * 56 + [151645, 198]], device=device)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], tokens), dim=1)
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        inputs["action_mask"] = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
        inputs["action_mask"][:, prefix:prefix + 56] = True
        inputs["proprio"] = torch.zeros(1, 8, device=device)
        inputs["actions"] = torch.zeros(1, 8, 7, device=device)
        inputs["action_valid"] = torch.ones(1, 8, device=device, dtype=torch.bool)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(inputs)
        result["loss"].backward()
        unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(unused, [])
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        model.eval()
        noise = torch.randn(1, 8, 7, device=device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict(inputs, noise=noise)
        self.assertEqual(output.shape, (1, 8, 7))
        self.assertTrue(torch.isfinite(output).all())
        doubled = {key: torch.cat((value, value), dim=0) for key, value in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            batched = model.predict(doubled, noise=torch.cat((noise, noise), dim=0))
        self.assertEqual(batched.shape, (2, 8, 7))
        torch.testing.assert_close(batched[0], batched[1], rtol=0, atol=0)
        torch.testing.assert_close(batched[:1], output, rtol=0.05, atol=0.03)
        with tempfile.TemporaryDirectory(prefix="dream-baseline-test-", dir=ROOT / "cache/tmp") as directory:
            receipt = model.save_trainable(directory)
            parameter = trainable[0]
            with torch.no_grad():
                parameter.add_(0.25)
            model.load_trainable(directory)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                restored = model.predict(inputs, noise=noise)
            torch.testing.assert_close(output, restored, rtol=0, atol=0)
        print("DREAM_BASELINE_REAL_PASS", {"loss": result["loss"].item(), "trainable": sum(p.numel() for p in trainable),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "saved_tensors": receipt["trainable_tensors"]}, flush=True)


if __name__ == "__main__":
    unittest.main()
