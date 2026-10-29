"""Exercise every upstream noise family on a real preflight image."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from eval_10k_robust import LegacyBinaryNumpy, ROOT
from libero.libero.envs import env_wrapper

env_wrapper.np = LegacyBinaryNumpy()
out = ROOT / "experiments/action_only/libero_10k_plus"
image = Image.open(out / "preflight_Camera_Viewpoints_agentview_image.png")
rows = []
for name in ("motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur"):
    np.random.seed(239)
    start = time.monotonic()
    result = np.asarray(getattr(env_wrapper, name)(image.copy(), severity=1))
    assert result.shape == (256, 256, 3) and np.isfinite(result).all() and result.std() > 0
    row = {"function": name, "seconds": time.monotonic()-start, "status": "PASS"}
    rows.append(row)
    print(row, flush=True)
(out / "noise_compat.json").write_text(json.dumps(rows, indent=2))
