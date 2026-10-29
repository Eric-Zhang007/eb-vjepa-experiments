"""Reuse measured gates only when the complete prior recipe contract matches."""
import hashlib
import json
from pathlib import Path
import shutil

import yaml


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(root):
    prior = root / "experiments/ctp_full_72h"
    out = root / "experiments/ctp_grid20k_20261008"
    config = yaml.safe_load((prior / "calibrated.yaml").read_text())
    evidence = json.loads((prior / "smoke/pass.json").read_text())
    if not (evidence["passed"] and evidence["sanity_passed_all_ranks"]
            and evidence["step"] == 100 and evidence["resumed_from"] == 90):
        raise RuntimeError("Prior 90+10 smoke did not pass")
    contract = evidence["contract"]
    for key, expected in contract.items():
        if key not in ("implementation_sha256", "manifest_sha256") and config[key] != expected:
            raise RuntimeError(f"Recipe mismatch: {key}")
    for name, expected in contract["implementation_sha256"].items():
        if digest(root / "ctp_full" / name) != expected:
            raise RuntimeError(f"Implementation changed since smoke: {name}")
    if digest(root / config["manifest"]) != contract["manifest_sha256"]:
        raise RuntimeError("Manifest differs from validated manifest")
    selection = json.loads((prior / "calibration/selection.json").read_text())
    for key in ("microbatch", "accumulation", "activation_checkpointing"):
        if config[key] != selection[key]:
            raise RuntimeError("Prior calibration mismatch")
    if config["lr_clock"] != "wall" or config["wall_seconds"] != 259200:
        raise RuntimeError("Expected original 72h wall-clock cosine")
    if out.exists():
        raise RuntimeError("Grid output already exists; refusing overwrite")
    if shutil.disk_usage(root).free < 150_000_000_000:
        raise RuntimeError("Insufficient disk for twenty full snapshots and EMA exports")
    out.mkdir()
    config.update(name="droid_vitl16_256_16f_ctp_grid20k_20261008",
                  output=str(out.relative_to(root)), snapshot_every=1000)
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    for relative in ("smoke/pass.json", "calibration/selection.json"):
        destination = out / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(prior / relative, destination)
    (out / "validation_reuse.json").write_text(json.dumps({
        "source": str(prior), "mode": "Reuse actual prior measurements; no new smoke claimed",
        "recipe_and_implementation_identical": True,
        "prior_smoke_completed_at": evidence["completed_at"],
        "sha256": {name: digest(prior / name) for name in
                   ("smoke/pass.json", "calibration/selection.json", "calibrated.yaml")},
        "changes": {"name": config["name"], "output": config["output"],
                    "snapshot_every": 1000, "cli_stop_step": 20000},
        "unchanged": "All objective, data, model, optimizer and 72h cosine settings",
    }, indent=2))
    print(out)


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
