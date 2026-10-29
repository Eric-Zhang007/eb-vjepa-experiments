"""Safely extract the verified Plus archive into an isolated assets directory."""
import fcntl
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

root = Path(__file__).resolve().parents[1]
out = root / "experiments/action_only"
lock = (out / "plus_extract.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
source = root / "checkpoints/benchmark_assets/libero_plus_assets.zip"
target = root / "checkpoints/benchmark_assets/libero_plus"
state = json.loads((out / "plus_assets_download.json").read_text())
if state["phase"] != "download_verified":
    raise RuntimeError("Archive verification is required")
target.mkdir(exist_ok=True)
count = 0
with zipfile.ZipFile(source) as archive:
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if "assets" not in parts:
            raise ValueError(f"Unexpected archive layout: {member.filename}")
        relative = Path(*parts[parts.index("assets"):])
        destination = (target / relative).resolve()
        if not destination.is_relative_to(target.resolve()) or ".." in parts:
            raise ValueError(f"Unsafe archive path: {member.filename}")
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        count += 1
        if count % 20000 == 0:
            print(f"Extracted {count} files", flush=True)
(out / "plus_extraction.json").write_text(json.dumps(
    {"phase": "extracted", "files": count, "target": str(target), "environment_validation": "pending"},
    indent=2))
print(f"Extracted {count} files to {target}", flush=True)
