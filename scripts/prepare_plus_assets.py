"""Download the official LIBERO-Plus asset archive; never launch GPU work."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    out = args.root / "experiments/action_only"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "plus_assets_download.log").open("a") as log:
            process = subprocess.Popen([sys.executable, __file__, "--root", str(args.root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(out / "plus_assets_download.log")}))
        return
    lock = (out / "plus_assets_download.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    archive = args.root / "checkpoints/benchmark_assets/libero_plus_assets.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    state_path = out / "plus_assets_download.json"
    state = {"pid": os.getpid(), "phase": "resolving", "archive": str(archive)}

    def status(phase, **kwargs):
        state.update(phase=phase, **kwargs)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    try:
        response = requests.get("https://huggingface.co/api/datasets/Sylvest/LIBERO-plus", timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
        response = requests.get(f"https://huggingface.co/api/datasets/Sylvest/LIBERO-plus/tree/{revision}", timeout=30)
        response.raise_for_status()
        asset = next(item for item in response.json() if item["path"] == "assets.zip")
        expected = asset["lfs"]["oid"]
        size = asset["size"]
        partial = archive.with_suffix(".partial")
        identity_path = archive.with_suffix(".identity.json")
        identity = {"revision": revision, "sha256": expected, "size": size}
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise RuntimeError("Asset revision changed; refusing to resume different content")
        if not identity_path.exists() and (partial.exists() or archive.exists()):
            raise RuntimeError("Unidentified existing archive; manual audit required")
        identity_path.write_text(json.dumps(identity, indent=2))
        if not archive.exists():
            if shutil.disk_usage(archive.parent).free < size + 10 * 1024**3:
                raise RuntimeError("Insufficient disk headroom for assets")
            url = f"https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/{revision}/assets.zip"
            status("downloading", **identity)
            # Start a fresh curl for each retry so the resume offset is reread.
            # HTTP authorization/range errors are not retried.
            deadline = time.monotonic() + 48 * 3600
            for attempt in range(1, 101):
                status("downloading", attempt=attempt,
                       downloaded_bytes=partial.stat().st_size if partial.exists() else 0)
                result = subprocess.run(["curl", "--http1.1", "--fail", "--location",
                    "--connect-timeout", "30", "--max-time", "1800", "--continue-at", "-",
                    "--output", str(partial), url])
                if result.returncode == 0:
                    break
                if result.returncode not in {18, 28, 35, 52, 55, 56, 92}:
                    raise RuntimeError(f"Non-retryable curl error {result.returncode}")
                if attempt == 100 or time.monotonic() >= deadline:
                    raise RuntimeError("Asset transfer exhausted bounded retry budget")
                print(f"Transient curl error {result.returncode}; resuming attempt {attempt + 1}", flush=True)
                time.sleep(20)
        target = archive if archive.exists() else partial
        status("verifying", **identity)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if target.stat().st_size != size or digest.hexdigest() != expected:
            raise RuntimeError("Asset archive size/hash verification failed")
        if target == partial:
            partial.replace(archive)
        status("download_verified", extraction="pending; environment not validated")
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
