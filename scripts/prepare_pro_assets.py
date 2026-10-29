"""Fetch official LIBERO-Pro BDDL/init assets into an isolated data directory."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    out = args.root / "experiments/action_only"
    out.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (out / "pro_assets_download.log").open("a") as log:
            process = subprocess.Popen([sys.executable, __file__, "--root", str(args.root)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        print(json.dumps({"pid": process.pid, "log": str(out / "pro_assets_download.log")}))
        return
    lock = (out / "pro_assets_download.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    target = args.root / "checkpoints/benchmark_assets/libero_pro"
    state_path = out / "pro_assets_download.json"
    state = {"pid": os.getpid(), "target": str(target), "repo": "zhouxueyang/LIBERO-Pro"}

    def status(phase, **extra):
        state.update(phase=phase, **extra)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    try:
        from huggingface_hub import HfApi, snapshot_download
        status("resolving")
        identity_path = out / "pro_assets_revision.json"
        if identity_path.exists():
            revision = json.loads(identity_path.read_text())["revision"]
        else:
            revision = HfApi().dataset_info(state["repo"]).sha
            identity_path.write_text(json.dumps({"revision": revision}))
        status("downloading", revision=revision)
        snapshot_download(repo_id=state["repo"], repo_type="dataset", revision=revision,
            local_dir=str(target), max_workers=2,
            allow_patterns=["bddl_files/**", "init_files/**", "metadata/**", "SHA256SUMS.txt", "README.md"])
        status("verifying")
        checked = 0
        for line in (target / "SHA256SUMS.txt").read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            path = (target / relative.lstrip("*")).resolve()
            if not path.is_relative_to(target.resolve()):
                raise ValueError("Invalid path in checksum manifest")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
            checked += 1
        if not checked:
            raise ValueError("Empty checksum manifest")
        status("download_verified", checked_files=checked, environment_validation="pending")
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
