#!/usr/bin/env python3
"""Download the pinned official Dream-VLA LIBERO RLDS dataset using curl, then verify every file."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

REPO = "openvla/modified_libero_rlds"
REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common = ["curl", "--fail", "--location", "--silent", "--show-error",
              "--connect-timeout", "30", "--retry", "5", "--retry-delay", "5"]
    if args.proxy:
        common += ["--proxy", args.proxy]
    endpoint = args.endpoint.rstrip("/")
    metadata = json.loads(subprocess.check_output(common + ["--max-time", "120",
        f"{endpoint}/api/datasets/{REPO}/revision/{REVISION}?blobs=true"]))
    if metadata["sha"] != REVISION:
        raise RuntimeError("API revision differs from pinned revision")
    (args.output / "source_metadata.json").write_text(json.dumps(metadata, indent=2))

    def check(path, entry):
        size = path.stat().st_size
        if size != entry["size"]:
            raise RuntimeError(f"{path.name}: size {size} != {entry['size']}")
        sha = hashlib.sha256()
        git = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
                git.update(chunk)
        if "lfs" in entry:
            if sha.hexdigest() != entry["lfs"]["sha256"]:
                raise RuntimeError(f"{path.name}: LFS SHA256 mismatch")
        elif git.hexdigest() != entry["blobId"]:
            raise RuntimeError(f"{path.name}: Git blob SHA1 mismatch")
        return {"file": entry["rfilename"], "bytes": size, "sha256": sha.hexdigest(),
                "git_blob_sha1": git.hexdigest(), "verified": True}

    def fetch(entry):
        name = entry["rfilename"]
        path = args.output / name
        if not path.resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Unsafe upstream filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            result = check(path, entry)
            print(f"VERIFIED existing {name}", flush=True)
            return result
        partial = path.with_name(path.name + ".partial")
        print(f"DOWNLOAD {name} ({entry['size']} bytes)", flush=True)
        # A fresh resolve request on each restart avoids expired CDN signed URLs.
        for attempt in range(4):
            status = subprocess.run(common + ["--continue-at", "-", "--output", str(partial),
                f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/{name}?download=true&attempt={attempt}"]).returncode
            if status == 0:
                break
            time.sleep(5)
        if status:
            raise RuntimeError(f"{name}: curl failed ({status}); partial retained for resume")
        result = check(partial, entry)
        os.replace(partial, path)
        print(f"VERIFIED {name}", flush=True)
        return result

    entries = sorted(metadata["siblings"], key=lambda entry: entry["size"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(fetch, entries))
    receipt = {"repo_id": REPO, "revision": REVISION, "endpoint": endpoint,
               "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "total_bytes": sum(f["bytes"] for f in files), "files": files}
    (args.output / "download_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(f"COMPLETE {len(files)} verified files; {receipt['total_bytes']} bytes", flush=True)


if __name__ == "__main__":
    main()
