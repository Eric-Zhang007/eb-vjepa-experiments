"""Read-only full AV1 decode audit; never replaces frames or dataset samples."""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import time
import av

root = Path(__file__).resolve().parents[1]
out = root / "experiments/action_only"
lock = (out / "video_audit.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
data = Path("/data/public/zhongy/starvla/Datasets/LEROBOT_LIBERO_DATA")
files = sorted(data.glob("*/videos/**/*.mp4"))
started = time.time()


def check(path):
    decoded = 0
    expected = None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            expected = stream.frames
            for frame in container.decode(stream):
                decoded += 1
        if expected and decoded != expected:
            raise ValueError(f"Decoded {decoded} != container frames {expected}")
        return None
    except Exception as exc:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "error": str(exc), "decoded_frames": decoded,
                "container_frames": expected, "sha256": digest, "bytes": path.stat().st_size}


failures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for index, failure in enumerate(pool.map(check, files), 1):
        if failure:
            failures.append(failure)
            print(json.dumps(failure), flush=True)
        if index % 100 == 0:
            print(f"Audited {index}/{len(files)}, failures={len(failures)}", flush=True)
report = {"status": "FAIL" if failures else "PASS", "files": len(files),
          "failures": failures, "elapsed_seconds": time.time() - started,
          "method": "PyAV/libdav1d, all packets, one decoder thread, no frame substitution"}
path = out / "video_audit.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(report, indent=2))
temporary.replace(path)
print(json.dumps({"status": report["status"], "files": len(files), "failures": len(failures)}), flush=True)
