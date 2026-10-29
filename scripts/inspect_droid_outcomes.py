"""Read schema and provenance fields only, never action/language values."""
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path("/data/users/jiachangzhang/eb_vjepa/data/droid_1.0.1")
for folder in ("meta/episodes", "data"):
    files = sorted((root / folder).rglob("*.parquet"))
    schema = pq.read_schema(files[0])
    print(folder, "FILES", len(files), "SCHEMA", schema.names, flush=True)
    candidates = [n for n in schema.names if any(
        word in n.lower() for word in ("success", "failure", "file_path", "folderpath", "terminal"))]
    print("CANDIDATES", candidates, flush=True)
    for name in candidates:
        print(name, pq.read_table(files[0], columns=[name]).slice(0, 5).to_pylist(), flush=True)
