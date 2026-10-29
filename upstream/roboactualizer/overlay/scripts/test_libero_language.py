"""Verify evaluation prompts come from the real Pro BDDL, not task filenames."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libero.libero import benchmark
from experiments.libero.libero_utils import get_libero_task_description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial_lan"]()
    records = []
    for index in range(suite.n_tasks):
        task = suite.get_task(index)
        records.append({"index": index, "filename_language": task.language,
                        "evaluation_language": get_libero_task_description(task)})
    changed = sum(item["filename_language"] != item["evaluation_language"] for item in records)
    assert changed > 0, "Language perturbation was not exercised"
    report = {"status": "PASS", "benchmark_module": benchmark.__file__,
              "tasks": records, "changed_prompts": changed,
              "scope": "Real BDDL prompt extraction only; no text embedding or policy rollout."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
