"""Ask the official Dream-VLA trainer to save at its next update and exit."""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
args = parser.parse_args()
run_dir = args.run_dir.resolve(strict=True)
if not (run_dir / 'dataset_statistics.json').is_file():
    raise SystemExit('Expected a Dream-VLA run directory containing dataset_statistics.json')
request = run_dir / 'PAUSE_REQUEST'
request.touch(exist_ok=True)
print(f'Pause requested: {request}')
print('Wait for the trainer to finish saving and exit before reusing its GPUs.')
