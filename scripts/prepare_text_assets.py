"""Fetch only the policy's official text encoder/tokenizer, without CUDA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--detach', action='store_true')
    args = parser.parse_args()
    out = args.root / 'experiments/action_only'
    out.mkdir(exist_ok=True, parents=True)
    if args.detach:
        with (out / 'text_download.log').open('a') as log:
            process = subprocess.Popen([sys.executable, __file__, '--root', str(args.root)],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       start_new_session=True)
        (out / 'text_download_launch.json').write_text(json.dumps({'pid': process.pid}))
        print(process.pid)
        return
    import fcntl
    lock = (out / 'text_download.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(args.root / 'checkpoints/text')
    os.environ['DIFFSYNTH_DOWNLOAD_SOURCE'] = 'modelscope'
    from Roboactualizer.models.wan22.helpers.loader import _resolve_configs
    text, tokenizer = _resolve_configs('Wan-AI/Wan2.2-TI2V-5B', 'Wan-AI/Wan2.1-T2V-1.3B')
    text.download_if_necessary()
    tokenizer.download_if_necessary()
    (out / 'text_assets.json').write_text(json.dumps({'text': text.path, 'tokenizer': tokenizer.path,
                                                     'validation': 'download only; strict loading pending'}, indent=2))


if __name__ == '__main__':
    main()
