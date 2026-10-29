"""Read-only test of the actual dataset adapter with current-only observations."""
import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from Roboactualizer.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / 'configs'),
                               version_base='1.3'):
        config = compose(config_name='train', overrides=['task=libero_action_only'])
    dataset = BaseLerobotDataset(
        dataset_dirs=[args.dataset],
        shape_meta=OmegaConf.to_container(config.data.train.shape_meta, resolve=True),
        action_size=16, obs_size=17, current_images_only=True,
        val_set_proportion=0, is_training_set=True,
    )
    dataset._set_return_images(False)
    report = {'dataset': args.dataset, 'frames': len(dataset), 'samples': [],
              'limitations': 'Real cached-mode base adapter only; no RGB cache/text cache or processor applied.'}
    for index in (0, 6, len(dataset) - 1):
        sample = dataset[index]
        action = sample['action']['default']
        state = sample['state']['default']
        assert action.shape == (16, 7) and torch.isfinite(action).all()
        assert state.shape == (17, 8) and torch.isfinite(state).all()
        assert sample['image_is_pad'].numel() == 1
        assert sample['action_is_pad'].numel() == 16
        report['samples'].append({'index': index, 'task': sample['task'],
                                  'action_shape': list(action.shape),
                                  'state_shape': list(state.shape),
                                  'padded_actions': int(sample['action_is_pad'].sum())})
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
