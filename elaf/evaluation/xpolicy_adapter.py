"""XPolicyLab transport adapter; simulator workers never load model weights.

Decoded XPolicyLab images are RGB (including its historical cv2 byte roundtrip).
Native state/action order is arm0, gripper0, arm1, gripper1. No implicit LIBERO
rotation, gripper inversion, normalization, or single-to-dual-arm padding occurs.
"""
from __future__ import annotations
from pathlib import Path
import uuid
import hashlib
import numpy as np


class Model:
    def __init__(self, model_cfg, client=None):
        self.cfg = dict(model_cfg)
        self.mode = self.cfg.get('action_type', 'joint')
        if self.mode not in ('joint', 'ee'):
            raise ValueError('XPolicy action_type must be joint or ee')
        self.arm_dims = list(self.cfg['arm_dims'])
        self.gripper_dims = list(self.cfg.get('gripper_dims', [1] * len(self.arm_dims)))
        if len(self.arm_dims) not in (1, 2) or len(self.arm_dims) != len(self.gripper_dims):
            raise ValueError('Specify one or two arms and matching gripper_dims')
        if any(type(d) is not int or d < 1 for d in self.arm_dims + self.gripper_dims):
            raise ValueError('All arm/gripper dimensions must be positive integers')
        self.action_dims = self.arm_dims if self.mode == 'joint' else [7] * len(self.arm_dims)
        self.action_dim = sum(self.action_dims + self.gripper_dims)
        self.action_space = 'xpolicy_joint_absolute' if self.mode == 'joint' else 'xpolicy_ee_absolute_wxyz'
        if self.cfg.get('action_space') != self.action_space:
            raise ValueError(f'Explicit action_space={self.action_space!r} required; LIBERO delta actions cannot be reused')
        if self.cfg.get('normalization') != 'service_physical':
            raise ValueError('normalization must be service_physical: service returns denormalized physical actions')
        self.camera = self.cfg['camera']
        self.flip_vertical = bool(self.cfg.get('flip_vertical', False))
        self.state_fields = self.cfg['state_fields']
        if not self.state_fields:
            raise ValueError('state_fields must declare the training proprio order')
        self.proprio_dim = sum(int(f['dim']) for f in self.state_fields)
        if client is None:
            from elaf.evaluation.rpc import PolicyClient
            auth = Path(self.cfg['auth_file']).read_bytes()
            client = PolicyClient((self.cfg.get('service_host', '127.0.0.1'), int(self.cfg['service_port'])), auth)
        self.client = client
        meta = client.metadata
        if meta.get('action_dim') != self.action_dim or meta.get('action_space') != self.action_space:
            raise ValueError(f'Checkpoint incompatible with native robot: need {self.action_dim}D {self.action_space}; service metadata={meta}')
        if 'proprio_dim' in meta and meta['proprio_dim'] != self.proprio_dim:
            raise ValueError('Checkpoint proprio_dim differs from declared state layout')
        self.horizon = int(meta['action_horizon'])
        self.execute_steps = int(self.cfg.get('execute_steps', self.horizon))
        if not 1 <= self.execute_steps <= self.horizon:
            raise ValueError('execute_steps must be within checkpoint action horizon')
        self.base_seed = int(self.cfg.get("seed", 0))
        if not 0 <= self.base_seed < 2**63:
            raise ValueError("seed must be in [0,2**63)")
        self.episode_index = -1
        self.reset()

    def reset(self):
        self.episode = uuid.uuid4().hex
        self.episode_index += 1
        self.replan_counts = {}
        self.observations = {}
        self.latest_ids = []

    def _convert_obs(self, obs):
        image = np.asarray(obs['vision'][self.camera]['color'])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError('Expected decoded HWC uint8 RGB camera; no silent BGR/float conversion')
        if self.flip_vertical:
            image = image[::-1]
        parts = []
        for field in self.state_fields:
            value = np.asarray(obs['state'][field['key']], dtype=np.float32).reshape(-1)
            if value.size != field['dim'] or not np.isfinite(value).all():
                raise ValueError(f"Invalid proprio field {field['key']}")
            parts.append(value)
        instruction = obs['instruction']
        if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
            instruction = instruction[0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Nonempty task instruction string required')
        return {'image': np.ascontiguousarray(image), 'proprio': np.concatenate(parts).tolist(), 'instruction': instruction}

    def update_obs(self, obs):
        env_id = int(obs.get('env_idx', 0))
        self.observations[env_id] = self._convert_obs(obs)
        self.latest_ids = [env_id]

    def update_obs_batch(self, obs_list):
        ids = [int(obs['env_idx']) for obs in obs_list]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate env_idx in batch')
        for env_id, obs in zip(ids, obs_list):
            self.observations[env_id] = self._convert_obs(obs)
        self.latest_ids = ids

    def _unpack(self, vector):
        result, offset = {}, 0
        for i, (arm_dim, grip_dim) in enumerate(zip(self.action_dims, self.gripper_dims)):
            prefix = ('left_' if i == 0 else 'right_') if len(self.arm_dims) == 2 else ''
            key = prefix + ('arm_joint_state' if self.mode == 'joint' else 'ee_pose')
            arm = vector[offset:offset + arm_dim].copy()
            if self.mode == 'ee':
                norm = np.linalg.norm(arm[3:7])
                if not np.isclose(norm, 1.0, atol=0.02):
                    raise ValueError('EE output quaternion must be normalized wxyz; invalid trained action representation')
                arm[3:7] /= norm
            result[key] = arm
            offset += arm_dim
            result[prefix + 'ee_joint_state'] = vector[offset:offset + grip_dim].copy()
            offset += grip_dim
        return result

    def get_action_batch(self, env_idx_list=None):
        ids = self.latest_ids if env_idx_list is None else [int(i) for i in env_idx_list]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Require nonempty unique active environment indices')
        missing = set(ids) - self.observations.keys()
        if missing:
            raise ValueError(f'Missing observations for environments {sorted(missing)}')
        observations = []
        for i in ids:
            count = self.replan_counts.get(i, 0)
            identity = f'{self.base_seed}:{self.episode_index}:{i}:{count}'.encode()
            seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'little') % (2**63)
            observations.append(dict(self.observations[i], noise_seed=seed))
        actions = np.asarray(self.client.predict(observations, episode_keys=[f'{self.episode}:{i}' for i in ids]), dtype=np.float32)
        if actions.shape != (len(ids), self.horizon, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid action batch shape/values {actions.shape}')
        result = [[self._unpack(action) for action in chunk[:self.execute_steps]] for chunk in actions]
        for i in ids:
            self.replan_counts[i] = self.replan_counts.get(i, 0) + 1
        return result

    def get_action(self):
        if len(self.latest_ids) != 1:
            raise ValueError('get_action requires exactly one observation')
        return self.get_action_batch(self.latest_ids)[0]

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()
