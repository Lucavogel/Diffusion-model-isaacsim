from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class GenericImageDataset(BaseImageDataset):
    def __init__(self, shape_meta: dict, dataset_path: str, horizon=1, pad_before=0, pad_after=0, n_obs_steps=None, n_latency_steps=0, use_cache=False, seed=42, val_ratio=0.0, max_train_episodes=None, delta_action=False):
        super().__init__()
        self.shape_meta = shape_meta
        keys = list(shape_meta['obs'].keys()) + ['action']
        self.replay_buffer = ReplayBuffer.copy_from_path(dataset_path, keys=keys)
        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=horizon, pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer, sequence_length=self.horizon, pad_before=self.pad_before, pad_after=self.pad_after, episode_mask=~self.train_mask)
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = { 'action': self.replay_buffer['action'][:] }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        for key, attr in self.shape_meta['obs'].items():
            if attr.get('type') == 'rgb':
                normalizer[key] = get_image_range_normalizer()
            else:
                obs_data = { key: self.replay_buffer[key][:] }
                obs_normalizer = LinearNormalizer()
                obs_normalizer.fit(data=obs_data, last_n_dims=1, mode=mode, **kwargs)
                normalizer[key] = obs_normalizer[key]
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = dict()
        for key, attr in self.shape_meta['obs'].items():
            if attr.get('type') == 'rgb':
                image = np.moveaxis(sample[key], -1, 1)/255.
                data[key] = torch.from_numpy(image).float()
            else:
                data[key] = torch.from_numpy(sample[key]).float()
        data['action'] = torch.from_numpy(sample['action']).float()
        return data
