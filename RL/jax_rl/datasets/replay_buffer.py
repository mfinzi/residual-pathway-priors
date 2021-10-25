from typing import Optional

import gym
import numpy as np

from jax_rl.datasets.dataset import Dataset
import pandas as pd
from .welford import Welford
from .equivariant_standardization import EquivStandardizer
import abc
import collections

import numpy as np

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int,rep,state_transform,inv_state_transform,standardize=False):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity
        self.restarts = np.zeros(capacity)
        
        self.running_stats = EquivStandardizer(rep,state_transform,inv_state_transform) if standardize else None

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert False, "Compatibility with running stats not implemented"
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, discount: float, next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = discount
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.running_stats is not None: self.running_stats.add_data(observation)

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        obs = self.observations[indx]
        next_obs = self.next_observations[indx]
        # if self.normalize:
        #     obs = (obs-self.running_stats.mean())/self.running_stats.std()
        #     next_obs = (next_obs-self.running_stats.mean())/self.running_stats.std()
        return Batch(observations=obs,
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=next_obs)

    def as_df(self):
        data = {
            'rewards':self.rewards[:self.insert_index],
            'masks':self.masks[:self.insert_index],
            'restarts':self.restarts[:self.insert_index],
        }
        for i in range(self.observations.shape[-1]):
            data[f'x{i}'] = self.observations[:self.insert_index,i]
            #data[f'next_x{i}'] = self.next_observations[:self.insert_index,i]
        for i in range(self.actions.shape[-1]):
            data[f'u{i}'] = self.actions[:self.insert_index,i]
        return pd.DataFrame(data)

    def save(self, path):
        self.as_df().to_csv(path)
