"""Model definitions for agents trained on Google Research Football."""

from typing import List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _action_dimensions(action_space: gym.Space) -> Tuple[int, Optional[List[int]]]:
    """Returns flattened action dimension and splits for MultiDiscrete spaces."""
    if isinstance(action_space, gym.spaces.Discrete):
        return int(action_space.n), None
    if isinstance(action_space, gym.spaces.MultiDiscrete):
        nvec: List[int] = [int(x) for x in action_space.nvec.tolist()]
        return int(sum(nvec)), nvec
    raise ValueError(f"Unsupported action space: {action_space}")


class MultiCategorical:
    """Utility distribution for MultiDiscrete action spaces."""

    def __init__(self, logits: torch.Tensor, nvec: Sequence[int]):
        self._nvec = list(nvec)
        self._dists: List[Categorical] = []
        cursor = 0
        for n in self._nvec:
            self._dists.append(Categorical(logits=logits[:, cursor : cursor + n]))
            cursor += n

    def sample(self) -> torch.Tensor:
        samples = [dist.sample() for dist in self._dists]
        return torch.stack(samples, dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        logps = []
        for idx, dist in enumerate(self._dists):
            logps.append(dist.log_prob(actions[..., idx]))
        return torch.stack(logps, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        ent = torch.stack([dist.entropy() for dist in self._dists], dim=-1)
        return ent.sum(dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaEncoder(nn.Module):
    """Lightweight IMPALA-style encoder used in the original paper."""

    def __init__(self, input_channels: int, hidden_size: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ImpalaBlock(input_channels, 32),
                ImpalaBlock(32, 64),
                ImpalaBlock(64, 96),
                ImpalaBlock(96, 128),
            ]
        )
        self.out_pool = nn.AdaptiveAvgPool2d((6, 8))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128 * 6 * 8, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.out_pool(x)
        x = self.flatten(x)
        return self.fc(x)


class FootballPolicy(nn.Module):
    """Shared actor-critic policy."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, hidden_size: int = 256):
        super().__init__()
        self.action_space = action_space
        self._action_dim, self._action_splits = _action_dimensions(action_space)

        obs_shape = observation_space.shape
        self._use_conv = len(obs_shape) == 3  # HWC

        if self._use_conv:
            in_channels = obs_shape[2]
            self.encoder = ImpalaEncoder(in_channels, hidden_size)
        else:
            flat = int(np.prod(obs_shape))
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
        self.actor = nn.Linear(hidden_size, self._action_dim)
        self.critic = nn.Linear(hidden_size, 1)

    def _prepare_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._use_conv:
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            if obs.dim() == 4 and obs.shape[-1] <= 32 and obs.shape[-1] != obs.shape[1]:
                # Env returns HWC; move channels to first position.
                obs = obs.permute(0, 3, 1, 2)
            obs = obs.float() / 255.0
        else:
            obs = obs.float()
        return obs

    def _distribution(self, logits: torch.Tensor):
        if self._action_splits is None:
            return Categorical(logits=logits)
        return MultiCategorical(logits, self._action_splits)

    def forward(self, obs: torch.Tensor):
        obs = self._prepare_obs(obs)
        features = self.encoder(obs)
        policy_logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return policy_logits, value

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = self._distribution(logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = self._distribution(logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
