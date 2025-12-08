"""Inference entrypoint compatible with gfootball's Player API."""

import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np
import torch

from gfootball.env import football_action_set
from gfootball.env import player_base
from rl_agents.networks import FootballPolicy


class Player(player_base.PlayerBase):
    """Loads a trained PyTorch policy and serves actions to the environment."""

    def __init__(self, player_config, env_config):
        super().__init__(player_config)
        self._config = env_config
        self._num_actions = len(football_action_set.get_action_set(env_config))
        controls = env_config.number_of_players_agent_controls()
        if controls > 1:
            self._action_space = gym.spaces.MultiDiscrete([self._num_actions] * controls)
        else:
            self._action_space = gym.spaces.Discrete(self._num_actions)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint_path = Path(os.environ.get("FOOTBALL_CHECKPOINT", "policy/best_model.pt"))
        self._policy: Optional[FootballPolicy] = None
        self._loaded = False

    def _lazy_load(self, obs: np.ndarray):
        if np.issubdtype(obs.dtype, np.floating):
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype)
        else:
            obs_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
        self._policy = FootballPolicy(obs_space, self._action_space).to(self._device)
        if self._checkpoint_path.exists():
            payload = torch.load(self._checkpoint_path, map_location=self._device)
            state_dict = payload.get("model_state", payload)
            self._policy.load_state_dict(state_dict)
            self._policy.eval()
            self._loaded = True
        else:
            print(f"[agent] checkpoint not found at {self._checkpoint_path}, running untrained policy.")

    def take_action(self, observations):
        obs = observations[0] if isinstance(observations, (list, tuple)) else observations
        if self._policy is None:
            self._lazy_load(np.asarray(obs))
        obs_tensor = torch.as_tensor(obs, device=self._device)
        with torch.no_grad():
            action, _, _, _ = self._policy.act(obs_tensor)
        action_np = action.cpu().numpy()
        if isinstance(self._action_space, gym.spaces.Discrete):
            return int(action_np)
        return action_np.astype(np.int64)
