# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Environment that can be used with OpenAI Baselines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import cv2
from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
import gym
import numpy as np


class GetStateWrapper(gym.Wrapper):
  """A wrapper that only dumps traces/videos periodically."""

  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self._wrappers_with_support = {
        'CheckpointRewardWrapper', 'FrameStack', 'GetStateWrapper',
        'SingleAgentRewardWrapper', 'SingleAgentObservationWrapper',
        'SMMWrapper', 'PeriodicDumpWriter', 'Simple115StateWrapper',
        'PixelsStateWrapper', 'DenseRewardWrapper'
    }

  def _check_state_supported(self):
    o = self
    while True:
      name = o.__class__.__name__
      if o.__class__.__name__ == 'FootballEnv':
        break
      assert name in self._wrappers_with_support, (
          'get/set state not supported'
          ' by {} wrapper').format(name)
      o = o.env

  def get_state(self):
    self._check_state_supported()
    to_pickle = {}
    return self.env.get_state(to_pickle)

  def set_state(self, state):
    self._check_state_supported()
    self.env.set_state(state)


class PeriodicDumpWriter(gym.Wrapper):
  """A wrapper that only dumps traces/videos periodically."""

  def __init__(self, env, dump_frequency, render=False):
    gym.Wrapper.__init__(self, env)
    self._dump_frequency = dump_frequency
    self._render = render
    self._original_dump_config = {
        'write_video': env._config['write_video'],
        'dump_full_episodes': env._config['dump_full_episodes'],
        'dump_scores': env._config['dump_scores'],
    }
    self._current_episode_number = 0

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    if (self._dump_frequency > 0 and
        (self._current_episode_number % self._dump_frequency == 0)):
      self.env._config.update(self._original_dump_config)
      if self._render:
        self.env.render()
    else:
      self.env._config.update({'write_video': False,
                               'dump_full_episodes': False,
                               'dump_scores': False})
      if self._render:
        self.env.disable_render()
    self._current_episode_number += 1
    return self.env.reset()


class Simple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state."""

  def __init__(self, env, fixed_positions=False):
    """Initializes the wrapper.

    Args:
      env: an envorinment to wrap
      fixed_positions: whether to fix observation indexes corresponding to teams
    Note: simple115v2 enables fixed_positions option.
    """
    gym.ObservationWrapper.__init__(self, env)
    action_shape = np.shape(self.env.action_space)
    shape = (action_shape[0] if len(action_shape) else 1, 115)
    self.observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
    self._fixed_positions = fixed_positions

  def observation(self, observation):
    """Converts an observation into simple115 (or simple115v2) format."""
    return Simple115StateWrapper.convert_observation(observation, self._fixed_positions)

  @staticmethod
  def convert_observation(observation, fixed_positions):
    """Converts an observation into simple115 (or simple115v2) format.

    Args:
      observation: observation that the environment returns
      fixed_positions: Players and positions are always occupying 88 fields
                       (even if the game is played 1v1).
                       If True, the position of the player will be the same - no
                       matter how many players are on the field:
                       (so first 11 pairs will belong to the first team, even
                       if it has less players).
                       If False, then the position of players from team2
                       will depend on number of players in team1).

    Returns:
      (N, 115) shaped representation, where N stands for the number of players
      being controlled.
    """

    def do_flatten(obj):
      """Run flatten on either python list or numpy array."""
      if type(obj) == list:
        return np.array(obj).flatten()
      return obj.flatten()

    final_obs = []
    for obs in observation:
      o = []
      if fixed_positions:
        for i, name in enumerate(['left_team', 'left_team_direction',
                                  'right_team', 'right_team_direction']):
          o.extend(do_flatten(obs[name]))
          # If there were less than 11vs11 players we backfill missing values
          # with -1.
          if len(o) < (i + 1) * 22:
            o.extend([-1] * ((i + 1) * 22 - len(o)))
      else:
        o.extend(do_flatten(obs['left_team']))
        o.extend(do_flatten(obs['left_team_direction']))
        o.extend(do_flatten(obs['right_team']))
        o.extend(do_flatten(obs['right_team_direction']))

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
      if len(o) < 88:
        o.extend([-1] * (88 - len(o)))

      # ball position
      o.extend(obs['ball'])
      # ball direction
      o.extend(obs['ball_direction'])
      # one hot encoding of which team owns the ball
      if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
      if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
      if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])

      active = [0] * 11
      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)
      final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)


class PixelsStateWrapper(gym.ObservationWrapper):
  """A wrapper that extracts pixel representation."""

  def __init__(self, env, grayscale=True,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._grayscale = grayscale
    self._channel_dimensions = channel_dimensions
    action_shape = np.shape(self.env.action_space)
    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(action_shape[0] if len(action_shape) else 1,
               channel_dimensions[1], channel_dimensions[0],
               1 if grayscale else 3),
        dtype=np.uint8)

  def observation(self, obs):
    o = []
    for observation in obs:
      assert 'frame' in observation, ("Missing 'frame' in observations. Pixel "
                                      "representation requires rendering and is"
                                      " supported only for players on the left "
                                      "team.")
      frame = observation['frame']
      if self._grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      frame = cv2.resize(frame, (self._channel_dimensions[0],
                                 self._channel_dimensions[1]),
                         interpolation=cv2.INTER_AREA)
      if self._grayscale:
        frame = np.expand_dims(frame, -1)
      o.append(frame)
    return np.array(o, dtype=np.uint8)


class SMMWrapper(gym.ObservationWrapper):
  """A wrapper that convers observations into a minimap format."""

  def __init__(self, env,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._channel_dimensions = channel_dimensions
    action_shape = np.shape(self.env.action_space)
    shape = (action_shape[0] if len(action_shape) else 1, channel_dimensions[1],
             channel_dimensions[0],
             len(
                 observation_preprocessing.get_smm_layers(
                     self.env.unwrapped._config)))
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=shape, dtype=np.uint8)

  def observation(self, obs):
    return observation_preprocessing.generate_smm(
        obs, channel_dimensions=self._channel_dimensions,
        config=self.env.unwrapped._config)


class SingleAgentObservationWrapper(gym.ObservationWrapper):
  """A wrapper that returns an observation only for the first agent."""

  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        dtype=env.observation_space.dtype)

  def observation(self, obs):
    return obs[0]


class SingleAgentRewardWrapper(gym.RewardWrapper):
  """A wrapper that converts an observation to a minimap."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    return reward[0]


class CheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {}
    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1

  def reset(self):
    self._collected_checkpoints = {}
    return self.env.reset()

  def get_state(self, to_pickle):
    to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
    return self.env.get_state(to_pickle)

  def set_state(self, state):
    from_pickle = self.env.set_state(state)
    self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
    return from_pickle

  def reward(self, reward):
    observation = self.env.unwrapped.observation()
    if observation is None:
      return reward

    assert len(reward) == len(observation)

    for rew_index in range(len(reward)):
      o = observation[rew_index]
      if reward[rew_index] == 1:
        reward[rew_index] += self._checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints.get(rew_index, 0))
        self._collected_checkpoints[rew_index] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != 0 or
          'ball_owned_player' not in o or
          o['ball_owned_player'] != o['active']):
        continue

      d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints.get(rew_index, 0) <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints.get(rew_index, 0))
        if d > threshold:
          break
        reward[rew_index] += self._checkpoint_reward
        self._collected_checkpoints[rew_index] = (
            self._collected_checkpoints.get(rew_index, 0) + 1)
    return reward


class DenseRewardWrapper(gym.Wrapper):
  """Reward shaping tuned for faster feedback on football objectives.

  Adds small, normalized bonuses/penalties for intermediate events and tracks
  episode metrics for logging.
  """

  def __init__(self, env, agent_controls_left=True):
    super(DenseRewardWrapper, self).__init__(env)
    self._agent_team = 0 if agent_controls_left else 1
    self._opponent_team = 1 - self._agent_team
    self._target_goal_x = 1.0  # Observations are mirrored so opponent goal is +1.
    self._progress_threshold = 1e-3
    self._shot_threshold_x = 0.9
    self._shot_band_y = 0.12
    self._close_x_threshold = 0.7
    self._close_reward_scale = 0.03
    self._stall_penalty = 0.002
    self._stall_progress_eps = 5e-4
    self._stall_x_threshold = 0.0
    self._save_reward = 0.05
    self._threat_distance = -0.8
    self._goalkeeper_index = 0
    self._back_pass_penalty = 0.05
    self._own_goal_penalty = 0.7
    self._reward_clip_min = -2.0
    self._reward_clip_max = 1.0
    self._reset_tracking(None)

  def _reset_tracking(self, obs):
    self._has_touched_ball = False
    self._last_ball_position = None
    self._last_ball_owned_team = None
    self._last_ball_owned_player = None
    self._last_game_mode = 0
    self._last_touch_team = -1
    self._episode_return = 0.0
    self._episode_steps = 0
    self._possession_steps = 0
    self._ball_goal_delta_sum = 0.0
    self._ball_goal_delta_count = 0
    self._shots_on_target = 0
    self._goals_for = 0
    self._goals_against = 0
    self._stall_steps = 0
    self._threat_active = False
    self._back_pass_to_keeper_count = 0
    if obs is not None:
      obs_dict = self._get_obs_dict(obs)
      self._last_ball_position = np.array(
          obs_dict.get('ball', np.zeros(3, dtype=np.float32)), dtype=np.float32)
      self._last_ball_owned_team = obs_dict.get('ball_owned_team', -1)
      self._last_ball_owned_player = obs_dict.get('ball_owned_player', -1)
      if self._last_ball_owned_team in (0, 1):
        self._last_touch_team = self._last_ball_owned_team
      self._last_game_mode = obs_dict.get('game_mode', 0)
      score = obs_dict.get('score', [0, 0])
      self._goals_for, self._goals_against = self._score_to_for_against(score)

  def reset(self):
    obs = self.env.reset()
    self._reset_tracking(obs)
    return obs

  def _score_to_for_against(self, score):
    if self._agent_team == 0:
      return score[0], score[1]
    return score[1], score[0]

  def _get_obs_dict(self, obs):
    if isinstance(obs, (list, tuple)):
      return obs[0]
    return obs

  def _compute_shaping(self, obs_dict, score_reward):
    shaping_reward = 0.0
    ball = np.array(obs_dict.get('ball', np.zeros(3, dtype=np.float32)), dtype=np.float32)
    ball_x, ball_y = float(ball[0]), float(ball[1])
    current_owner = obs_dict.get('ball_owned_team', -1)
    current_player = obs_dict.get('ball_owned_player', -1)
    game_mode = obs_dict.get('game_mode', 0)

    if current_owner in (0, 1):
      self._last_touch_team = current_owner

    if not self._has_touched_ball and current_owner == self._agent_team:
      shaping_reward += 0.01
      self._has_touched_ball = True

    if (self._last_ball_owned_team == self._agent_team and
        current_owner == self._agent_team and
        current_player not in (-1, None) and
        self._last_ball_owned_player not in (-1, None) and
        current_player != self._last_ball_owned_player):
      shaping_reward += 0.04

    if self._last_ball_owned_team == self._agent_team and current_owner == self._opponent_team:
      shaping_reward -= 0.05

    if (self._last_game_mode == 0 and game_mode != 0 and
        self._last_touch_team == self._agent_team and score_reward == 0):
      shaping_reward -= 0.02

    if self._last_ball_position is not None:
      delta_x = ball_x - float(self._last_ball_position[0])
      if self._last_touch_team == self._agent_team and delta_x > self._progress_threshold:
        shaping_reward += 0.02 * min(1.0, delta_x / 0.05)
      if self._last_touch_team == self._opponent_team and delta_x < -self._progress_threshold:
        shaping_reward -= 0.02 * min(1.0, -delta_x / 0.05)
      if (self._last_touch_team == self._agent_team and
          float(self._last_ball_position[0]) <= self._shot_threshold_x < ball_x and
          abs(ball_y) <= self._shot_band_y):
        shaping_reward += 0.05
        self._shots_on_target += 1
      prev_goal_distance = self._target_goal_x - float(self._last_ball_position[0])
      goal_distance = self._target_goal_x - ball_x
        self._ball_goal_delta_sum += prev_goal_distance - goal_distance
        self._ball_goal_delta_count += 1

        # Closer to goal bonus when in control.
        if current_owner == self._agent_team and ball_x > self._close_x_threshold:
          close_gain = (ball_x - self._close_x_threshold) / (self._target_goal_x - self._close_x_threshold)
          shaping_reward += self._close_reward_scale * min(1.0, close_gain)

        # Penalize stalling in own half with the ball.
        if current_owner == self._agent_team and ball_x < self._stall_x_threshold:
          if delta_x <= self._stall_progress_eps:
            self._stall_steps += 1
            shaping_reward -= self._stall_penalty
          else:
            self._stall_steps = 0
        else:
          self._stall_steps = 0

        # Detect defensive saves: opponent near our goal, then we regain control without conceding.
        if current_owner == self._opponent_team and ball_x < self._threat_distance:
          self._threat_active = True
        if self._threat_active and current_owner == self._agent_team and score_reward >= 0:
          shaping_reward += self._save_reward
          self._threat_active = False
        if ball_x > -0.5:
          self._threat_active = False

    # Penalize excesso de recuos para o goleiro apos o primeiro.
    if (self._last_ball_owned_team == self._agent_team and
        current_owner == self._agent_team and
        current_player == self._goalkeeper_index and
        self._last_ball_owned_player not in (-1, None) and
        self._last_ball_owned_player != self._goalkeeper_index):
      self._back_pass_to_keeper_count += 1
      if self._back_pass_to_keeper_count > 1:
        shaping_reward -= self._back_pass_penalty

    # Penaliza fortemente gol contra (gol negativo com ultimo toque nosso).
    if score_reward < 0 and self._last_touch_team == self._agent_team:
      shaping_reward -= self._own_goal_penalty

    return shaping_reward

  def _build_episode_metrics(self):
    possession_fraction = (self._possession_steps / float(max(1, self._episode_steps)))
    ball_progress_avg = (self._ball_goal_delta_sum / float(max(1, self._ball_goal_delta_count)))
    return {
        'episode_return': float(self._episode_return),
        'goals': int(self._goals_for),
        'goals_against': int(self._goals_against),
        'shots_on_target': int(self._shots_on_target),
        'possession_fraction': float(possession_fraction),
        'ball_progress_avg': float(ball_progress_avg),
        'steps': int(self._episode_steps),
        'success': 1 if self._goals_for > 0 else 0,
        'win': 1 if self._goals_for > self._goals_against else 0,
    }

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs_dict = self._get_obs_dict(obs)
    score_reward = 0.0
    if isinstance(info, dict):
      score_reward = float(info.get('score_reward', 0.0))
    shaping_reward = self._compute_shaping(obs_dict, score_reward)
    reward_array = np.asarray(reward, dtype=np.float32) + shaping_reward
    reward_array = np.clip(reward_array, self._reward_clip_min, self._reward_clip_max)

    avg_reward = float(np.mean(reward_array))
    self._episode_return += avg_reward
    self._episode_steps += 1
    if obs_dict.get('ball_owned_team', -1) == self._agent_team:
      self._possession_steps += 1

    score = obs_dict.get('score', [0, 0])
    self._goals_for, self._goals_against = self._score_to_for_against(score)

    self._last_ball_owned_team = obs_dict.get('ball_owned_team', -1)
    self._last_ball_owned_player = obs_dict.get('ball_owned_player', -1)
    if self._last_ball_owned_team in (0, 1):
      self._last_touch_team = self._last_ball_owned_team
    self._last_game_mode = obs_dict.get('game_mode', self._last_game_mode)
    self._last_ball_position = np.array(
        obs_dict.get('ball', np.zeros(3, dtype=np.float32)), dtype=np.float32)

    if done:
      info = dict(info or {})
      info['episode_metrics'] = self._build_episode_metrics()
    return obs, reward_array, done, info


class FrameStack(gym.Wrapper):
  """Stack k last observations."""

  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.obs = collections.deque([], maxlen=k)
    low = env.observation_space.low
    high = env.observation_space.high
    low = np.concatenate([low] * k, axis=-1)
    high = np.concatenate([high] * k, axis=-1)
    self.observation_space = gym.spaces.Box(
        low=low, high=high, dtype=env.observation_space.dtype)

  def reset(self):
    observation = self.env.reset()
    self.obs.extend([observation] * self.obs.maxlen)
    return self._get_observation()

  def get_state(self, to_pickle):
    to_pickle['FrameStack'] = self.obs
    return self.env.get_state(to_pickle)

  def set_state(self, state):
    from_pickle = self.env.set_state(state)
    self.obs = from_pickle['FrameStack']
    return from_pickle

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.obs.append(observation)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return np.concatenate(list(self.obs), axis=-1)


class MultiAgentToSingleAgent(gym.Wrapper):
  """Converts raw multi-agent observations to single-agent observation.

  It returns observations of the designated player on the team, so that
  using this wrapper in multi-agent setup is equivalent to controlling a single
  player. This wrapper is used for scenarios with control_all_players set when
  agent controls only one player on the team. It can also be used
  in a standalone manner:

  env = gfootball.env.create_environment(env_name='tests/multiagent_wrapper',
      number_of_left_players_agent_controls=11)
  observations = env.reset()
  single_observation = MultiAgentToSingleAgent.get_observation(observations)
  single_action = agent.run(single_observation)
  actions = MultiAgentToSingleAgent.get_action(single_action, observations)
  env.step(actions)
  """

  def __init__(self, env, left_players, right_players):
    assert left_players < 2
    assert right_players < 2
    players = left_players + right_players
    gym.Wrapper.__init__(self, env)
    self._observation = None
    if players > 1:
      self.action_space = gym.spaces.MultiDiscrete([env._num_actions] * players)
    else:
      self.action_space = gym.spaces.Discrete(env._num_actions)

  def reset(self):
    self._observation = self.env.reset()
    return self._get_observation()

  def step(self, action):
    assert self._observation, 'Reset must be called before step'
    action = MultiAgentToSingleAgent.get_action(action, self._observation)
    self._observation, reward, done, info = self.env.step(action)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return MultiAgentToSingleAgent.get_observation(self._observation)

  @staticmethod
  def get_observation(observation):
    assert 'designated' in observation[
        0], 'Only raw observations can be converted'
    result = []
    for obs in observation:
      if obs['designated'] == obs['active']:
        result.append(obs)
    return result

  @staticmethod
  def get_action(actions, orginal_observation):
    assert 'designated' in orginal_observation[
        0], 'Only raw observations can be converted'
    result = [football_action_set.action_builtin_ai] * len(orginal_observation)
    action_idx = 0
    for idx, obs in enumerate(orginal_observation):
      if obs['designated'] == obs['active']:
        assert action_idx < len(actions)
        result[idx] = actions[action_idx]
        action_idx += 1
    return result
