import collections
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif env_name.startswith('bigym-'):
        # BiGym (copied from sibling fqc/ project, dataset from HDF5)
        import os as _os, h5py
        from envs import bigym_utils as _bg
        _freq = int(_os.environ.get('BIGYM_FREQ', '50'))  # must match the dataset freq (25 or 50)
        env = _bg.make_env(env_name, frequency=_freq, use_cameras=False)
        eval_env = _bg.make_env(env_name, frequency=_freq, use_cameras=False)
        env = EpisodeMonitor(env)
        eval_env = EpisodeMonitor(eval_env)
        hdf5_path = _os.environ.get(
            'BIGYM_HDF5',
            '/share_data/zhuzecheng/workspace/psi-post-rl/fqc/data/bigym/vlm_critic_train_v2.hdf5',
        )
        print(f'Loading BiGym dataset: {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            obs = f['observations'][:].astype(np.float32)
            act = f['actions'][:].astype(np.float32)
            rew = f['rewards'][:].astype(np.float32)
            term = f['terminals'][:].astype(np.float32)
            masks = f['masks'][:].astype(np.float32) if 'masks' in f else (1.0 - term).astype(np.float32)
            next_obs = f['next_observations'][:].astype(np.float32) if 'next_observations' in f else None
        if next_obs is None:
            # Build from observations shifted by 1 (terminal frames use same obs)
            next_obs = np.concatenate([obs[1:], obs[-1:][None][0]])
        print(f'  Loaded {len(obs)} transitions, action_dim={act.shape[-1]}, obs_dim={obs.shape[-1]}')
        dataset = dict(observations=obs, actions=act, rewards=rew,
                       terminals=term, masks=masks, next_observations=next_obs)
        train_dataset = Dataset.create(**dataset)
        val_dataset = None
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils
        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit. Must come AFTER bigym branch because 'pen' matches 'open' in bigym task names.
        import d4rl.hand_manipulation_suite  # noqa
        from envs import d4rl_utils
        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
