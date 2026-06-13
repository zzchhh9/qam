"""BiGym environment utilities for FQC.

Provides:
- make_env(): Creates BiGymBridgeEnv wrapped with EpisodeMonitor
- get_dataset(): Loads pre-converted HDF5 dataset from data/bigym/

Pre-requisite: Run scripts/convert_bigym_demos.py in BiGym conda env first.
"""
import os

import h5py
import numpy as np

from envs.bigym_eval_bridge import BiGymBridgeEnv, BIGYM_TASKS
from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


# Default data directory
DATA_DIR = os.path.join('/share_data/zhuzecheng/workspace/psi-post-rl/fqc/data/bigym')

# Map env_name to HDF5 task name
ENV_TO_TASK = {
    'bigym-reachtargetdual-v0': 'ReachTargetDual',
    'bigym-reachtarget-v0': 'ReachTarget',
    'bigym-wallcupboardclose-v0': 'WallCupboardClose',
    'bigym-wallcupboardopen-v0': 'WallCupboardOpen',
    'bigym-drawertopopen-v0': 'DrawerTopOpen',
    'bigym-flipcup-v0': 'FlipCup',
    'bigym-stackblocks-v0': 'StackBlocks',
    'bigym-dishwasheropen-v0': 'DishwasherOpen',
    'bigym-putcups-v0': 'PutCups',
    'bigym-moveplate-v0': 'MovePlate',
    'bigym-pickbox-v0': 'PickBox',
    'bigym-drawersallopen-v0': 'DrawersAllOpen',
    'bigym-drawersallclose-v0': 'DrawersAllClose',
    'bigym-removesandwich-v0': 'RemoveSandwich',
}


_LONG_EPISODE_TASKS = {'bigym-dishwasheropen-v0', 'bigym-dishwasherclose-v0',
                        'bigym-pickbox-v0', 'bigym-putcups-v0', 'bigym-stackblocks-v0',
                        'bigym-drawersallopen-v0', 'bigym-drawersallclose-v0',
                        'bigym-removesandwich-v0'}

def make_env(env_name, max_episode_steps=500, frequency=25, use_cameras=False):
    """Create BiGym bridge environment wrapped with EpisodeMonitor.

    Args:
        env_name: BiGym environment name (e.g., 'bigym-reachtargetdual-v0').
        max_episode_steps: Maximum steps per episode before truncation.
        frequency: Control frequency (used to find obs normalization bounds).

    Returns:
        Gymnasium environment with flat observation and [-1, 1] action spaces.
    """
    if env_name in _LONG_EPISODE_TASKS:
        max_episode_steps = max(max_episode_steps, 2000)

    # Load normalization bounds from HDF5 (for obs and action denormalization)
    obs_min = obs_max = act_min = act_max = None
    task_name = ENV_TO_TASK.get(env_name)
    if task_name:
        vision_path = os.path.join(DATA_DIR, f'{task_name}_{frequency}hz_vision_demo_only.hdf5')
        base_path = os.path.join(DATA_DIR, f'{task_name}_{frequency}hz.hdf5')
        hdf5_path = vision_path if os.path.exists(vision_path) else base_path
        if os.path.exists(hdf5_path):
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    if 'obs_min' in f.attrs and 'obs_max' in f.attrs:
                        obs_min = f.attrs['obs_min'].tolist()
                        obs_max = f.attrs['obs_max'].tolist()
                    if 'act_min' in f.attrs and 'act_max' in f.attrs:
                        act_min = f.attrs['act_min'].tolist()
                        act_max = f.attrs['act_max'].tolist()
            except Exception:
                pass

    env = BiGymBridgeEnv(env_name, max_episode_steps=max_episode_steps,
                         obs_min=obs_min, obs_max=obs_max,
                         act_min=act_min, act_max=act_max,
                         frequency=frequency, use_cameras=use_cameras)
    env = EpisodeMonitor(env)
    return env


def get_dataset(env, env_name, normalize_reward=False, frequency=25, data_dir=None):
    """Load pre-converted BiGym HDF5 dataset.

    Args:
        env: Environment instance (used for verification).
        env_name: BiGym environment name.
        normalize_reward: If True, normalize rewards (subtract mean, divide by std).
        frequency: Control frequency used during conversion (default 25Hz).
        data_dir: Override data directory.

    Returns:
        Dataset with observations, actions, next_observations, terminals, rewards, masks.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    task_name = ENV_TO_TASK.get(env_name)
    if task_name is None:
        raise ValueError(
            f"Unknown BiGym env: {env_name}. Available: {list(ENV_TO_TASK.keys())}"
        )

    hdf5_path = os.path.join(data_dir, f'{task_name}_{frequency}hz.hdf5')
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(
            f"Dataset not found: {hdf5_path}\n"
            f"Run: conda activate bigym && cd fqc && "
            f"MUJOCO_GL=egl python scripts/convert_bigym_demos.py "
            f"--task {task_name} --freq {frequency}"
        )

    # Load HDF5
    with h5py.File(hdf5_path, 'r') as f:
        observations = f['observations'][:].astype(np.float32)
        actions = f['actions'][:].astype(np.float32)
        rewards = f['rewards'][:].astype(np.float32)
        terminals = f['terminals'][:].astype(np.float32)
        next_observations = f['next_observations'][:].astype(np.float32)
        masks = f['masks'][:].astype(np.float32)

        # Read metadata
        obs_dim = int(f.attrs['obs_dim'])
        act_dim = int(f.attrs['act_dim'])
        num_episodes = int(f.attrs['num_episodes'])
        num_transitions = int(f.attrs['num_transitions'])

    # Verify dimensions match the environment
    env_obs_dim = env.observation_space.shape[0]
    env_act_dim = env.action_space.shape[0]
    if obs_dim != env_obs_dim:
        print(f"WARNING: Dataset obs_dim={obs_dim} != env obs_dim={env_obs_dim}. "
              f"Using dataset dimensions.")
    if act_dim != env_act_dim:
        print(f"WARNING: Dataset act_dim={act_dim} != env act_dim={env_act_dim}. "
              f"Using dataset dimensions.")

    # Normalize rewards if requested
    reward_mean = rewards.mean()
    reward_std = rewards.std()
    if normalize_reward and reward_std > 0:
        rewards = (rewards - reward_mean) / reward_std
        print(f"Loaded {env_name}: {num_transitions} transitions, {num_episodes} episodes, "
              f"reward: mean={reward_mean:.4f} std={reward_std:.4f} -> normalized")
    else:
        print(f"Loaded {env_name}: {num_transitions} transitions, {num_episodes} episodes, "
              f"reward: mean={reward_mean:.4f} std={reward_std:.4f}")

    return Dataset.create(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        terminals=terminals,
        rewards=rewards,
        masks=masks,
        freeze=False,
    )
