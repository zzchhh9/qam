"""BiGym evaluation bridge.

Runs BiGym evaluation in a subprocess using a separate conda env
(which has mujoco 3.1.5, gymnasium 0.29.2), while the main training
runs in the 'qam' env (JAX/Flax, mujoco 3.8.0).

Usage:
    from envs.bigym_eval_bridge import BiGymBridgeEnv
    env = BiGymBridgeEnv('bigym-reachtargetdual-v0')
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""
import json
import os
import subprocess

import gymnasium
import numpy as np


BIGYM_PYTHON = '/home/zhuzecheng/miniconda3/envs/ds/bin/python'
BIGYM_PATH = '/share_data/zhuzecheng/workspace/diffusionsafeguards/bigym'

# Subprocess server script (runs in BiGym conda env)
EVAL_SCRIPT = r'''
import os, sys, json, warnings
os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore")

# Redirect stdout noise to stderr during imports
_real_stdout = sys.stdout
sys.stdout = sys.stderr

sys.path.insert(0, sys.argv[1])  # BiGym path

import numpy as np
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.utils.observation_config import ObservationConfig

# Monkey-patch for mujoco 3.8.0: joint.qpos returns array, not scalar
import bigym.robots.floating_base as _fb
_orig_pz = _fb.RobotFloatingBase._pelvis_z.fget
def _fix_pz(self):
    try:
        return _orig_pz(self)
    except TypeError:
        if self._position_actuators[2]:
            joint = self._mojo.physics.bind(self._position_actuators[2].joint)
            return float(np.asarray(joint.qpos).flatten()[0])
        else:
            pelvis = self._mojo.physics.bind(self._pelvis.mjcf)
            return float(pelvis.pos[2])
_fb.RobotFloatingBase._pelvis_z = property(_fix_pz)

# Task registry (matches convert_bigym_demos.py)
from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.envs.cupboards import (
    WallCupboardClose, WallCupboardOpen, DrawerTopOpen, DrawerTopClose,
    DrawersAllOpen, DrawersAllClose,
)
from bigym.envs.manipulation import FlipCup, StackBlocks
from bigym.envs.dishwasher import DishwasherOpen, DishwasherClose, DishwasherOpenTrays
from bigym.envs.move_plates import MovePlate
from bigym.envs.pick_and_place import PickBox, PutCups, RemoveSandwich

TASK_REGISTRY = {
    'ReachTargetDual': ReachTargetDual,
    'ReachTarget': ReachTarget,
    'WallCupboardClose': WallCupboardClose,
    'WallCupboardOpen': WallCupboardOpen,
    'DrawerTopOpen': DrawerTopOpen,
    'DrawerTopClose': DrawerTopClose,
    'FlipCup': FlipCup,
    'StackBlocks': StackBlocks,
    'DishwasherOpen': DishwasherOpen,
    'DishwasherClose': DishwasherClose,
    'DishwasherOpenTrays': DishwasherOpenTrays,
    'MovePlate': MovePlate,
    'PickBox': PickBox,
    'PutCups': PutCups,
    'DrawersAllOpen': DrawersAllOpen,
    'DrawersAllClose': DrawersAllClose,
    'RemoveSandwich': RemoveSandwich,
}

# Generic auto-registration: add every BiGym manipulation task class (has _success).
try:
    import bigym.envs.pick_and_place as _m_pnp
    import bigym.envs.manipulation as _m_manip
    import bigym.envs.move_plates as _m_mp
    import bigym.envs.dishwasher as _m_dish
    import bigym.envs.cupboards as _m_cup
    for _m in (_m_pnp, _m_manip, _m_mp, _m_dish, _m_cup):
        for _k, _v in vars(_m).items():
            if isinstance(_v, type) and not _k.startswith('_') and hasattr(_v, '_success') and _k not in TASK_REGISTRY:
                TASK_REGISTRY[_k] = _v
except Exception:
    pass

OBS_KEYS_ORDER = [
    'proprioception',
    'proprioception_grippers',
    'proprioception_floating_base',
    'proprioception_floating_base_actions',
]


def flatten_obs(obs_dict, env=None):
    parts = []
    for key in OBS_KEYS_ORDER:
        if key in obs_dict:
            parts.append(np.asarray(obs_dict[key], dtype=np.float32).flatten())
    priv_keys = sorted(k for k in obs_dict if k not in OBS_KEYS_ORDER)
    for key in priv_keys:
        val = obs_dict[key]
        if isinstance(val, np.ndarray) and val.dtype in (np.float32, np.float64):
            parts.append(np.asarray(val, dtype=np.float32).flatten())
    # Append object state if available (box position + quaternion for PickBox etc.)
    if env is not None:
        obj_state = get_object_state(env)
        if len(obj_state) > 0:
            parts.append(obj_state)
    return np.concatenate(parts)


def get_object_state(env):
    """Extract manipulable object state (pos + quat) from BiGym env."""
    parts = []
    # PickBox / StoreBox: self.box
    if hasattr(env, 'box') and hasattr(env.box, 'body'):
        pos = np.asarray(env.box.body.get_position(), dtype=np.float32)
        quat = np.asarray(env.box.body.get_quaternion(), dtype=np.float32)
        parts.extend([pos, quat])
    # RemoveSandwich / FlipSandwich: self.sandwich
    if hasattr(env, 'sandwich') and hasattr(env.sandwich, 'body'):
        pos = np.asarray(env.sandwich.body.get_position(), dtype=np.float32)
        quat = np.asarray(env.sandwich.body.get_quaternion(), dtype=np.float32)
        parts.extend([pos, quat])
    # DishwasherOpen/Close: self.dishwasher — [door, bottom_tray, middle_tray] task target.
    # Gated by BIGYM_DW_STATE=1 so proprio-only tasks keep their original obs dim.
    import os as _os
    if _os.environ.get('BIGYM_DW_STATE', '0') == '1' and hasattr(env, 'dishwasher'):
        try:
            dw = np.asarray(env.dishwasher.get_state(), dtype=np.float32).ravel()
            parts.append(dw)
        except Exception:
            pass
    if parts:
        return np.concatenate(parts)
    return np.array([], dtype=np.float32)


def denormalize_action(action_norm, low, high):
    """Map action from [-1, 1] to [low, high]."""
    return (action_norm + 1.0) / 2.0 * (high - low) + low


# Parse config from argv
config = json.loads(sys.argv[2])
task_name = config['task']
freq = config['frequency']
floating_dofs_str = config['floating_dofs']

# Build floating DOFs
dof_map = {'pelvis_x': PelvisDof.X, 'pelvis_y': PelvisDof.Y,
           'pelvis_z': PelvisDof.Z, 'pelvis_rz': PelvisDof.RZ}
floating_dofs = [dof_map[d] for d in floating_dofs_str]

# Create env
action_mode = JointPositionActionMode(
    floating_base=True,
    floating_dofs=floating_dofs,
    absolute=True,
)
use_cameras = config.get('use_cameras', False)
obs_config = ObservationConfig(proprioception=True, privileged_information=True)
env_cls = TASK_REGISTRY[task_name]
env = env_cls(
    action_mode=action_mode,
    observation_config=obs_config,
    control_frequency=freq,
)
import base64

cam_renderer = None
if use_cameras:
    import mujoco
    model = env.unwrapped._mojo.physics.model._model
    data = env.unwrapped._mojo.physics.data._data
    cam_res = 84
    renderer = mujoco.Renderer(model, height=cam_res, width=cam_res)
    cam_names = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name and ('wrist' in name.lower() or 'hand' in name.lower()):
            cam_names.append(name)
    if len(cam_names) < 2:
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name and name not in cam_names:
                cam_names.append(name)
            if len(cam_names) >= 2:
                break
    sys.stderr.write(f"Cameras found: {cam_names}\n")
    sys.stderr.flush()

    def render_cameras():
        imgs = {}
        for ci, cname in enumerate(cam_names[:2]):
            renderer.update_scene(data, camera=cname)
            img = renderer.render()
            key = 'rgb_left_wrist' if ci == 0 else 'rgb_right_wrist'
            imgs[key] = base64.b64encode(img.tobytes()).decode('ascii')
            imgs[key + '_shape'] = list(img.shape)
        return imgs
else:
    def render_cameras():
        return {}

# Action denormalization: use data bounds if provided, else env bounds
if config.get('act_min') is not None:
    act_low = np.array(config['act_min'], dtype=np.float32)
    act_high = np.array(config['act_max'], dtype=np.float32)
else:
    act_low = env.action_space.low.copy()
    act_high = env.action_space.high.copy()

# Report spaces
obs_dict, _ = env.reset()
obs_flat = flatten_obs(obs_dict, env)
obs_dim = obs_flat.shape[0]
act_dim = env.action_space.shape[0]

# Restore stdout for protocol
sys.stdout = _real_stdout

# Obs normalization bounds (from config, if provided)
obs_norm_min = np.array(config.get('obs_min', []), dtype=np.float32) if config.get('obs_min') else None
obs_norm_max = np.array(config.get('obs_max', []), dtype=np.float32) if config.get('obs_max') else None

def normalize_obs(obs_flat):
    if obs_norm_min is not None and len(obs_norm_min) == len(obs_flat):
        obs_range = np.maximum(obs_norm_max - obs_norm_min, 1e-6)
        return np.clip(2.0 * (obs_flat - obs_norm_min) / obs_range - 1.0, -1, 1).astype(np.float32)
    return obs_flat

# Send initial spaces info
print(json.dumps({
    "type": "init",
    "obs_dim": obs_dim,
    "act_dim": act_dim,
}), flush=True)

# Protocol: JSON commands over stdin, JSON responses over stdout
for line in sys.stdin:
    cmd = json.loads(line.strip())

    if cmd["type"] == "reset":
        obs_dict, info = env.reset()
        obs_flat = normalize_obs(flatten_obs(obs_dict, env))
        _ts = float(info.get("task_success", 0.0))
        safe_info = {"task_success": _ts, "success": _ts}
        if use_cameras:
            safe_info.update(render_cameras())
        print(json.dumps({
            "obs": obs_flat.tolist(),
            "info": safe_info,
        }), flush=True)

    elif cmd["type"] == "step":
        action_norm = np.array(cmd["action"], dtype=np.float32)
        action_raw = denormalize_action(action_norm, act_low, act_high)
        action_raw = np.clip(action_raw, env.action_space.low, env.action_space.high)
        obs_dict, reward, terminated, truncated, info = env.step(action_raw)
        obs_flat = normalize_obs(flatten_obs(obs_dict, env))
        _ts = float(info.get("task_success", 0.0))
        safe_info = {"task_success": _ts, "success": _ts}
        if use_cameras:
            safe_info.update(render_cameras())
        print(json.dumps({
            "obs": obs_flat.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": safe_info,
        }), flush=True)

    elif cmd["type"] == "render":
        res = cmd.get("resolution", 480)
        try:
            import mujoco
            m = env.unwrapped._mojo.physics.model._model
            d = env.unwrapped._mojo.physics.data._data
            r = mujoco.Renderer(m, height=res, width=res)
            r.update_scene(d)
            frame = r.render()
            r.close()
            frame_b64 = base64.b64encode(frame.tobytes()).decode('ascii')
            print(json.dumps({
                "frame": frame_b64,
                "shape": list(frame.shape),
            }), flush=True)
        except Exception as e:
            print(json.dumps({"frame": None, "error": str(e)}), flush=True)

    elif cmd["type"] == "close":
        env.close()
        break
'''


# -- Task config (must match convert_bigym_demos.py) --
BIGYM_TASKS = {
    'bigym-reachtargetdual-v0': {
        'task': 'ReachTargetDual',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_rz'],
        'action_dim': 15,
    },
    'bigym-reachtarget-v0': {
        'task': 'ReachTarget',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_rz'],
        'action_dim': 15,
    },
    'bigym-wallcupboardclose-v0': {
        'task': 'WallCupboardClose',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-wallcupboardopen-v0': {
        'task': 'WallCupboardOpen',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-drawertopopen-v0': {
        'task': 'DrawerTopOpen',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-flipcup-v0': {
        'task': 'FlipCup',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-stackblocks-v0': {
        'task': 'StackBlocks',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-dishwasheropen-v0': {
        'task': 'DishwasherOpen',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-dishwasherclose-v0': {
        'task': 'DishwasherClose',
        'frequency': 50,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-dishwasheropentrays-v0': {
        'task': 'DishwasherOpenTrays',
        'frequency': 50,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-putcups-v0': {
        'task': 'PutCups',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-moveplate-v0': {
        'task': 'MovePlate',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_rz'],
        'action_dim': 15,
    },
    'bigym-pickbox-v0': {
        'task': 'PickBox',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-drawersallopen-v0': {
        'task': 'DrawersAllOpen',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-drawersallclose-v0': {
        'task': 'DrawersAllClose',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
    'bigym-removesandwich-v0': {
        'task': 'RemoveSandwich',
        'frequency': 25,
        'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
        'action_dim': 16,
    },
}

# Generic auto-fill: register the full BiGym manipulation task set (50Hz / 16D default).
_SWEEP_TASKS = [
    'DishwasherCloseTrays', 'DrawerTopClose', 'DrawerTopOpen', 'DrawersAllOpen', 'WallCupboardOpen',
    'PutCups', 'TakeCups', 'StoreBox', 'PickBox', 'SaucepanToHob', 'StoreKitchenware', 'ToastSandwich',
    'FlipSandwich', 'FlipCup', 'FlipCutlery', 'StackBlocks', 'MovePlate', 'MoveTwoPlates',
    'CupboardsOpenAll', 'CupboardsCloseAll',
]
for _task in _SWEEP_TASKS:
    _en = 'bigym-' + _task.lower() + '-v0'
    if _en not in BIGYM_TASKS:
        BIGYM_TASKS[_en] = {
            'task': _task, 'frequency': 50,
            'floating_dofs': ['pelvis_x', 'pelvis_y', 'pelvis_z', 'pelvis_rz'],
            'action_dim': 16,
        }


def _decode_cameras(info):
    """Decode base64 camera images from subprocess info dict."""
    import base64
    decoded = {}
    for k in list(info.keys()):
        if k.endswith('_shape'):
            continue
        shape_key = k + '_shape'
        if shape_key in info and 'rgb' in k:
            shape = info[shape_key]
            raw = base64.b64decode(info[k])
            img = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
            decoded[k] = img
            del info[k]
            del info[shape_key]
    return decoded


class BiGymBridgeEnv(gymnasium.Env):
    """Gymnasium-compatible wrapper that runs BiGym in a subprocess."""

    def __init__(self, env_name, max_episode_steps=500, obs_min=None, obs_max=None, act_min=None, act_max=None, frequency=None, use_cameras=False):
        super().__init__()
        if env_name not in BIGYM_TASKS:
            raise ValueError(
                f"Unknown BiGym env: {env_name}. "
                f"Available: {list(BIGYM_TASKS.keys())}"
            )
        self.env_name = env_name
        self._task_config = dict(BIGYM_TASKS[env_name])
        if frequency is not None:
            self._task_config['frequency'] = frequency
        if obs_min is not None:
            self._task_config['obs_min'] = obs_min
        if obs_max is not None:
            self._task_config['obs_max'] = obs_max
        if act_min is not None:
            self._task_config['act_min'] = act_min
        if act_max is not None:
            self._task_config['act_max'] = act_max
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._use_cameras = use_cameras
        if use_cameras:
            self._task_config['use_cameras'] = True

        # Will be set after subprocess init
        self._obs_dim = None
        self._act_dim = self._task_config['action_dim']
        self._proc = None

        # Start subprocess and get actual dimensions
        self._start_process()

        # Set spaces using actual dims from subprocess
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,), dtype=np.float32,
        )

    def _start_process(self):
        """Start the BiGym subprocess."""
        env = os.environ.copy()
        env['MUJOCO_GL'] = 'egl'

        config_json = json.dumps(self._task_config)
        self._proc = subprocess.Popen(
            [BIGYM_PYTHON, '-c', EVAL_SCRIPT, BIGYM_PATH, config_json],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        # Read init message with space dimensions
        init_line = self._proc.stdout.readline()
        if not init_line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(
                f'BiGym subprocess failed to start: {stderr[-1000:]}'
            )
        init_msg = json.loads(init_line)
        self._obs_dim = init_msg['obs_dim']
        self._act_dim = init_msg['act_dim']

    def _send(self, cmd):
        """Send command and receive response."""
        self._proc.stdin.write(json.dumps(cmd) + '\n')
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr = self._proc.stderr.read()
            raise RuntimeError(f'BiGym subprocess died: {stderr[-1000:]}')
        return json.loads(line)

    def reset(self, **kwargs):
        if self._proc is None or self._proc.poll() is not None:
            self._start_process()
        resp = self._send({"type": "reset"})
        obs = np.array(resp["obs"], dtype=np.float32)
        info = resp.get("info", {})
        if self._use_cameras:
            info.update(_decode_cameras(info))
        self._step_count = 0
        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        resp = self._send({"type": "step", "action": action.tolist()})
        obs = np.array(resp["obs"], dtype=np.float32)
        reward = resp["reward"]
        terminated = resp["terminated"]
        truncated = resp["truncated"]
        info = resp.get("info", {})
        if self._use_cameras:
            info.update(_decode_cameras(info))

        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self, resolution=480):
        """Render MuJoCo sim view via subprocess."""
        resp = self._send({"type": "render", "resolution": resolution})
        if resp.get("frame") is None:
            return None
        import base64
        raw = base64.b64decode(resp["frame"])
        shape = resp["shape"]
        return np.frombuffer(raw, dtype=np.uint8).reshape(shape)

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"type": "close"})
            except Exception:
                pass
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()
