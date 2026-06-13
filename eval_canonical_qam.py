"""Standalone eval for canonical QAM checkpoints.

Loads a saved QAMAgent params, evaluates N episodes on BiGym, writes a result txt.
Usage:
    CUDA_VISIBLE_DEVICES=3 python eval_canonical_qam.py \
        <ckpt.pkl> <num_episodes> <out_txt>
"""
import os, sys, pickle, time
import numpy as np
import jax, jax.numpy as jnp, flax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ckpt_path = sys.argv[1]
num_eps = int(sys.argv[2])
out_txt = sys.argv[3]

# Match training config
HORIZON = 8
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 10001

ENV_NAME = os.environ.get('BIGYM_ENV', 'bigym-removesandwich-v0')
from envs.env_utils import make_env_and_datasets
env, eval_env, train_dataset, val_dataset = make_env_and_datasets(ENV_NAME)

from agents.qam import QAMAgent, get_config
config = get_config()
config['horizon_length'] = HORIZON
config['action_chunking'] = True
config['inv_temp'] = 0.3
config['flow_steps'] = 10
config['discount'] = 0.99
config['batch_size'] = 256
config['num_qs'] = 10

# Match main.py: example_batch = train_dataset.sample(()) → unbatched shapes
example_batch = train_dataset.sample(())
ex_obs = example_batch['observations']
ex_act = example_batch['actions']
print(f'ex_obs={ex_obs.shape} ex_act={ex_act.shape}')

agent = QAMAgent.create(SEED, ex_obs, ex_act, config)

with open(ckpt_path, 'rb') as f:
    save_dict = pickle.load(f)
try:
    agent = flax.serialization.from_state_dict(agent, save_dict['agent'])
except Exception:
    # inference-only checkpoint (optimizer state stripped, see checkpoints/) -> load params only
    sd = save_dict['agent']
    net_params = flax.serialization.from_state_dict(agent.network.params, sd['network']['params'])
    agent = agent.replace(network=agent.network.replace(params=net_params))
print(f'Loaded {ckpt_path}')

action_dim = env.action_space.shape[0]
rng = jax.random.PRNGKey(SEED)

n_success = 0
n_done = 0
ep_lens = []
for ep in range(num_eps):
    obs, info = env.reset()
    action_queue = []
    done = False
    step = 0
    while not done:
        if len(action_queue) == 0:
            rng, key = jax.random.split(rng)
            a = np.array(agent.sample_actions(observations=obs, rng=key))
            a = a.reshape(-1, action_dim)
            for ai in a:
                action_queue.append(ai)
        action = np.clip(action_queue.pop(0), -1, 1)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
    success = info.get('success', False) or info.get('task_success', False) or (info.get('episode', {}).get('return', 0) > 0.5)
    n_success += int(success)
    n_done += 1
    ep_lens.append(step)
    print(f'  Ep {ep}: {"SUCCESS" if success else "FAIL"} steps={step}  (running: {n_success}/{n_done} = {n_success*100/n_done:.0f}%)', flush=True)

sr = n_success / num_eps
mean_len = float(np.mean(ep_lens))
os.makedirs(os.path.dirname(out_txt) or '.', exist_ok=True)
with open(out_txt, 'w') as f:
    f.write(f'{n_success}/{num_eps}\n')
print(f'\nFINAL: {n_success}/{num_eps} = {sr*100:.1f}% SR, mean_ep_len={mean_len:.0f}')
print(f'Saved to {out_txt}')
