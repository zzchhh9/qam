"""Standalone eval for QChunking (acfql) checkpoints.
Usage:
    python eval_canonical_acfql.py <ckpt.pkl> <num_episodes> <out_txt> [<horizon> [<seed>]]
"""
import os, sys, pickle
import numpy as np
import jax, jax.numpy as jnp, flax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ckpt_path = sys.argv[1]
num_eps = int(sys.argv[2])
out_txt = sys.argv[3]
HORIZON = int(sys.argv[4]) if len(sys.argv) > 4 else 4
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 10001

ENV_NAME = os.environ.get('BIGYM_ENV', 'bigym-removesandwich-v0')
from envs.env_utils import make_env_and_datasets
env, eval_env, train_dataset, val_dataset = make_env_and_datasets(ENV_NAME)

from agents.acfql import ACFQLAgent, get_config
config = get_config()
config['horizon_length'] = HORIZON
config['action_chunking'] = True
config['flow_steps'] = 10
config['discount'] = 0.99
config['batch_size'] = 256
config['num_qs'] = 2
config['alpha'] = 100.0
config['actor_type'] = 'distill-ddpg'

example_batch = train_dataset.sample(())
ex_obs = example_batch['observations']
ex_act = example_batch['actions']
print(f'ex_obs={ex_obs.shape} ex_act={ex_act.shape} horizon={HORIZON}')

agent = ACFQLAgent.create(SEED, ex_obs, ex_act, config)

with open(ckpt_path, 'rb') as f:
    save_dict = pickle.load(f)
agent = flax.serialization.from_state_dict(agent, save_dict['agent'])
print(f'Loaded {ckpt_path}')

action_dim = env.action_space.shape[0]
rng = jax.random.PRNGKey(SEED)

n_success = 0
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
    ep_lens.append(step)
    print(f'  Ep {ep}: {"SUCCESS" if success else "FAIL"} steps={step}  (running: {n_success}/{ep+1} = {n_success*100//(ep+1)}%)', flush=True)

os.makedirs(os.path.dirname(out_txt) or '.', exist_ok=True)
with open(out_txt, 'w') as f:
    f.write(f'{n_success}/{num_eps}\n')
print(f'\nFINAL: {n_success}/{num_eps} = {n_success*100/num_eps:.1f}% SR, mean_len={np.mean(ep_lens):.0f}')
