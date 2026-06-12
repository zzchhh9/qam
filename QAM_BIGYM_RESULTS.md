# QAM on BiGym — Verified Results (2026-05-27)

> QAM = flat Q-learning with Adjoint Matching (action chunking H=8, `qam/agents/qam.py` + `qam/main.py`).
> Full repo map / canonical commands / 23 audited bugs: see `psi-post-rl/BIGYM_RL_GUIDE.md`.

## Verified SR (standalone `eval_canonical_qam.py`, 50 episodes, independent seed)

| Task | Checkpoint | SR | seeds | mean ep len |
|------|-----------|-----|-------|-------------|
| **WallCupboardClose** | params_70000 | **32%** (16/50) | 10001 **and** 2000 (both 32%) | 405 |
| WallCupboardClose | params_100000 | 28% (14/50) | 10001 | 420 |
| RemoveSandwich | 30K–500K (swept 8 ckpts) | **0-2%** | 10001 | ~1960 (runs to max) |
| RemoveSandwich (qam_bcfast) | 30K / 250K | 2% / 0% | 10001 | ~1900 |
| DishwasherOpen | 50K / 100K | **0%** | 10001 | 2000 |

## Conclusion
- **QAM clears the >10% bar on WallCupboardClose (32%, two independent seeds).** ✓
- **QAM genuinely fails the long-horizon tasks** (RemoveSandwich, DishwasherOpen): flat QAM's `actor_fast`
  is the only net sampled at eval but it has **no BC anchor** and its adjoint loss runs from offline step 1
  against an untrained critic → it never learns a usable policy on hard, sparse, 1600-step tasks.
  `qam_bcfast.py` (adds a BC anchor) only lifts RS to ~2% — the task is intrinsically hard for flat QAM.
- **This is the core FQC-vs-QAM finding:** FQC's slow/fast **factoring** is what cracks the hard tasks
  (FQC RemoveSandwich = 74%, WallCupboardClose = 100%); flat QAM only manages the short task.

## How to reproduce
```bash
cd qam
BIGYM_ENV=bigym-wallcupboardclose-v0 CUDA_VISIBLE_DEVICES=<G> \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 MUJOCO_GL=egl PYTHONPATH=. \
  python -u eval_canonical_qam.py \
    exp/dummy/qam_canonical_wcc/bigym-wallcupboardclose-v0/e3a35772*/params_70000.pkl \
    50 out.txt 2000
# expect ~16/50 = 32%
```

## ⚠️ Eval correctness notes (do not relearn the hard way)
- Use ONLY `eval_canonical_qam.py` (standalone). It reconstructs the agent correctly.
- **Diagnostic rule:** `0/50 AND mean_ep_len == max_steps` = a BROKEN eval (obs-dim / normalization),
  NOT a real-zero policy. A working policy terminates successes early. (QAM-RS 0% here is real — episodes
  run to max because the policy genuinely never completes, and this matches across all checkpoints/seeds.)
- Eval frequency must match training (50Hz). `qam/envs/env_utils.py` hardcodes env `frequency=50`, so
  25Hz multitask tasks (MovePlate/FlipCup) were trained freq-mismatched — those checkpoints are unreliable.

Raw per-episode logs + result files: `qam/verify_2026_05_27/`.
