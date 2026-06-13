# QAM best checkpoints

GitHub blocks Git LFS uploads on **forks** (this repo is a fork of `colinqiyangli/qam`), so the published
QAM checkpoints live in the **`fqc` repo** instead: `fqc/checkpoints/`.

| File (in fqc/checkpoints/) | Task | 50-ep SR |
|----------------------------|------|----------|
| `DrawersAllClose_qam_88.pkl` | DrawersAllClose | **88%** |
| `DishwasherClose_qam_20.pkl` | DishwasherClose | 20% |

They are optimizer-state-stripped (inference-only); `eval_canonical_qam.py` (here) auto-loads them via a
params-only fallback:
```bash
BIGYM_ENV=bigym-drawersallclose-v0 BIGYM_HDF5=/path/to/DrawersAllClose_50hz.hdf5 BIGYM_FREQ=50 \
  PYTHONPATH=. python eval_canonical_qam.py /path/to/fqc/checkpoints/DrawersAllClose_qam_88.pkl 50 out.txt 2000
```
