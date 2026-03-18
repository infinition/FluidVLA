# Step 2a -- Synthetic Pick & Place

This directory isolates the pure synthetic version of the imitation learning pipeline.

Entry points:

- `synthetic_env.py`
- `train_step2a.py`

Purpose:

- validate the full stack before Isaac Sim,
- produce homogeneous demos for simple training,
- serve as a clean starting point before more physical collection.

Commands:

```bash
python experiments/step2a_synthetic/synthetic_env.py --episodes 1000
python experiments/step2a_synthetic/train_step2a.py --dataset ./data/step2a_synthetic --epochs 50 --save_dir ./checkpoints/step2a_synthetic
python experiments/step2a_synthetic/train_step2a.py --dataset ./data/step2a_synthetic --epochs 50 --no_pde --save_dir ./checkpoints/step2a_synthetic
```

Fine-tuning toward more physical setups:

```bash
python experiments/step2a_synthetic/train_step2a.py --dataset ./data/step2a_synthetic --epochs 50 --checkpoint ./checkpoints/step2a_synthetic/best.pt
```

Consolidated metrics are centralized in the root [README.md](../../README.md).
