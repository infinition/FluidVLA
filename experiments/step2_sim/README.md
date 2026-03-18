# Step 2 -- Isaac Sim and Pick & Place Training

This directory covers the integrated vision-action pipeline around the Pick & Place
environment, with demonstration collection and policy training.

Entry points:

- `isaac_env.py`
- `train_step2.py`

Use cases:

- synthetic collection,
- Isaac Sim collection,
- imitation learning on demonstrations,
- latency benchmark and adaptive compute behavior.

Commands:

```bash
python experiments/step2_sim/isaac_env.py --mode synthetic --episodes 1000 --image_size 64 --save_dir ./data/step2_sim
python experiments/step2_sim/train_step2.py --dataset ./data/step2_sim --epochs 50 --batch_size 32 --d_model 128 --save_dir ./checkpoints/step2_sim
python experiments/step2_sim/train_step2.py --dataset ./data/step2_sim --epochs 50 --eq_weight 0.05 --save_dir ./checkpoints/step2_sim
python experiments/step2_sim/isaac_env.py --mode collect --episodes 1000 --save_dir ./data/step2c_isaac
python experiments/step2_sim/train_step2.py --dataset ./data/step2c_isaac --epochs 100 --eq_weight 0.1 --checkpoint ./checkpoints/step2_sim/best.pt --save_dir ./checkpoints/step2c_isaac
python experiments/step2_sim/isaac_env.py --mode eval --show_gui --checkpoint ./checkpoints/step2c_isaac/best.pt
```

This step documents usage; consolidated results are in the root [README.md](../../README.md).
