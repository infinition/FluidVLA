# Step 3 -- LeRobot and Real Hardware

Goal: validate FluidVLA on a real SO-101 robotics pipeline using datasets recorded
with LeRobot.

## Recommended Pipeline

1. Record demonstrations with LeRobot.
2. Convert the local LeRobot dataset into `episode_XXXX.npz` files.
3. Train `FluidBotVLA` with the robot's real dimensions.
4. Measure latency and adaptive compute to document edge advantages.

## Main Scripts

- `convert_lerobot_dataset.py`
  - LeRobot -> FluidVLA bridge.
  - Reads a local `data/meta/videos` LeRobot dataset and writes `.npz` episodes
    compatible with FluidVLA training scripts.
- `train_lerobot_so101.py`
  - FluidVLA training on episodes converted from SO-101 demonstrations.
  - Saves `best.pt`, `history.json`, and `benchmark.json`.
- `lerobot_inference.py`
  - Real-time hardware inference prototype (V2).
  - Threaded architecture with camera, inference, and motor control loops.

## Convert a Real LeRobot Dataset

```bash
python experiments/step3_lerobot/convert_lerobot_dataset.py \
  --lerobot-root /path/to/lerobot \
  --repo-id local/so101_balle_bol_test \
  --dataset-root /path/to/recordings/so101_balle_bol_dashboard_01 \
  --output-dir ./data/step3_lerobot/so101_front \
  --camera-key observation.images.front \
  --image-size 224 \
  --n-frames 4
```

Output format per episode:

- `frames`: `(steps, 3, T, H, W)`
- `proprios`: `(steps, 6)` for current SO-101
- `actions`: `(steps, 6)` for current SO-101
- `reward`: `(1,)`, default `1.0` for valid human demos

## Train on SO-101

```bash
python experiments/step3_lerobot/train_lerobot_so101.py \
  --dataset ./data/step3_lerobot/so101_front \
  --save_dir ./checkpoints/step3_lerobot/so101_front \
  --epochs 40 \
  --batch_size 16 \
  --d_model 128 \
  --n_layers 3 \
  --eq_weight 0.02
```

## Benchmark Only

```bash
python experiments/step3_lerobot/train_lerobot_so101.py \
  --dataset ./data/step3_lerobot/so101_front \
  --checkpoint ./checkpoints/step3_lerobot/so101_front/best.pt \
  --benchmark
```

## What This Step Proves

- FluidVLA trains on a real robotics dataset collected with LeRobot.
- The model works with actual SO-101 dimensions: `action_dim=6`, `proprio_dim=6`.
- Latency and adaptive compute are measurable post-training via `benchmark.json`.

Consolidated results and overall project status are in the root [README.md](../../README.md).
