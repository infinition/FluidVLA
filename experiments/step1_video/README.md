# Step 1 -- Video Prediction

Goal: validate spatio-temporal diffusion, memory scaling as a function of frame count,
and adaptive compute at inference.

Entry point:

- `train_step1_video.py`

Base command:

```bash
python experiments/step1_video/train_step1_video.py --epochs 30 --d_model 128
```

Calibration presets:

```bash
python experiments/step1_video/train_step1_video.py --workers 0 --batch_size 8 --epochs 1 --seq_len 6 --d_model 32 --n_layers 2 --max_train_samples 512 --max_test_samples 128 --epsilon 0.08 --min_steps 3 --stop_patience 2
python experiments/step1_video/train_step1_video.py --workers 0 --batch_size 8 --epochs 1 --seq_len 6 --d_model 32 --n_layers 2 --max_train_samples 512 --max_test_samples 128 --epsilon 0.09 --min_steps 3 --stop_patience 2
```

What this step covers:

- Moving MNIST video benchmark,
- internal step instrumentation,
- `epsilon`, `min_steps`, `stop_patience` calibration,
- VRAM growth measurement with temporal length.

Detailed results are centralized in the root [README.md](../../README.md).
