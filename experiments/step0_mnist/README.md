# Step 0 -- Image Classification

Goal: validate that the PDE core learns useful visual features without attention.

Entry point:

- `train_step0.py`

Target datasets:

- MNIST
- CIFAR-10

Commands:

```bash
python experiments/step0_mnist/train_step0.py --dataset mnist --model tiny
python experiments/step0_mnist/train_step0.py --dataset mnist --model small
python experiments/step0_mnist/train_step0.py --dataset cifar10 --model small
python experiments/step0_mnist/train_step0.py --dataset cifar10 --model small --batch_size 512
```

What this step validates:

- convergence,
- training stability,
- learning useful features without attention,
- spatial core soundness before moving to video.

Consolidated results and interpretation are in the root [README.md](../../README.md).
