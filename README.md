# FluidVLA

![Status](https://img.shields.io/badge/status-research_prototype-0b6b8a)
![Architecture](https://img.shields.io/badge/architecture-transformer--free-10b981)
![Core](https://img.shields.io/badge/core-reaction--diffusion_PDE-2563eb)
![Focus](https://img.shields.io/badge/focus-edge_robotics-f59e0b)
![Language](https://img.shields.io/badge/code-python-3776ab)

![FluidVLA platform](https://github.com/user-attachments/assets/92e1bcc1-6ab2-44e1-bc60-60e7b2d97669)

**A Transformer-free Vision-Language-Action model for real-time edge robotics.**

FluidVLA is a research program that replaces quadratic self-attention with a local reaction-diffusion dynamic for vision, video, control, and ultimately embedded Vision-Language-Action systems.

The goal is not to propose a slightly lighter Transformer. The goal is to change the computational primitive itself to achieve better compatibility with continuous video, bounded memory, adaptive compute, and edge robotics.

---

## Empirical Snapshot

### 1. Imitation Learning (VLA End-to-End)

Validation on a synthetic 7-DOF pick-and-place:

- **Best Val MSE:** **0.01345**
- **Latency:** **~4.1 ms** per step
- **Effective FPS:** **~244 Hz**
- **Adaptive compute:** **1 / 12 steps** post-training in the reported regime

### 2. Spatial Memory Scaling

| Resolution | FluidVLA | ViT estimated |
| --- | ---: | ---: |
| 128x128 | 59.2 MB | ~900 MB |
| 256x256 | 202.9 MB | ~14,000 MB |

### 3. Video Scaling

| Frames | FluidVLA | Transformer estimated |
| --- | ---: | ---: |
| 8 | 36.9 MB | ~272 MB |
| 16 | 62.9 MB | ~1,088 MB |
| 32 | 114.6 MB | ~4,352 MB |

### 4. Zero-Attention Classification Baseline

| Dataset | Accuracy | Params |
| --- | ---: | ---: |
| MNIST | 91.75% | ~0.30M |
| CIFAR-10 | 62.25% | ~0.31M |

### 5. Medical 3D Historical Validation

| Config | Val Dice | Latency | VRAM |
| --- | ---: | ---: | ---: |
| PDE OFF | 0.9129 | ~60 ms | ~632 MiB |
| PDE ON | 0.9177 | ~89 ms | ~632 MiB |

### 6. Real LeRobot SO-101 Progress Snapshot

Measured state as of March 10, 2026 on the real dataset `local/so101_balle_bol_test`:

- **Dataset:** **44 episodes / 18,081 samples / 2 cameras / 15 fps**
- **Bridge validated:** LeRobot -> FluidVLA `.npz` conversion
- **Smoke training:** **best.pt** generated with `action_dim=6`, `proprio_dim=6`
- **Full run front-only V1:** **Val MSE 16.69330** at epoch 10 -- converged but produces a fixed point at live deployment
- **Live diagnostic V1:** frozen policy, constant raw deltas regardless of image (confirmed with `filter_alpha=1.0`)
- **V2 pipeline implemented:** spatial pool 4x4, normalized delta-actions, cosine loss, optional action chunking
- **V2 re-conversion in progress:** `--delta-actions --filter-static 0.5 --subsample-static 4`
- **Measured delta statistics:** `mean=[0.17, -0.77, -1.98, -0.25, -0.08, -3.55]` / `std=[3.67, 5.07, 4.55, 2.39, 1.68, 8.59]`
- **Static frames (|delta| < 0.5 deg):** **0/18,081 (0.0%)** -- dataset is fully usable as-is
- **V2 training:** pending
- **V2 live test:** pending

---

## The Edge Robotics Dilemma

Current VLAs almost all inherit from Transformers originally designed for NLP. In embedded robotics, this decision quickly hits the KV-cache memory and latency wall.

As spatial resolution and temporal length increase, computational pressure explodes and forces destructive trade-offs: shortened context, video degraded to a few frames, discontinuous perception, inability to properly reason about continuous motion.

The central intuition of the project is simple: a robot that only sees the world through sparse snapshots cannot properly model physics, velocity, or continuous manipulation.

---

## The Solution

**FluidVLA replaces the attention matrix with a reaction-diffusion core with lightweight memory.**

Instead of making all tokens communicate via a dense matrix, information propagates locally in space and time, with a small global memory state and an iterative computation that can stop early when the scene stabilizes.

$$
u_{t+1} = \operatorname{LayerNorm}\left(u_t + dt \cdot \left[\sum_k D_k \cdot \nabla^2(u_t) + R(u_t, \theta) + \alpha \cdot h_t + \alpha_{loc} \cdot m_{loc}\right]\right)
$$

| Term | Role |
| --- | --- |
| $\sum_k D_k \cdot \nabla^2(u_t)$ | multi-scale local diffusion |
| $R(u_t, \theta)$ | nonlinear per-position reaction |
| $h_t$ | lightweight global memory |
| $m_{loc}$ | low-resolution local memory |
| $dt$ | learned integration step |

**Adaptive compute / Turing Equilibrium.**

The model can reduce the number of PDE steps at inference when the scene is simple or stable. In practice, some phases converge in very few iterations while more complex scenes retain more computation steps.

---

## Quick Start

### Install

```bash
git clone https://github.com/infinition/FluidVLA.git
cd FluidVLA
pip install -e .
```

### Verify

```bash
python -m pytest tests/ -v
```

### Train

```bash
python experiments/step0_mnist/train_step0.py --dataset mnist --model small
python experiments/step0_mnist/train_step0.py --dataset cifar10 --model small --batch_size 512
python experiments/step1_video/train_step1_video.py --epochs 30 --d_model 128
python experiments/step2_sim/isaac_env.py --mode synthetic --episodes 1000 --image_size 64
python experiments/step2_sim/train_step2.py --dataset ./data/step2_sim --epochs 50 --batch_size 32 --d_model 128
```

### Web Platform

```bash
python fluidvla_server.py --port 7860
```

---

## Repository Structure

```text
FluidVLA/
├── fluidvla/                   Core package
│   └── core/
│       ├── diffusion.py                Multi-scale Laplacian operators
│       ├── fluid_layer.py              Reaction-diffusion PDE layer
│       ├── fluid_layer3d.py            3D volumetric variant
│       ├── vision_models.py            Image classifier
│       ├── video_models.py             Video encoder
│       ├── vla_models.py               VLA action heads
│       ├── fluid_medical_model.py      3D medical segmentation
│       └── ...
├── experiments/
│   ├── step0_mnist/            Image classification
│   ├── step1_video/            Video + adaptive compute
│   ├── step1b_medical_msd/     Medical 3D segmentation
│   ├── step2_sim/              Isaac Sim pick & place
│   ├── step2a_synthetic/       Synthetic imitation learning
│   ├── step2d_so101_urdf/      URDF viewer
│   ├── step3_lerobot/          Real robot (LeRobot SO-101)
│   └── _archive/
├── tests/                      Smoke tests
├── fluidvla_server.py          Web platform server
├── fluidvla_platform/          Web UI and dataset explorer
├── setup.py                    Package installation
├── requirements.txt            Dependencies
└── LICENSE                     MIT
```

Design principles:

- a single main root README,
- a stable facade via fluidvla.core for public imports,
- a local README per step for commands and usage,
- a `data/` subfolder per experiment,
- legacy variants go to archive.

---

## Experiment Pipeline

- [experiments/README.md](experiments/README.md)
- [experiments/step0_mnist/README.md](experiments/step0_mnist/README.md)
- [experiments/step1_video/README.md](experiments/step1_video/README.md)
- [experiments/step1b_medical_msd/README.md](experiments/step1b_medical_msd/README.md)
- [experiments/step2_sim/README.md](experiments/step2_sim/README.md)
- [experiments/step2a_synthetic/README.md](experiments/step2a_synthetic/README.md)
- [experiments/step2d_so101_urdf/README.md](experiments/step2d_so101_urdf/README.md)
- [experiments/step3_lerobot/README.md](experiments/step3_lerobot/README.md)

---

## Consolidated Results

### Step 0 - Image Classification

Validates that reaction-diffusion learns visual features without attention.

| Dataset | Accuracy | Params | Status |
| --- | ---: | ---: | --- |
| MNIST | 91.75% | ~0.30M | measured |
| CIFAR-10 | 62.25% | ~0.31M | measured |

Correct reading: mechanistic validation, not a SOTA benchmark.

### Spatial Memory Scaling

| Resolution | Pixels | FluidVLA VRAM | ViT estimated |
| --- | ---: | ---: | ---: |
| 32x32 | 1,024 | 13.3 MB | ~15 MB |
| 64x64 | 4,096 | 22.8 MB | ~60 MB |
| 128x128 | 16,384 | 59.2 MB | ~900 MB |
| 256x256 | 65,536 | 202.9 MB | ~14,000 MB |

On this point, the signal is genuinely impressive. Even with the necessary caution that the ViT column remains an estimate, the gap observed on the FluidVLA side is massive enough to justify the architectural interest.

### Step 1 - Video, Scaling and Adaptive Compute

Step 1 is the most important architectural validation after classification.

Changes historically integrated in this step:

- correction of a legacy motion loss that provided no real signal,
- addition of a proper spatial gradient loss,
- cleaned VRAM benchmark,
- separation between stop turbulence and differentiable turbulence,
- instrumentation: `steps_used`, `final_turbulence`, `min_turbulence`.

| Mode | Settings | Steps | Test MSE | Status |
| --- | --- | ---: | ---: | --- |
| Quality | `epsilon=0.08, min_steps=3, stop_patience=2` | 12.00 | 0.17804 | measured |
| Adaptive | `epsilon=0.09, min_steps=3, stop_patience=2` | 7.50 | 0.19695 | measured |
| Aggressive | `epsilon~0.11-0.12` | 3.00 | 0.20317 | measured |

This result shows that the compute dial is not a presentation gimmick but a real quality/compute trade-off control at inference.

### Step 1b - Medical 3D / MSD

This branch has two historical layers:

- an initial BrainTumour 3D prototype, now archived,
- a generalized pipeline over the 10 MSD tasks, now canonical.

#### Historical BrainTumour 3D Validation

Consolidated setup:

- dataset: MSD `Task01_BrainTumour`,
- 4 MRI modalities, volume shape `(240, 240, 155, 4)`,
- main protocol: `16 train / 4 val / 5 epochs`,
- loss: `Cross-Entropy + Soft Dice`,
- crop: `128^3`,
- model: `FluidBotMedical3D` with `d_model=32`, `n_layers=2`, `max_steps=6`, i.e., `16,632` parameters.

The historical medical step mainly produced an important methodological result: the discovery and correction of a real diffusion calibration bug.

| Config | Val Dice | Reading |
| --- | ---: | --- |
| PDE OFF | 0.8490 | control baseline |
| PDE ON before fix | 0.3846 | invalid comparison, diffusion broken |
| PDE ON fixed, `scale=0.05` | 0.8756 | becomes competitive again |
| PDE ON fixed, `scale=0.08` | 0.8867 | best point of the ablation |

Main controlled result:

| Config | Val Dice | Latency | VRAM | Delta Dice |
| --- | ---: | ---: | ---: | ---: |
| PDE OFF | 0.9129 | ~60 ms | ~632 MiB | - |
| PDE ON, `scale=0.08` | 0.9177 | ~89 ms | ~632 MiB | +0.0048 |

The gain is not enormous, but it is real, measurable, and achieved without memory explosion.

Reported baseline comparison against U-Net 3D:

| Model | Val Dice | Params | Latency | VRAM |
| --- | ---: | ---: | ---: | ---: |
| FluidVLA PDE ON | 0.9177 | 16,632 | ~44 ms GPU | ~632 MiB |
| UNet3D_Tiny | 0.8494 | 88,278 | 272 ms CPU | ~326 MiB |
| UNet3D_Std | 0.8233 | 5,603,746 | 2704 ms CPU | ~1891 MiB |

This comparison should be read as a signal of potential, not as a universally locked benchmark victory, since latencies are not measured on the same hardware.

#### Active MSD Pipeline

The current canonical pipeline covers:

- all 10 Medical Segmentation Decathlon tasks,
- single-modal and multi-modal inputs,
- FluidVLA training,
- U-Net 3D Tiny and Std baselines,
- unified inference,
- slice, multislice, 3D PNG and HTML renders,
- comparable splits via the same seed.

Supported tasks:

- Task01_BrainTumour
- Task02_Heart
- Task03_Liver
- Task04_Hippocampus
- Task05_Prostate
- Task06_Lung
- Task07_Pancreas
- Task08_HepaticVessel
- Task09_Spleen
- Task10_Colon

### Step 2a - Synthetic Imitation Learning

| Metric | Value | Status |
| --- | ---: | --- |
| Best Val MSE | 0.01345 | measured |
| Latency | ~4.1 ms | measured |
| Effective FPS | ~244 Hz | measured |
| Adaptive compute | 1/12 steps post-training | measured |

A vision + proprio + action stack that converges, holds around 4.1 ms, and loops at ~244 Hz on consumer GPU is already a very strong result for this phase.

### Step 2b - Isaac Camera Validation

| Metric | Value | Status |
| --- | ---: | --- |
| Black frames | 0 / 139 | measured |
| Red cube detection | 100% | measured |
| Green target detection | 100% | measured |
| Capture latency | ~1.6 ms/step | measured |
| Oracle success | 100% | measured |

### Step 2c - More Physical Isaac Collection and Training

Status: in progress.

### Step 2d - SO-101 URDF Viewer

Status: visualization and demonstration tool present.

### Step 3 - Real Hardware / Jetson / LeRobot

Status: active, with first offline validation on real LeRobot data.

Objectives:

- real-time latency on embedded hardware,
- bounded memory footprint,
- stable perception-action loop on real hardware.

Projection present in the repository: Jetson latency around ~40 ms, to be considered estimated until a proper full benchmark is finalized.

#### Step 3a - LeRobot Dataset Bridge

Goal: transform a real local LeRobot dataset into episodes compatible with the FluidVLA data contract.

Results measured on our local SO-101 dataset `local/so101_balle_bol_test` / `so101_balle_bol_dashboard_01`:

- 44 episodes,
- 18,081 frames,
- 2 recorded cameras: `observation.images.front` and `observation.images.wrist`,
- `fps=15`,
- `action_dim=6`,
- `proprio_dim=6`.

Bridge implementation: [experiments/step3_lerobot/convert_lerobot_dataset.py](experiments/step3_lerobot/convert_lerobot_dataset.py)

Validated smoke conversion:

| Element | Value | Status |
| --- | --- | --- |
| Source dataset | `so101_balle_bol_dashboard_01` | measured |
| Camera used for smoke test | `observation.images.front` | measured |
| Converted episodes | 2 | measured |
| Output format | `frames=(steps,3,4,96,96)` | measured |
| Output proprio | `(steps,6)` | measured |
| Output actions | `(steps,6)` | measured |

Correct reading: real LeRobot data can already be injected into the FluidVLA pipeline without depending on the historical Step 3 prototype.

#### Step 3b - Offline Learning on Real LeRobot Data

Goal: verify that FluidVLA learns offline from real SO-101 demonstrations collected with LeRobot.

Dedicated script: [experiments/step3_lerobot/train_lerobot_so101.py](experiments/step3_lerobot/train_lerobot_so101.py)

Smoke training measured on the converted subset `so101_front_smoketest`:

| Metric | Value | Status |
| --- | ---: | --- |
| Samples | 843 | measured |
| Parameters | ~0.16M | measured |
| Train MSE at epoch 1 | 2125.58554 | measured |
| Val MSE at epoch 1 | 1494.26098 | measured |
| Generated checkpoint | `best.pt` | measured |

Correct reading: this is not yet a final performance result, but an integration and offline learning validation on our real LeRobot data.

Full single-camera `front` run currently in progress on `so101_front_full`:

| Epoch | Train MSE | Val MSE | Val L1 | Eval latency | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 183.42098 | 22.78010 | 2.68786 | 6.40 ms | measured |
| 2 | 20.78908 | 20.58880 | 2.56215 | 6.29 ms | measured |
| 3 | 18.82856 | 19.26452 | 2.43343 | 6.77 ms | measured |
| 4 | 18.01675 | 18.03719 | 2.27493 | 6.55 ms | best provisional point |
| 5 | 17.62973 | 18.93611 | 2.65407 | 6.81 ms | slight val regression |
| 6 | 17.09176 | 18.58472 | 2.41866 | 6.93 ms | partial stabilization |
| 7 | 16.80891 | 17.82738 | 2.25727 | 6.77 ms | new best point |
| 8 | 16.65505 | 17.39248 | 2.21785 | 7.35 ms | new best point |
| 9 | 16.21888 | 17.01312 | 2.19877 | 7.64 ms | new best point |
| 10 | 16.29993 | 16.69330 | 2.19973 | 7.43 ms | new best point |
| 11 | 15.78486 | 17.39699 | 2.26212 | 6.67 ms | val regression, best remains epoch 10 |
| 12 | 15.63500 | 17.35693 | 2.38019 | 6.74 ms | high plateau, best remains epoch 10 |

Active configuration for this run:

- dataset: `data/step3_lerobot/so101_front_full`,
- checkpoints: `checkpoints/step3_lerobot/so101_front_full`,
- model: `~0.52M` parameters,
- input: `(3, 4, 224, 224)`,
- `action_dim=6`, `proprio_dim=6`,
- `max_steps=12`, `epsilon=0.02`, `batch_size=16`, `epochs=40`.

Correct reading: the important proof is already acquired. FluidVLA does not merely load this real data, it learns on it with a real validation decrease. The next milestone is now consolidation of the best checkpoint followed by a proper benchmark.

#### Step 3c - Runtime and Edge Validation

Status: partially started, not yet properly validated, and not yet tested in a live loop on the robot.

Current state:

- benchmark script available,
- benchmark run on smoke test checkpoint,
- benchmark run on the current best real checkpoint,
- smoke latency numbers available but non-canonical due to GPU contention or under-trained model,
- adaptive compute not yet observed on this smoke test.

Smoke benchmark currently measured on `so101_front_smoketest`:

| Metric | Value | Status |
| --- | ---: | --- |
| Mean latency | 77.84 ms | measured but non-canonical |
| P95 latency | 93.35 ms | measured but non-canonical |
| FPS | 12.85 | measured but non-canonical |
| Avg steps | 12.0 | measured |
| Dynamic steps | 12.0 | measured |

Dedicated benchmark on the current best real checkpoint `so101_front_full/best.pt`:

| Metric | Value | Status |
| --- | ---: | --- |
| Mean latency | 112.80 ms | measured, strict GPU benchmark |
| P50 latency | 109.48 ms | measured |
| P95 latency | 133.55 ms | measured |
| P99 latency | 153.56 ms | measured |
| FPS | 8.87 | measured |
| Avg steps | 12.0 | measured |
| Static steps | 12.0 | measured |
| Dynamic steps | 12.0 | measured |

Runtime epsilon sweep on the same checkpoint, without retraining:

| Epsilon | Mean latency | FPS | Avg steps | Static steps | Dynamic steps | Reading |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.20 | 99.31 ms | 10.07 | 12.0 | 12.0 | 12.0 | no triggering |
| 0.30 | 40.33 ms | 24.80 | 4.33 | 9.33 | 4.33 | first clear triggering |
| 0.40 | 44.09 ms | 22.68 | 4.33 | 7.0 | 4.33 | adaptive active |
| 0.50 | 32.98 ms | 30.32 | 3.67 | 4.0 | 4.33 | very promising zone |
| 0.60 | 34.96 ms | 28.61 | 3.67 | 3.33 | 3.67 | adaptive active |
| 0.80 | 34.06 ms | 29.36 | 3.67 | 3.0 | 3.67 | adaptive active |
| 1.00 | 33.87 ms | 29.53 | 3.67 | 3.0 | 3.67 | adaptive active |

Quality control on a clean validation split with the same checkpoint:

| Epsilon | Val MSE | Val L1 | Val steps | Batch latency val | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.02 | 16.69330 | 2.19973 | 12.0 | 7.14 ms | quality reference |
| 0.30 | 16.69442 | 2.20014 | 9.33 | 6.62 ms | nearly identical quality |
| 0.50 | 17.05135 | 2.27522 | 4.0 | 3.14 ms | slight quality degradation, large compute gain |

Correct reading for Step 3 at this stage:

- offline real learning on SO-101 is validated,
- the structural memory advantage of FluidVLA is supported by the benchmarks already present in the Spatial Memory Scaling and Video Scaling sections,
- but we have not yet finalized a canonical memory measurement specific to this Step 3 run nor a live test on the robot,
- and the real-time / adaptive compute demonstration on this real checkpoint is starting to become credible via runtime benchmark through an epsilon sweep. The most interesting point observed so far is `epsilon=0.30`, which maintains nearly identical validation quality while finally triggering early stopping. `epsilon=0.50` goes further on compute but begins to slightly degrade quality.

Current adaptive compute diagnostic on real data:

- during training, early stop is disabled by design in the PDE core,
- in evaluation, the full real run still stays at `12.0/12`,
- the current Step 3 setting `epsilon=0.02` is stricter than the video presets that show adaptation,
- therefore one must clearly distinguish "real pipeline that learns" from "real adaptive compute already validated": the former is acquired, the latter remains to be calibrated,
- one must also clearly distinguish "architectural memory advantage already measured in the project" from "canonical memory profile of the real Step 3 runtime", the latter still needing proper measurement.

What will count as canonical Step 3 results:

- full training on the complete real dataset,
- clean GPU benchmark without concurrent load,
- dedicated benchmark of the best real checkpoint with explored adaptive settings,
- live inference connected to the current LeRobot stack,
- then, ideally, a dual-camera `front + wrist` variant.

#### Step 3 V2 -- Diagnosis and correction of the frozen policy

The first live test of the V1 checkpoint (`so101_front_full/best.pt`, `Val MSE = 16.69330`) on the real SO-101 robot revealed a frozen policy: joints barely vary regardless of the image, confirmed by disabling all smoothing (`filter_alpha=1.0`).

**Root cause diagnosis -- 6 causes identified, ranked by impact:**

1. `AdaptiveAvgPool3d(1)` in `vla_models.py` crushes all spatial information from the PDE backbone before the decision. The model no longer knows *where* objects are in the image.
2. Absolute action supervision + MSE on a dataset where action ~ proprio (gap ~1-3 deg) directly rewards immobility.
3. No statistical normalization of actions or proprios -- raw scales (shoulder_lift: +/-90 deg vs gripper: 0-100) imbalance the MSE.
4. Single-step prediction without temporal horizon -- no chunking, no trajectory, guaranteed convergence toward a static attractor under ambiguity.
5. Front-only insufficient for fine grasping phase (no wrist view).
6. Action MLP too small (134->256->128->6) to exploit the backbone signal.

**Confirmation from logs:** `raw_delta` values are constant at +/-0.001 deg during 480+ consecutive steps, confirming the model learned a fixed point and not a policy.

**What the logs prove against other hypotheses:**

- It is not the EMA filter (constant even with `alpha=1.0`),
- It is not `max_delta` (deltas are well below the limit),
- It is not the camera (open and feeding the loop),
- It is not the adaptive compute (PDE steps vary normally).

**V2 corrections implemented:**

| # | Problem | File | Correction |
|---|---------|------|------------|
| 1 | Global pool crushes spatial info | `vla_models.py` | `AdaptiveAvgPool3d((1, 4, 4))` -- retains 16 spatial positions |
| 2 | Absolute action biased toward immobility | `convert_lerobot_dataset.py` | `--delta-actions` mode: target = normalized displacement |
| 3 | No state/action normalization | `convert_lerobot_dataset.py` | Normalizes proprios and actions by mean/std, saves `norm_stats.json` |
| 4 | No temporal chunking | `vla_models.py`, `train_lerobot_so101.py` | Optional `chunk_size` (1 by default, 4-8 recommended) |
| 5 | Action MLP undersized | `vla_models.py` | New `SpatialActionHead`: 2054->512->256->6*chunk |
| 6 | No directional signal in the loss | `train_lerobot_so101.py` | Cosine direction loss added |

**Architectural impact: the PDE core is strictly unchanged.** No modifications in `fluid_layer.py`, `video_models.py`, `diffusion.py`. Corrections target exclusively the pooling layer, the action head, the supervision formulation, and preprocessing.

**V2 modified files:**

- `fluidvla/core/vla_models.py` -- `SpatialActionHead` + `FluidBotVLA` with `spatial_pool_size` and `chunk_size`
- `fluidvla/core/fluid_model.py` -- added `SpatialActionHead` export
- `fluidvla/core/__init__.py` -- added `SpatialActionHead` export
- `experiments/step3_lerobot/convert_lerobot_dataset.py` -- normalization, delta-actions, static filtering
- `experiments/step3_lerobot/train_lerobot_so101.py` -- cosine loss, chunking, V2 config in checkpoint
- `experiments/step3_lerobot/lerobot_inference.py` -- `ActionDenormalizer`, normalized proprio, chunk execution

**Normalization statistics measured on the dataset:**

```
proprio mean : [ -1.29 -71.31  67.14  74.21  10.03  13.57]
proprio std  : [14.03 42.11 35.66  9.47 12.44 15.86]
delta mean   : [ 0.17 -0.77 -1.98 -0.25 -0.08 -3.55]
delta std    : [3.67  5.07  4.55  2.39  1.68  8.59]
Static frames (max|delta| < 0.5°): 0/18,081 (0.0%)
```

**Current status:**

- V2 dataset re-conversion: in progress,
- V2 training (`spatial_pool_size=4`, `chunk_size=1`, `--cosine_loss`): pending,
- V2 live test: pending.

#### Step 3d+ - LeRobot Optimization Backlog

For the continuation of Step 3, improvement tracks are organized by cost and risk level.

**Level 0 - Critical corrections (V2 -- in progress)**

- ~~frozen policy diagnostic~~ -- **done**, 6 root causes identified,
- ~~V2 pipeline implementation (spatial pool, delta-actions, normalization, cosine loss)~~ -- **done**,
- V2 dataset re-conversion -- **in progress**,
- V2 training and validation that the model learns non-zero deltas -- **to do**,
- V2 live test on the robot -- **to do**.

**Level 1 - Quick Wins runtime**

- epsilon runtime tuning, now validated as the main lever to activate adaptive compute without retraining; `epsilon=0.30` is the current best candidate,
- FP16 inference to reduce GPU cost and runtime memory footprint,
- systematic replacement of `torch.no_grad()` with `torch.inference_mode()` in dedicated inference paths,
- stabilization of a latency-oriented Step 3 runtime preset with clear documentation of the quality/compute trade-off.

**Level 2 - Software turbo**

- `torch.compile()` trial on the Step 3 inference loop,
- TensorRT export exploration for stabilized Step 3 checkpoints,
- comparative benchmark: native PyTorch vs compiled PyTorch vs TensorRT,
- canonical measurement of latency, VRAM, and jitter for each backend.

**Level 3 - Robotic and perception improvements**

- ~~action chunking~~ -- **implemented** in V2, to validate with `chunk_size=4` after `chunk_size=1` validation,
- dual-camera `front + wrist` fusion to reduce visual ambiguity, especially on depth and grasping phases,
- comparison of `front-only`, `wrist-only`, then `front + wrist`,
- validation of the dual-camera effect on action quality and, potentially, on natural reduction of adaptive steps,
- cleaner camera calibration and geometric consistency between views before proper synchronous fusion,
- possible addition of lightweight action smoothing or low-pass filter to limit jitter during live deployment,
- test of a hybrid policy + robot safeguards mode for initial real trials.

**Level 4 - System architecture**

- stricter separation of camera, inference, and motor control loops,
- software jitter reduction for extended live tests,
- study of a lower-level inference loop if Python becomes the bottleneck,
- option of retraining with lower `max_steps` if latency must be constrained by design rather than runtime only.

**Level 5 - Data, robustness, and learning efficiency**

- dataset curation to remove or tag the noisiest, most hesitant, or heavily occluded episodes,
- comparison of full dataset vs cleaner subset to measure the effect on model stability and confidence,
- analytical segmentation of demos into `approach`, `grasp`, `transport`, `place` phases to understand where the model hesitates most,
- visual robustness tests beyond simple Val MSE: occlusions, blur, small lighting variations, camera displacement, modest placement changes,
- comparison of action quality under perturbations between `front`, `wrist`, and future `front + wrist`.

**Level 6 - Policy outputs and confidence control**

- ~~action normalization~~ -- **done** in V2 (mean/std normalization per joint, delta-actions),
- addition of a confidence or uncertainty score to slow down or stabilize commands in ambiguous cases,
- possibility of retaining the previous action or limiting action delta when confidence drops,
- explicit instrumentation of zones where the policy becomes hesitant during live deployment.

**Level 7 - Compression and edge variants**

- creation of a dedicated `edge` preset with smaller image, fewer frames, or reduced `d_model`,
- distillation of a strong Step 3 checkpoint toward a lighter and faster variant,
- comparison between purely runtime optimization and cost reduction through model design,
- targeted retraining with lower `max_steps` if runtime optimizations alone are insufficient.

**Level 8 - Profiling and industrialization**

- layer-by-layer profiling to identify the real bottleneck between image encoder, PDE core, CPU/GPU copies, and camera capture,
- canonical benchmark of jitter and temporal stability, not just mean latency,
- backend comparison: native PyTorch vs `torch.compile()` vs TensorRT on the same checkpoints,
- exploration of a lower-level rewrite only if stabilized Python results show that software runtime becomes the real blocker.

Correct reading: levels 1 and 2 remain the next most cost-effective levers. Levels 3 and 4 become priorities once a first stable live mode exists. Levels 5 to 8 serve to transform a working prototype into a more robust, explainable, and credible edge robotics stack.

**Recommended Step 3 LeRobot battle plan**

To avoid scattering efforts, the recommended order is as follows:

1. ~~frozen policy diagnostic~~ -- **done**,
2. ~~V2 pipeline implementation~~ -- **done**,
3. V2 dataset re-conversion -- **in progress**,
4. V2 training with `spatial_pool_size=4`, `chunk_size=1`, `--cosine_loss`,
5. verify that V2 model learns non-zero deltas (Val MSE on normalized deltas < 1.0),
6. V2 live test on the robot,
7. if V2 works -> move to `chunk_size=4` then `chunk_size=8`,
8. `epsilon=0.30` as default runtime preset,
9. FP16 inference + `torch.inference_mode()`,
10. `wrist-only` baseline, then `front + wrist` comparison,
11. `torch.compile()` + TensorRT,
12. heavy system optimization if necessary.

Correct reading: points 1 and 2 are done, point 3 is in progress. Points 4 to 6 are the current blocker. Points 7 to 12 become relevant once the robot moves.

---

## Measured vs Estimated vs Future

To avoid any loss of rigor, the repository distinguishes three categories.

### Measured

- Step 0 classification results,
- image and video benchmarks present in the project,
- Step 1 adaptive compute calibration,
- Step 2a synthetic imitation learning,
- Step 2b Isaac Sim camera validation,
- historical medical 3D validation on BrainTumour,
- active MSD medical pipeline,
- real LeRobot -> FluidVLA `.npz` bridge validated on local SO-101 dataset,
- Step 3 smoke training offline on real LeRobot data with `action_dim=6` and `proprio_dim=6`,
- complete conversion of the real SO-101 `front` dataset over 44 episodes,
- full Step 3 single-camera `front` run with observed convergence and best `Val MSE` measured at `16.69330`,
- first V1 live test on real SO-101 robot: frozen policy confirmed (raw deltas constant at +/-0.001 deg),
- V2 diagnostic: 6 root causes identified and documented,
- V2 pipeline implemented: spatial pool 4x4, normalized delta-actions, cosine loss, action chunking,
- dataset normalization statistics measured: `delta_std=[3.67, 5.07, 4.55, 2.39, 1.68, 8.59]`,
- static frames in the dataset: 0/18,081 (0.0%) with 0.5 deg threshold.

### Estimated

- some Transformer comparisons used as scaling intuition,
- Jetson projection,
- extrapolations beyond exact benchmark.

### Future

- persistent BeliefField memory,
- imaginary rollout,
- local text module,
- symbols and scratchpad,
- native dual-camera fusion,
- V2 training validation and V2 live test on the robot,
- full real hardware benchmark,
- canonical adaptive compute validation on real robot data.

---

## Roadmap

### Phase 1 - Architectural Validation

- Step 0: attention-free classification
- Step 1: video + scaling + adaptive compute
- Step 1b: 3D medical segmentation and volumetric validation
- Step 2a: synthetic imitation learning
- Step 2b: Isaac camera validation
- Step 2c: more physical collection and training
- Step 2d: URDF viewer and 3D demo
- Step 3a: real LeRobot dataset bridge -> FluidVLA offline, **validated**
- Step 3b: offline single-camera training on full real SO-101 dataset, **V1 converged** (Val MSE 16.69), **V2 in progress**
- Step 3b-v2: frozen policy diagnostic + V2 corrections (spatial pool, delta-actions, normalization, cosine loss), **implemented**, training pending
- Step 3c: proper benchmark and adaptive compute validation on real checkpoint
- Step 3d: dual-camera `front + wrist` fusion on real LeRobot data
- Step 3e: live SO-101 inference via current LeRobot stack, **first V1 test executed** (frozen policy), **V2 test pending**
- Step 3f: embedded Jetson / edge robotics benchmark

### Phase 2 - Bio-Inspired Extensions

- leaky memory / synaptic fatigue,
- dynamic spatial pruning,
- semantic inhibition,
- modulated spatial integration,
- lightweight Hebbian adaptation.

### Phase 3 - Memory and Planning

- persistent BeliefField,
- Imaginary Rollout,
- distillation of heavier teachers toward a lighter PDE student.

### Phase 4 - Multimodality and Reasoning

- local text diffusion,
- symbolic anchors,
- external scratchpad,
- reasoning rollout,
- extensions toward certain scientific domains.

#### BeliefField

BeliefField is the persistent memory component designed to maintain a latent state across multiple calls, without falling back to a giant KV-cache.

```text
BeliefField_t = λ · BeliefField_{t-1} + Write(u_t, obs_t) - Decay(BeliefField_{t-1})
Action_t      = Policy(u_t, Read(BeliefField_t))
```

#### Imaginary Rollout

Imaginary Rollout is the short-range planning component: a few candidate latent futures, a fast action selection, and a local see -> simulate -> act mechanism.

#### Symbolic Anchors

Symbolic anchors aim to manipulate a small set of salient symbols or identities at a cost closer to $O(K)$ than $O(N^2)$.

#### External ScratchPad

The ScratchPad is distinct from BeliefField:

- BeliefField for continuous latent memory,
- ScratchPad for explicit structured memory needed for certain multi-step reasoning tasks.

---

## Progress Log

### 2026-03-10

- Root README enriched to track Step 3 like other repository experiments.
- Real LeRobot -> FluidVLA bridge implemented and validated on the local SO-101 dataset.
- Smoke conversion validated on 2 episodes with coherent `frames`, `proprios`, `actions` outputs.
- Smoke training validated with generation of `best.pt`, `history.json`, and `benchmark.json`.
- Full `front` conversion completed on 44 episodes in `data/step3_lerobot/so101_front_full`.
- Full single-camera `front` run launched in `checkpoints/step3_lerobot/so101_front_full_train.log`.
- The full run then passed its first plateau: epoch 7, `Val MSE = 17.82738`, `Val L1 = 2.25727`, eval latency `~6.77 ms`.
- The run continues improving at epoch 8 with `Val MSE = 17.39248` and `Val L1 = 2.21785`.
- Epochs 9 and 10 extend this trend with a new best point at `Val MSE = 16.69330` at epoch 10.
- Epoch 11 drops on the train side but rises in validation to `17.39699`, leaving the best checkpoint unchanged at epoch 10.
- Epoch 12 currently confirms a validation plateau above epoch 10 with `Val MSE = 17.35693` and `Val L1 = 2.38019`.
- Code diagnostic confirmed: adaptive compute cannot activate in train mode and has not yet emerged in eval with `epsilon=0.02`.
- Runtime epsilon sweep executed on `best.pt`: no effect up to `0.20`, then clear triggering from `0.30` onward with `avg_steps ~ 4.33` and latency around `40 ms`; the `0.50-1.00` zone drops to around `3.67` steps and `~33-35 ms`.
- Quality control then executed on a clean validation split: `epsilon=0.30` maintains nearly the same quality as `0.02`, while `0.50` slightly degrades validation to gain further compute savings.

### 2026-03-10 (evening) -- V2 Diagnostic

- First live test of the V1 checkpoint (`so101_front_full/best.pt`) on the real SO-101 robot.
- Result: frozen policy, the robot barely moves. Raw deltas constant at +/-0.001 deg during 480+ steps.
- Confirmed with `filter_alpha=1.0` and `max_delta=6`: the problem comes from the model, not the filter or limits.
- Full diagnostic executed on the 6 pipeline files (vla_models, video_models, fluid_layer, train, convert, inference).
- 6 root causes identified: (1) global pool destroys spatial info, (2) absolute action biased toward immobility, (3) no normalization, (4) no chunking, (5) MLP too small, (6) no directional signal.
- V2 pipeline implemented: `vla_models.py` (SpatialActionHead, spatial_pool_size, chunk_size), `convert_lerobot_dataset.py` (delta-actions, normalization, static filtering), `train_lerobot_so101.py` (cosine loss, chunking, V2 config), `lerobot_inference.py` (ActionDenormalizer, normalized proprio, chunk execution).
- Facades updated: `fluid_model.py` and `__init__.py` export `SpatialActionHead`.
- PDE core strictly unchanged (fluid_layer.py, video_models.py, diffusion.py).
- V2 dataset re-conversion launched with `--delta-actions --filter-static 0.5 --subsample-static 4`.
- Delta statistics measured: `mean=[0.17, -0.77, -1.98, -0.25, -0.08, -3.55]`, `std=[3.67, 5.07, 4.55, 2.39, 1.68, 8.59]`.
- Unexpected result: 0% static frames (|delta| < 0.5 deg) -- the dataset is more dynamic than expected, the problem is clearly algorithmic.

### Next Steps

- Verify that V2 re-conversion is complete and that `norm_stats.json` is present.
- Launch V2 training: `--spatial_pool_size 4 --chunk_size 1 --cosine_loss --epochs 100`.
- Monitor that Val MSE (on normalized deltas) drops below 1.0 -- sign that the model is learning something beyond immobility.
- If it converges -> test live on the robot and verify that raw_delta varies with the scene.
- If raw_delta varies -> move to `chunk_size=4` for a planning horizon.
- If everything remains frozen -> investigate whether the 4x4 spatial pool is sufficient or if a larger MLP / more data is needed.

---

## Web Platform

The repository contains a standalone local server for orchestrating training, inference, and model comparison.

Currently integrated features:

- dataset scanning,
- automatic checkpoint scanning,
- training and inference job launching,
- PNG, HTML, and JSON file rendering,
- modular SPA for experiments,
- native Dataset Explorer,
- REST API and WebSocket streams.

Useful entry points:

- [fluidvla_server.py](fluidvla_server.py)
- [fluidvla_platform/dataset_explorer.py](fluidvla_platform/dataset_explorer.py)
- [fluidvla_platform/interactive.html](fluidvla_platform/interactive.html)
- [start_platform.bat](start_platform.bat)
- [fluidvla/core/README.md](fluidvla/core/README.md)
- [experiments/step1b_medical_msd](experiments/step1b_medical_msd)

---

## Data and Checkpoint Conventions

| Directory | Produced by | Consumed by |
| --- | --- | --- |
| `data/step1_video/` | download or Moving MNIST generation | `train_step1_video.py` |
| `data/step1b_medical_msd/<Task>/` | manual import of MSD datasets | `train_fluidvla_msd.py`, `train_unet3d_msd.py`, `infer_msd.py` |
| `data/step2_sim/` | `isaac_env.py --mode synthetic` | `train_step2.py` |
| `data/step2a_synthetic/` | `synthetic_env.py` | `train_step2a.py` |
| `data/step2c_isaac/` | `isaac_env.py --mode collect` | fine-tuning / more physical Step 2 |
| `data/step3_lerobot/` | conversion from real local LeRobot datasets | `train_lerobot_so101.py`, Step 3 benchmarks |
| `checkpoints/fluidvla/<Task>/` | `train_fluidvla_msd.py` | `infer_msd.py`, web platform |
| `checkpoints/unet3d/<Task>/` | `train_unet3d_msd.py` | `infer_msd.py`, web platform |
| `checkpoints/step2_sim/` | `train_step2.py` | Isaac evaluation / fine-tuning |
| `checkpoints/step2a_synthetic/` | `train_step2a.py` | URDF viewer or controlled resumptions |
| `checkpoints/step2c_isaac/` | `train_step2.py` on `data/step2c_isaac/` | URDF viewer and Step 3 bridge |
| `checkpoints/step2d_so101_urdf/` | `so101_urdf_viewer.py` | demonstration and inspection |
| `checkpoints/step3_lerobot/` | `train_lerobot_so101.py` on converted real LeRobot data | benchmark, future live and edge passes |

The [data/README.md](data/README.md) directory fixes these conventions in compact form.

---

## How to Cite

The project does not yet have a finalized public preprint or definitive DOI. In the meantime, the recommended citation form is a provisional software citation.

```bibtex
@software{fluidvla_prototype,
  title  = {FluidVLA: Transformer-Free Vision-Language-Action via Reaction-Diffusion PDEs},
  author = {infinition},
  year   = {2026},
  note   = {Research prototype, code repository},
  url    = {https://github.com/infinition/FluidVLA}
}
```

When a preprint or stabilized public release exists, this section should be replaced with the canonical reference.

---

## Preprint Status

- **Status:** research prototype
- **Paper:** not released yet
- **Code:** active and evolving
- **Claim level:** strong architectural evidence on several axes, but not yet a final product or definitive real-hardware paper result

---

## License

Project maintained by **infinition**.

If a public page, contact email, or official mirror repository needs to be exposed, this section is the proper place to add it.
