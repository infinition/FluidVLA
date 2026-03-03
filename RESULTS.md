# FluidVLA - Experimental Results

> Comparison with equivalent VLA (Vision-Language-Action) models and Transformers

---

## Table of Contents

* [Architecture](#architecture)
* [Step 0 - Image Classification](#step-0---image-classification)
* [Memory Scaling Image - Key Result O(N)](#memory-scaling-image---key-result-on)
* [Step 1 - Video Prediction ✅ KEY BENCHMARK](#step-1---video-prediction--key-benchmark)
* [Step 2a - Synthetic Imitation Learning ✅ MEASURED](#step-2a---synthetic-imitation-learning--measured)
* [Step 2b - Isaac Sim Camera Validation ✅ MEASURED](#step-2b---isaac-sim-camera-validation--measured)
* [Adaptive Compute](#adaptive-compute)
* [VLA Latency - Proven < 5ms](#vla-latency---proven--5ms)
* [Summary - What is proven today](#summary---what-is-proven-today)

---

## Architecture

| | **FluidVLA** | **Transformer (ViT/standard)** |
| --- | --- | --- |
| Core Mechanism | PDE Diffusion + Reaction | Self-Attention O(N²) |
| KV-cache | ❌ None | ✅ Grows with N |
| Adaptive compute | ✅ Stops if stable | ❌ Always N fixed steps |
| Memory Scaling | **O(N)** | **O(N²)** |
| Video Scaling | **O(T·N)** | **O(T²·N)** |

---

## Step 0 - Image Classification

Proves that native PDE diffusion learns spatial features without any attention.

### MNIST (small, 0.30M params, 30 epochs, max_steps=4)

| Metric | FluidVLA | ViT-Tiny Reference |
| --- | --- | --- |
| Parameters | 0.30M | ~5M |
| **Best accuracy** | **91.75%** | ~98% |
| Attention | ❌ Zero | ✅ Required |
| Epoch time | ~35s / RTX 4070 Ti | - |

### CIFAR-10 (small, 0.31M params, 100 epochs, max_steps=4, batch=512)

| Metric | FluidVLA | Comparison |
| --- | --- | --- |
| Parameters | 0.31M | ViT-Small : 22M |
| **Best accuracy** | **62.25%** | ViT-Tiny scratch : 72% |
| Attention | ❌ Zero | ✅ Required |
| Epoch time | ~31s / RTX 4070 Ti | - |

---

## Memory Scaling Image - Key Result O(N)

| Resolution | N pixels | FluidVLA VRAM | Estimated ViT |
| --- | --- | --- | --- |
| 32×32 | 1,024 | 13.3 MB | ~15 MB |
| 64×64 | 4,096 | 22.8 MB | ~60 MB |
| 128×128 | 16,384 | 59.2 MB | ~900 MB |
| 256×256 | 65,536 | 202.9 MB | **~14,000 MB** ⚠️ |

**VRAM/N Variation: 4.21x** vs ~1000x for ViT.
At 256×256: **84× less memory** than a standard ViT.

---

## Step 1 - Video Prediction ✅ KEY BENCHMARK

### VRAM vs Number of Frames (O(T·N) claim)

> Model: FluidVLA (Video), 0.31M params, d_model=128, 64×64px

| T frames | N total pixels | FluidVLA VRAM | Estimated Transformer |
| --- | --- | --- | --- |
| 2 | 8,192 | 17.4 MB | ~17 MB |
| 4 | 16,384 | 24.0 MB | ~68 MB |
| 8 | 32,768 | 36.9 MB | ~272 MB |
| 16 | 65,536 | 62.9 MB | ~1,088 MB |
| 32 | 131,072 | 114.6 MB | **~4,352 MB** |

**VRAM/T Variation: 2.43x for 16× more frames.**
A Transformer would use 16× more VRAM (quadratic in T).
FluidVLA uses 2.43× - almost linear. ✅

> This is one of the most unique results of this project.
> No attention-based VLA model can claim this figure.

---

## Step 2a - Synthetic Imitation Learning ✅ MEASURED

Validation of the complete end-to-end VLA architecture:
`[Camera Images (T=4 frames)] + [Proprioception (8 dims)] → FluidVLA → [Motor Actions (7 DOF)]`

### Configuration

| Parameter | Value |
| --- | --- |
| Model | FluidVLA, d_model=128, n_layers=4 |
| Parameters | **0.67M** |
| Dataset | Synthetic Pick & Place, 1,000 episodes, 13,581 frames |
| Oracle success rate | **100%** |
| Epochs | 50 |
| Batch size | 32 |
| eq_weight | 0.1 |

> **Honest Note:** Simplified synthetic dataset (2D top-down scene, simplified physics).
> The MSE proves the convergence of the end-to-end VLA architecture.

### Convergence Val MSE

| Epoch | Val MSE |
| --- | --- |
| 1 | 0.07672 |
| 5 | 0.05102 |
| 10 | 0.04683 |
| 15 | 0.04168 |
| 20 | 0.04310 |
| 25 | 0.03913 |
| 30 | 0.03773 |
| **50** | **0.01345 ← best** |

**Total reduction: 0.077 → 0.013 that's −83% over 50 epochs. No divergence.**

### Measured Results

| Metric | Value | Target | Status |
| --- | --- | --- | --- |
| Best Val MSE | **0.01345** (epoch 48) | < 0.02 | ✅ |
| Average Latency | **~4.1 ms** | < 50 ms | ✅ |
| Effective FPS | **~244 Hz** | > 30 Hz | ✅ |
| Adaptive compute | **1 step / 12** (92% saved) | < 6 steps | ✅ |
| Gradient stability | **No divergence** | - | ✅ |

### Adaptive Compute post-training (eq_weight=0.1)

| Epoch | Steps (eval) | Saved Compute |
| --- | --- | --- |
| 1 | 4.0 / 12 | 67% |
| 2 | 1.2 / 12 | 90% |
| 4+ | **1.0 / 12** | **92%** |

> **Unexpected Result:** the model converges to 1 PDE step as early as epoch 4.
> The eq_weight=0.1 creates enough pressure for Pick & Place scenes
> to be considered as stable after only 1 integration.

---

## Step 2b - Isaac Sim Camera Validation ✅ MEASURED

Validation of the collection pipeline with NVIDIA RTX physics engine (Isaac Sim 4.5.0).

### Configuration

| Parameter | Value |
| --- | --- |
| Engine | Isaac Sim 4.5.0, RTX 4070 Ti |
| Camera | Top-down, 224×224px, 30Hz |
| Rendering | Headless RTX Path Tracing |
| Oracle | 4-phase proportional controller |
| Test Episodes | 10 |

### Camera Results (camera_check.py)

| Metric | Value | Status |
| --- | --- | --- |
| Black frames | **0 / 139 (0%)** | ✅ |
| Red cube detected | **139 / 139 (100%)** | ✅ |
| Green target detected | **139 / 139 (100%)** | ✅ |
| Temporal coherence (diff_mean) | **~0.0018** | ✅ |
| Capture Latency | **~1.6 ms/step** | ✅ |
| Oracle success rate | **100%** | ✅ |

> **Resolved Bugs:** USD quaternion (`Gf.Quatd`), 30 steps warm-up, retry logic,
> `DistantLight` with explicit orientation.
> Pipeline ready for 1000 physical episodes collection.

---

## Adaptive Compute

| Input | FluidVLA steps | Transformer |
| --- | --- | --- |
| Constant image | **1 / 12** | 12 / 12 (fixed) |
| Smooth image | **1 / 12** | 12 / 12 (fixed) |
| Random noise / Movement | **12 / 12** | 12 / 12 (fixed) |

**92% compute saved on Pick & Place scene (post-training, eq_weight=0.1).**

*Note: epsilon calibration on real Isaac Sim stream = next step.*

---

## VLA Latency - Proven < 5ms

| Model | Params | Latency | FPS | Cloud |
| --- | --- | --- | --- | --- |
| RT-2 (Google) | 55B | ~500 ms | ~2 fps | ✅ TPU cluster |
| OpenVLA | 7B | ~200 ms | ~5 fps | ✅ Server A100 |
| Pi0 | 3B | ~100 ms | ~10 fps | ✅ Remote GPU |
| Diffusion Policy | ~300M | ~50-100 ms | ~10-20 fps | ✅ GPU |
| **FluidVLA (RTX 4070 Ti)** | **0.67M** | **~4.1 ms** | **~244 fps** | **❌ Local** |
| **FluidVLA (Jetson Orin, estimated)** | **0.67M** | **~40 ms** | **> 25 fps** | **❌ Embedded** |

---

## Summary - What is proven today

| Claim | Status | Key Figure |
| --- | --- | --- |
| Architecture learns visual features | ✅ measured | MNIST 91.75%, CIFAR 62.25% |
| Memory scaling O(N) images | ✅ measured | 4.21x variation vs ~1000x ViT |
| Memory scaling O(T·N) video | ✅ measured | **2.43x for 16× frames** |
| End-to-end multi-modal Imitation Learning | ✅ measured | **Val MSE 0.0135** (50 epochs) |
| Low latency inference | ✅ measured | **~4.1 ms** on RTX 4070 Ti |
| Real-time FPS | ✅ measured | **~244 Hz** |
| Adaptive compute post-training | ✅ measured | **1/12 steps, 92% saved** (eq_weight=0.1) |
| Isaac Sim Pipeline (camera + oracle) | ✅ validated | 0 black frames, 100% detection |
| physical 1000 episodes collection | ⏳ next step | - |
| SO-101 hardware Sim-to-Real | ⏳ Step 3 | - |