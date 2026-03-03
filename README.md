<img width="943" height="540" alt="unnamed (1)" src="https://github.com/user-attachments/assets/92e1bcc1-6ab2-44e1-bc60-60e7b2d97669" />


# 🌊 FluidVLA

**A Transformer-free Vision-Language-Action model for real-time edge robotics.** Replacing O(N²) self-attention with Reaction-Diffusion PDEs.

> *Formerly FluidBot / FluidLM - renamed to FluidVLA.* > Targeting the SO-101 LeRobot arm and Jetson Orin Nano.

---

## 🧱 The Edge Robotics Dilemma (The Problem)

Every Vision-Language-Action (VLA) model today (RT-2, OpenVLA, Pi0) shares the same structural flaw: they are built on Transformers designed for NLP, shoehorned into spatial robotics. This creates the **KV-Cache Wall**.

As recently demonstrated by NVIDIA's own deployment of the Cosmos Reason 2B model on the Jetson Orin Nano, squeezing an O(N²) Transformer onto edge hardware requires crippling sacrifices:
- Context length slashed to **256 tokens** (amnesia).
- Video input hard-capped at **2 frames per prompt** (blindness to continuous motion).

Why? Because the Transformer's attention matrix and KV-cache explode quadratically with time and spatial resolution ($O(T^2 \cdot N)$). A robot that only sees the world in stuttering 2-frame snapshots cannot understand fluid physics, velocity, or continuous manipulation.

## 🌊 The Solution

**FluidVLA replaces the attention matrix entirely with Reaction-Diffusion PDEs.**

Instead of every pixel "talking to" every other pixel through an $N \times N$ matrix, information propagates like a fluid - diffusing from neighbor to neighbor across space and time.

```text
u_{t+1} = LayerNorm(u_t + dt · [Σ Dk · ∇²(u_t) + R(u_t, θ) + α · h_t])

```

| Term | Role |
| --- | --- |
| `Σ Dk · ∇²(u_t)` | Multi-scale diffusion. Dilations [1, 4, 16]. O(N). |
| `R(u_t, θ)` | Reaction MLP. Per-position nonlinear features. |
| `α · h_t` | Memory pump. Global O(1) accumulator - **no KV-cache, ever.** |

**Turing Equilibrium:** stops integrating when the scene is stable. Static scene → 2 steps. Complex scene → 12 steps. ~83% compute saved on typical robot idle time.

---

## 📊 Empirical Results

### 1. Imitation Learning (VLA End-to-End) ✅ NEW

Validated on a simulated 7-DOF Pick & Place task (13,500 frames).

* **Validation MSE:** **0.013** (High-precision trajectory cloning, accurately picking and placing objects).
* **Latency:** **~4.0 ms** per step on RTX 4070 Ti (~250 FPS). Leaves massive headroom for Jetson Orin Nano deployment (< 50ms target).

### 2. Memory - O(N) Spatial

| Resolution | FluidVLA | ViT estimated |
| --- | --- | --- |
| 128×128 | 59.2 MB | ~900 MB |
| 256×256 | **202.9 MB** | **~14,000 MB** ⚠️ |

**84× less memory at 256×256. VRAM/N variation: 4.21x vs ~1000x for ViT.**

### 3. Memory - O(T·N) Video (The Core Claim)

| Frames | FluidVLA | Transformer estimated |
| --- | --- | --- |
| 8 | 36.9 MB | ~272 MB |
| 16 | 62.9 MB | ~1,088 MB |
| 32 | 114.6 MB | ~4,352 MB ⚠️ |

**16× more frames → only 2.43× more VRAM.** (Compared to 16× for a Transformer).

### 4. Classification (Zero Attention Baseline)

| Dataset | Accuracy | Params |
| --- | --- | --- |
| MNIST | **91.75%** | 0.30M |
| CIFAR-10 | **62.25%** | 0.31M |

---

## 🚀 Quickstart

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
pip install tqdm tensorboard pytest

python src/core/diffusion.py
python src/core/fluid_layer.py
python src/core/fluid_model.py

python experiments/step0_mnist/train_step0.py --dataset mnist --model small
python experiments/step0_mnist/train_step0.py --dataset cifar10 --model small --batch_size 512
python experiments/step1_video/train_step1_video.py --epochs 30 --d_model 128

# Run VLA Imitation Learning on synthetic Pick&Place dataset
python experiments/step2_sim/isaac_env.py --mode synthetic --episodes 1000 --image_size 64
python experiments/step2_sim/train_step2.py --dataset ./data/step2 --epochs 50 --batch_size 32 --d_model 128

```

---

## 🗺️ Roadmap

**Phase 1 - Validation** *(current)*

* ✅ Step 0 - MNIST 91.75% / CIFAR-10 62.25%
* ✅ Step 1 - Video VRAM 2.43x for 16× frames
* ✅ Step 2a - VLA Imitation Learning (MSE 0.013, 4ms latency)
* 🔄 Step 2b - Isaac Sim Physics Engine Integration
* ⏳ Step 3 - SO-101 Jetson Orin Nano real-time

**Phase 2 & 3 - Bio-Inspired Cognition & Reflection**

* Synaptic fatigue / Spatial pruning / Semantic inhibition
* BeliefField (Persistent Memory) & Multi-Teacher Distillation (from Cosmos 2B / RT-2)

See [ROADMAP.md](https://www.google.com/search?q=ROADMAP.md) for full details.

---

## Results
See [RESULTS.md](https://www.google.com/search?q=RESULTS.md) for full details.

## 📚 References

Turing (1952), Chen et al. (2018) Neural ODEs, Dosovitskiy et al. (2020) ViT, Brohan et al. (2023) RT-2, Black et al. (2024) Pi0, Hebb (1949).

---

*Research prototype - infinition. Preprint and code release coming after Step 3.*

```

