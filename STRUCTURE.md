FluidVLA — Complete and coherent overhaul
1. Final project structure
FluidVLA/
├── src/core/
│   ├── diffusion.py               # unchanged
│   ├── fluid_layer.py             # + use_pde flag
│   └── fluid_model.py             # + use_pde propagated
│
├── experiments/
│   ├── step0_mnist/
│   │   └── train_step0.py         # + --no_pde flag
│   ├── step1_video/
│   │   └── train_step1_video.py   # + --no_pde flag
│   ├── step2a_synthetic/          # ← was "step2_sim"
│   │   ├── synthetic_env.py       # oracle + isolated synthetic env
│   │   └── train_step2a.py        # training on synth
│   ├── step2b_isaac_validate/     # Isaac Sim camera validation
│   │   └── camera_check.py        # unchanged, path updated
│   ├── step2c_isaac_collect/      # physical collection + training
│   │   ├── isaac_env.py           # IsaacPickPlace isolated
│   │   └── train_step2c.py        # fine-tune from step2a ckpt
│   ├── step2d_so101_urdf/         # 🆕 3D Viewer SO-101 URDF
│   │   ├── so101_urdf_viewer.py   # Isaac Sim + URDF + live inference
│   │   └── README.md              # URDF guide + alternatives
│   └── step3_lerobot/
│       └── lerobot_inference.py   # unchanged, paths updated
│
├── data/
│   ├── step2a_synthetic/          # synthetic episodes
│   ├── step2c_isaac/              # Isaac Sim physical episodes
│   └── step3_lerobot/             # hand-guided hardware demos
│
├── checkpoints/
│   ├── step2a/best.pt             # MSE 0.013, 4.1ms ✅
│   ├── step2c_isaac/best.pt       # physical fine-tuning
│   ├── step2d_so101_urdf/best.pt  # validated 3D viewer
│   └── step3/best.pt              # Jetson deployment
│
├── ROADMAP.md
└── README.md