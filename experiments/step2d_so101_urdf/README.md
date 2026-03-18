# Step 2d -- SO-101 URDF Viewer

Goal: provide a demo-oriented 3D viewer to load the SO-101 in Isaac Sim and drive
trajectories from FluidVLA checkpoints.

Entry point:

- `so101_urdf_viewer.py`

Prerequisites:

- Isaac Sim 4.x,
- SO-101 URDF,
- Step 2a or Step 2c checkpoint.

Commands:

```bash
python experiments/step2d_so101_urdf/so101_urdf_viewer.py --checkpoint ./checkpoints/step2c_isaac/best.pt --urdf /path/to/so101.urdf --episodes 5 --show_gui
python experiments/step2d_so101_urdf/so101_urdf_viewer.py --checkpoint ./checkpoints/step2a_synthetic/best.pt --urdf /path/to/so101.urdf --headless
python experiments/step2d_so101_urdf/so101_urdf_viewer.py --urdf /path/to/so101.urdf --random_weights
```

This step is primarily a visualization and demonstration tool. Results and roadmap
position are centralized in the root [README.md](../../README.md).
