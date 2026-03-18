# Step 1b -- Medical MSD Segmentation

Goal: serve as the canonical medical 3D pipeline for MSD tasks.

Entry points:

- `msd_dataset.py`
- `train_fluidvla_msd.py`
- `train_unet3d_msd.py`
- `infer_msd.py`
- `list_cases.py`

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

Generic commands:

```bash
python experiments/step1b_medical_msd/list_cases.py --data_dir ./data/step1b_medical_msd/Task09_Spleen
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
python experiments/step1b_medical_msd/train_unet3d_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
python experiments/step1b_medical_msd/infer_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --model_type fluidvla --checkpoint ./checkpoints/fluidvla/Task09_Spleen/best_fluidvla.pt --case spleen_1.nii.gz --output_dir ./inference_outputs/Task09_Spleen
```

Task-specific presets:

- Hippocampus: `--depth 64 --height 64 --width 64`
- Prostate: `--depth 128 --height 128 --width 64`
- Pancreas / HepaticVessel / Colon: `--crop_mode mixed` often useful

Examples:

```bash
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task01_BrainTumour --data_dir ./data/step1b_medical_msd/Task01_BrainTumour --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task04_Hippocampus --data_dir ./data/step1b_medical_msd/Task04_Hippocampus --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4 --depth 64 --height 64 --width 64 --crop_mode mixed
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task05_Prostate --data_dir ./data/step1b_medical_msd/Task05_Prostate --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4 --depth 128 --height 128 --width 64 --crop_mode mixed
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task07_Pancreas --data_dir ./data/step1b_medical_msd/Task07_Pancreas --epochs 10 --batch_size 1 --max_train_samples 24 --max_val_samples 6 --depth 128 --height 128 --width 128 --d_model 32 --n_layers 2 --max_steps 6 --diffusion_scale 0.08 --lr 0.001 --binary --crop_mode mixed
```

Typical outputs:

- `checkpoints/fluidvla/<Task>/best_fluidvla.pt`
- `checkpoints/unet3d/<Task>/best_unet3d_tiny.pt`
- `checkpoints/unet3d/<Task>/best_unet3d_std.pt`
- `inference_outputs/<Task>/...`

Historical reference:

- [medical_step0_results_exhaustive.md](medical_step0_results_exhaustive.md)

Results, limitations, and this step's position in the overall trajectory are centralized in the root [README.md](../../README.md).
