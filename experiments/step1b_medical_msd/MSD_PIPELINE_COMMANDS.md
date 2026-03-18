# FluidVLA -- Step Medical MSD Commands

> Canonical commands for the active 3D medical pipeline in this repo.

---

## Canonical location

```text
FluidVLA-main/
+-- experiments/
    +-- step1b_medical_msd/
        +-- msd_dataset.py
        +-- train_fluidvla_msd.py
        +-- train_unet3d_msd.py
        +-- infer_msd.py
        +-- list_cases.py
```

This directory replaces the older limited `step_medical_brats` prototype and serves as the main medical pipeline.

---

## Supported tasks

| Task | Organ | Channels | Recommended crop |
| --- | --- | ---: | --- |
| Task01_BrainTumour | Brain tumour | 4 | 128x128x128 |
| Task02_Heart | Left atrium | 1 | 128x128x128 |
| Task03_Liver | Liver + tumour | 1 | 128x128x128 |
| Task04_Hippocampus | Hippocampus | 1 | 64x64x64 |
| Task05_Prostate | Prostate | 2 | 128x128x64 |
| Task06_Lung | Lung nodule | 1 | 128x128x128 |
| Task07_Pancreas | Pancreas + tumour | 1 | 128x128x128 |
| Task08_HepaticVessel | Vessels + tumour | 1 | 128x128x128 |
| Task09_Spleen | Spleen | 1 | 128x128x128 |
| Task10_Colon | Colon | 1 | 128x128x128 |

---

## Prerequisites

```bat
pip install torch nibabel scikit-image plotly matplotlib numpy
```

---

## Generic commands

Canonical data convention: `./data/step1b_medical_msd/<Task>`

List available cases:

```bat
python experiments/step1b_medical_msd/list_cases.py --data_dir ./data/step1b_medical_msd/Task09_Spleen
```

Train FluidVLA:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
```

Train U-Net 3D:

```bat
python experiments/step1b_medical_msd/train_unet3d_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
```

FluidVLA inference:

```bat
python experiments/step1b_medical_msd/infer_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --model_type fluidvla --checkpoint ./checkpoints/fluidvla/Task09_Spleen/best_fluidvla.pt --case spleen_1.nii.gz --output_dir ./inference_outputs/Task09_Spleen
```

U-Net Std inference:

```bat
python experiments/step1b_medical_msd/infer_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --model_type unet3d_std --checkpoint ./checkpoints/unet3d/Task09_Spleen/best_unet3d_std.pt --case spleen_1.nii.gz --output_dir ./inference_outputs/Task09_Spleen
```

U-Net Tiny inference:

```bat
python experiments/step1b_medical_msd/infer_msd.py --task Task09_Spleen --data_dir ./data/step1b_medical_msd/Task09_Spleen --model_type unet3d_tiny --checkpoint ./checkpoints/unet3d/Task09_Spleen/best_unet3d_tiny.pt --case spleen_1.nii.gz --output_dir ./inference_outputs/Task09_Spleen
```

---

## Useful examples per task

BrainTumour:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task01_BrainTumour --data_dir ./data/step1b_medical_msd/Task01_BrainTumour --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4
python experiments/step1b_medical_msd/infer_msd.py --task Task01_BrainTumour --data_dir ./data/step1b_medical_msd/Task01_BrainTumour --model_type fluidvla --checkpoint ./checkpoints/fluidvla/Task01_BrainTumour/best_fluidvla.pt --case BRATS_001.nii.gz --output_dir ./inference_outputs/Task01_BrainTumour
```

Hippocampus:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task04_Hippocampus --data_dir ./data/step1b_medical_msd/Task04_Hippocampus --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4 --depth 64 --height 64 --width 64 --crop_mode mixed
```

Prostate:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task05_Prostate --data_dir ./data/step1b_medical_msd/Task05_Prostate --binary --epochs 5 --batch_size 1 --max_train_samples 16 --max_val_samples 4 --depth 128 --height 128 --width 64 --crop_mode mixed
```

Pancreas:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task07_Pancreas --data_dir ./data/step1b_medical_msd/Task07_Pancreas --epochs 10 --batch_size 1 --max_train_samples 24 --max_val_samples 6 --depth 128 --height 128 --width 128 --d_model 32 --n_layers 2 --max_steps 6 --diffusion_scale 0.08 --lr 0.001 --binary --crop_mode mixed
```

---

## Practical notes

- `MSDDataset(seed=42)` aligns train/val splits between FluidVLA and U-Net.
- `Task04_Hippocampus` and `Task05_Prostate` require a specific crop.
- `Task07_Pancreas` and `Task08_HepaticVessel` are structurally challenging; modest Dice scores are expected on small subsets.
- `infer_msd.py` generates slice, multislice, 3D PNG, and interactive HTML renders.

---

## Output files

```text
checkpoints/fluidvla/<Task>/best_fluidvla.pt
checkpoints/unet3d/<Task>/best_unet3d_tiny.pt
checkpoints/unet3d/<Task>/best_unet3d_std.pt
inference_outputs/<Task>/fluidvla_<case>_slice.png
inference_outputs/<Task>/fluidvla_<case>_multislice.png
inference_outputs/<Task>/fluidvla_<case>_3d.png
inference_outputs/<Task>/fluidvla_<case>_3d.html
```


## crop_mode recommendations

- `center`: BrainTumour, Heart, Liver, Lung, Spleen
- `mixed`: Hippocampus, Prostate, Pancreas, HepaticVessel, Colon
- `foreground`: useful for debug / targeted validation

Pancreas example:

```bat
python experiments/step1b_medical_msd/train_fluidvla_msd.py --task Task07_Pancreas --data_dir ./data/step1b_medical_msd/Task07_Pancreas --epochs 10 --batch_size 1 --max_train_samples 24 --max_val_samples 6 --depth 128 --height 128 --width 128 --d_model 32 --n_layers 2 --max_steps 6 --diffusion_scale 0.08 --lr 0.001 --binary --crop_mode mixed
```
