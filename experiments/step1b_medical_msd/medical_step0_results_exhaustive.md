# FluidVLA - Medical Step 0
## Exhaustive transcription of results from the research log

Source: `res.pdf`

---

## Page 1 - Cover / headline metrics

# RESEARCH LOG

## FluidVLA
### Medical Step 0
### 3D PDE-based segmentation on real multimodal MRI volumes
### MSD Task01_BrainTumour - Binary tumour segmentation

- **0.017M** Parameters
- **44ms** Inference latency
- **0.9177** Best Val Dice (PDE ON)

---

## Page 2 - Objective and setup

# OBJECTIVE & SETUP

**Question studied:**
Can the FluidVLA PDE core be extended to real 3D medical volumes, and does calibrated diffusion provide measurable benefit over a no-PDE control?

### Dataset
- **MSD Task01_BrainTumour**
- **4 MRI modalities** - shape **240 x 240 x 155 x 4**
- Labels: **0 (background) + 1/2/3 (tumour)**
- Binary simplification: **background vs tumour > 0**

### Benchmark scope
- **16 train / 4 val samples**
- **5 epochs**, **AdamW** optimizer
- **CE + Soft Dice** loss
- Crop size: **128^3**

### Model - FluidBotMedical3D
- `d_model = 32`
- `n_layers = 2`
- `max_steps = 6`
- Parameters: **16,632 (0.017M)**
- Positive diffusion: **sigma(p) x diffusion_scale**

### Stress test criteria
- Volumetric spatial reasoning
- Multimodal input fusion (4 channels)
- Lesion boundary localization
- Memory-efficient 3D processing

---

## Page 3 - Calibration bug: root cause and fix

# THE CALIBRATION BUG - ROOT CAUSE & FIX

## Before fix
- `diffusion_scale` not wired.
- The parameter only applied in the `signed_diffusion` path.
- With `signed_diffusion=False` (default), changing **0.25 -> 0.05 -> 0.01** produced identical results.
- Effective diffusion: **uncontrolled, always same strength**.

## After fix
- Sigmoid-bounded positive diffusion:
  - `return diffusion_scale x sigmoid(p)`
- Coefficient mean now scales correctly:
  - `scale=0.05 -> mean ~= 0.025`
  - `scale=0.08 -> mean ~= 0.040`
  - `scale=0.01 -> mean ~= 0.005`

## Ablation: effect of the bug (8 train / 2 val)

| Config | Val Dice | Status |
|---|---:|---|
| PDE OFF | 0.8490 | Baseline |
| PDE ON - BEFORE fix | 0.3846 | Broken |
| PDE ON - diffusion_scale=0.05 (fixed) | 0.8756 | Competitive |
| PDE ON - diffusion_scale=0.08 (fixed) | 0.8867 | Better |

---

## Page 4 - Main results: 16 train / 4 val / 5 epochs

# RESULTS - 16 TRAIN / 4 VAL - 5 EPOCHS

## PDE OFF
- **Val Dice:** 0.9129
- **Latency:** ~60ms
- **VRAM:** ~632 MiB

## PDE ON (scale = 0.08)
- **Val Dice:** 0.9177
- **Latency:** ~89ms
- **VRAM:** ~632 MiB
- **Winner**

### Reported delta
- **Delta Dice:** +0.0048
- **Delta Latency:** +29ms
- **Model size:** 0.017M params
- **Tradeoff:** real and measurable

---

## Page 5 - Visual inference, central slice

# VISUAL INFERENCE - CENTRAL SLICE | z = 50 | BRATS_001.nii.gz

### Reported metrics for this case
- **Latency:** 44.16 ms
- **Steps used:** 6 / 6
- **Final turbulence:** 0.2218
- **Case:** BRATS_001

### Figure content
The page shows a 4-panel visualization for the central slice:
1. **MRI modality 0** at `z=50`
2. **Ground truth** mask
3. **Prediction** mask
4. **Overlay** of prediction on the MRI slice

### Reported interpretation
- Core tumour mass correctly localized.
- Minor false positives at slice edges.
- Prediction slightly undershoots ground truth boundary at top.

---

## Page 6 - Visual inference, 5 axial slices

# VISUAL INFERENCE - 5 AXIAL SLICES | z = 16 to z = 89 | BRATS_001.nii.gz

### Figure content
This page presents a multi-row comparison across several axial slices. Each row includes:
- MRI slice
- Ground truth
- Prediction
- Overlay

The displayed slice positions are visually organized around the following indices:
- `z=16`
- `z=34`
- `z=52`
- `z=70`
- `z=89`

### Reported interpretation
- `z=34 / 52 / 70`: strong overlap
- `z=16`: early tumour onset detected
- `z=89`: clean negative (no false positives)

---

## Page 7 - Visual inference, 3D tumour surfaces

# VISUAL INFERENCE - 3D TUMOUR SURFACES | BRATS_001.nii.gz

### Figure content
This page shows a 3D surface rendering comparison between:
- **Green = Ground truth**
- **Red = Prediction**

The embedded plot title also indicates approximately:
- `latency = 279.22 ms`
- `steps = 6.00`
- `final_turb = 0.2218`

### Reported interpretation
- **Green = Ground truth** - main mass well bounded
- **Red = Prediction** - correct core + scattered false positives at low-z slices

---

## Page 8 - Honest research scorecard

# HONEST RESEARCH SCORECARD

## Proven
- Real 3D multimodal NIfTI volumes processed end-to-end
- Binary tumour segmentation learns and converges
- Calibration bug identified: `diffusion_scale` was unwired
- After fix: PDE ON slightly outperforms PDE OFF (**0.9177 vs 0.9129**)
- FluidVLA paradigm viable for volumetric medical segmentation
- **0.017M** parameter model achieves **>91% Dice** at **44ms** inference

## Not proven yet
- Clinical relevance or deployment readiness
- Superiority over U-Net / MONAI baselines
- Multi-class BraTS subregion segmentation
- Real adaptive compute savings at inference
- Robustness on large-scale BrainTumour splits

---

## Page 9 - Conclusion and next steps

# CONCLUSION & NEXT STEPS

## Main conclusion
**Medical Step 0 is a success.**

The stated reason is not that FluidVLA is already a clinical model, but that the **3D PDE paradigm can be ported to real multimodal NIfTI data, debugged, calibrated, and made competitive against its own no-PDE control**. The document concludes that the **reaction-diffusion core is alive in 3D medical space**.

## Immediate next steps

### 01 - 3D U-Net baseline
Small reference model to establish an honest external comparison point.

### 02 - Multi-class segmentation
Full BraTS 3-class (**TC / WT / ET**) instead of binary - the real benchmark.

### 03 - Adaptive compute at inference
Validate that simpler volumes exit early; measure compute savings per case.

### 04 - Larger training splits
**100+ samples** to reduce noise and strengthen the Dice signal.

---

## Page 10 - Baseline comparison: FluidVLA vs 3D U-Net

# BASELINE COMPARISON - FluidVLA vs 3D U-Net

**Shared setup:**
- Same config
- **16 train / 4 val / 5 epochs**
- Binary BrainTumour
- CUDA

## Best FluidVLA PDE ON
- **Configuration:** `scale=0.08` - reaction-diffusion
- **Val Dice:** 0.9177
- **Parameters:** 16,632
- **Latency:** ~44ms (GPU)
- **VRAM:** ~632 MiB

## UNet3D_Tiny
- **Configuration:** `features=4` - ~iso-param
- **Val Dice:** 0.8494
- **Parameters:** 88,278
- **Latency:** 272ms (CPU)
- **VRAM:** ~326 MiB

## UNet3D_Std
- **Configuration:** `features=32` - standard
- **Val Dice:** 0.8233
- **Parameters:** 5,603,746
- **Latency:** 2704ms (CPU)
- **VRAM:** ~1891 MiB

## Reported takeaway
- FluidVLA outperforms `UNet_Std` by **+0.0944 Dice**
- With **340x fewer parameters**
- The document explicitly notes that this is a **partial advantage due to the data-limited regime (16 samples)** and that it must be **confirmed on 100+ samples**

---

## Page 11 - Visual comparison, central slice

# VISUAL COMPARISON - CENTRAL SLICE | BRATS_001.nii.gz

### Models shown

#### FluidVLA PDE ON
- Dice **0.9177**
- **44ms**

#### UNet3D_Tiny
- Dice **0.8750**
- **207ms**

#### UNet3D_Std
- Dice **0.8855**
- **100ms**

### Figure content
The page compares three central-slice visual outputs, each with MRI image, ground truth, prediction, and overlay. The layout visually suggests:
- FluidVLA predicts a mask that is closer to the irregular tumour contour.
- Both U-Net outputs look smoother and more regularized.

---

## Page 12 - Visual comparison, multi-slice U-Net vs U-Net

# VISUAL COMPARISON - MULTI-SLICE | UNet3D_Tiny vs UNet3D_Std

### Figure content
This page shows side-by-side multi-slice visual comparisons for:
- **Tiny**
- **Std**

Each side includes multiple MRI slices, ground-truth masks, prediction masks, and overlays.

### Reported interpretation
- Both U-Nets produce smooth but morphologically imprecise masks.
- They lose irregular tumour extensions visible in ground truth.

---

# Consolidated exhaustive results summary

## 1. Task and problem framing
This experiment evaluates whether the FluidVLA PDE core can be transferred from earlier research contexts to **real 3D multimodal medical MRI segmentation** on **MSD Task01_BrainTumour**, using a **binary tumour segmentation** objective.

## 2. Data and preprocessing
- 4 MRI modalities per case
- Volume shape: **240 x 240 x 155 x 4**
- Original labels: **0, 1, 2, 3**
- Binary target used: **tumour if label > 0**, otherwise background
- Crop size: **128^3**

## 3. Training protocol
- Split: **16 training / 4 validation**
- **5 epochs**
- **AdamW** optimizer
- **Cross-Entropy + Soft Dice** loss

## 4. Model under test
**FluidBotMedical3D** with:
- `d_model = 32`
- `n_layers = 2`
- `max_steps = 6`
- **16,632 parameters** total (**0.017M**)

The diffusion mechanism is described as **positive diffusion** with coefficient form:
- `sigma(p) x diffusion_scale`

## 5. Calibration bug and correction
A key methodological result of the report is the identification of a calibration bug:
- `diffusion_scale` was not actually connected in the default `signed_diffusion=False` path.
- Therefore multiple nominal scale values produced the same effective diffusion.
- This made earlier PDE-on results invalid as a calibrated comparison.

After the fix:
- the implementation became `diffusion_scale x sigmoid(p)`
- mean diffusion coefficients scaled correctly with the chosen `diffusion_scale`

This correction materially changed outcomes:
- broken PDE-on run before fix: **0.3846 Val Dice**
- calibrated PDE-on after fix at `scale=0.05`: **0.8756**
- calibrated PDE-on after fix at `scale=0.08`: **0.8867**
- PDE-off baseline in that ablation: **0.8490**

## 6. Main controlled comparison: PDE OFF vs PDE ON
Under the main setting (**16 train / 4 val / 5 epochs**):

### PDE OFF
- **Val Dice:** 0.9129
- **Latency:** ~60ms
- **VRAM:** ~632 MiB

### PDE ON (`scale=0.08`)
- **Val Dice:** 0.9177
- **Latency:** ~89ms
- **VRAM:** ~632 MiB

### Net effect
- **Accuracy gain:** +0.0048 Dice
- **Latency cost:** +29ms
- Memory use stays essentially unchanged in the report.

This is presented as a **real but modest measurable gain** from the PDE mechanism once properly calibrated.

## 7. Reported inference behaviour on BRATS_001
For the showcased central slice:
- **Latency:** 44.16 ms
- **Steps used:** 6 / 6
- **Final turbulence:** 0.2218

Qualitative findings:
- tumour core is correctly localized
- some false positives remain at slice edges
- upper boundary is slightly under-segmented

Across multiple slices:
- early tumour onset is detected at `z=16`
- strong overlap is observed at `z=34`, `z=52`, and `z=70`
- no false positives are reported at `z=89`

In 3D:
- main tumour bulk is bounded correctly
- scattered false positives remain in lower-z regions

## 8. Claimed proven points
The report explicitly claims the following are **proven** within the scope of this experiment:
- end-to-end processing of real multimodal 3D NIfTI volumes
- successful convergence on binary tumour segmentation
- detection and correction of the diffusion calibration bug
- slight but measurable PDE advantage over no-PDE control
- viability of the FluidVLA paradigm for volumetric medical segmentation
- **>91% Dice** with a **0.017M parameter** model at about **44ms** inference

## 9. Claimed unproven points
The report also explicitly states that the following are **not proven yet**:
- clinical readiness
- superiority over broader U-Net / MONAI baselines
- multi-class BraTS subregion segmentation
- genuine adaptive compute savings at inference
- robustness on larger train/validation regimes

## 10. Baseline comparison against 3D U-Nets
The later part of the report adds an external comparison.

### FluidVLA PDE ON
- **Val Dice:** 0.9177
- **Params:** 16,632
- **Latency:** ~44ms (GPU)
- **VRAM:** ~632 MiB

### UNet3D_Tiny
- **Val Dice:** 0.8494
- **Params:** 88,278
- **Latency:** 272ms (CPU)
- **VRAM:** ~326 MiB

### UNet3D_Std
- **Val Dice:** 0.8233
- **Params:** 5,603,746
- **Latency:** 2704ms (CPU)
- **VRAM:** ~1891 MiB

### Important methodological caution
The comparison is not hardware-matched in the page text:
- FluidVLA latency is reported on **GPU**
- U-Net latencies are reported on **CPU**

So the Dice comparison is informative, but the latency comparison is not fully apples-to-apples from the information shown in the document.

## 11. Qualitative comparison against U-Nets
The visual comparison pages claim:
- FluidVLA better preserves irregular tumour morphology
- U-Net masks are smoother but less precise morphologically
- U-Nets tend to lose irregular extensions visible in ground truth

## 12. Final conclusion stated by the report
The final conclusion is that **Medical Step 0 is a success**, not because the system is ready for clinical use, but because the FluidVLA PDE core has now been shown to:
- transfer to real 3D medical data,
- be debugged and calibrated in that context,
- and slightly outperform its own no-PDE control.

## 13. Next steps stated by the report
1. Add a small **3D U-Net baseline** for honest external reference.
2. Move to **full BraTS multi-class segmentation** (`TC / WT / ET`).
3. Validate **adaptive compute** and early-exit savings per case.
4. Scale to **100+ samples** to reduce noise and better validate the Dice signal.

---

# Clean extraction of all key numerical results

## Core experiment
- Parameters: **16,632** (**0.017M**)
- Best validation Dice (PDE ON): **0.9177**
- Inference latency headline: **44ms**

## Main comparison
- PDE OFF Dice: **0.9129**
- PDE ON Dice: **0.9177**
- Dice gain: **+0.0048**
- PDE OFF latency: **~60ms**
- PDE ON latency: **~89ms**
- Latency delta: **+29ms**
- VRAM for both main settings: **~632 MiB**

## Bug ablation
- PDE OFF: **0.8490**
- PDE ON before fix: **0.3846**
- PDE ON fixed, scale=0.05: **0.8756**
- PDE ON fixed, scale=0.08: **0.8867**

## Example case BRATS_001
- Central-slice latency: **44.16 ms**
- Steps used: **6 / 6**
- Final turbulence: **0.2218**
- 3D surface latency shown in figure: **279.22 ms**

## External baseline page
- FluidVLA PDE ON Dice: **0.9177**
- FluidVLA params: **16,632**
- FluidVLA latency: **~44ms (GPU)**
- FluidVLA VRAM: **~632 MiB**

- UNet3D_Tiny Dice: **0.8494**
- UNet3D_Tiny params: **88,278**
- UNet3D_Tiny latency: **272ms (CPU)**
- UNet3D_Tiny VRAM: **~326 MiB**

- UNet3D_Std Dice: **0.8233**
- UNet3D_Std params: **5,603,746**
- UNet3D_Std latency: **2704ms (CPU)**
- UNet3D_Std VRAM: **~1891 MiB**

## Visual comparison page figures
- FluidVLA PDE ON: **Dice 0.9177 | 44ms**
- UNet3D_Tiny: **Dice 0.8750 | 207ms**
- UNet3D_Std: **Dice 0.8855 | 100ms**

---

# Notes on fidelity
This markdown is an exhaustive structured transcription of the text and visually readable annotations from the PDF, including figure captions and page-level interpretations. Where a figure contained embedded text visible in the page image, it has been incorporated when legible.
