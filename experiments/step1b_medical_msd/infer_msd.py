"""
infer_msd.py
============
Universal visual inference for all MSD tasks.
Works with both FluidVLA and U-Net from their respective checkpoints.

Generates for each model:
  - <output_dir>/<model_tag>_slice.png       -- central slice + overlay
  - <output_dir>/<model_tag>_multislice.png  -- 5 axial slices
  - <output_dir>/<model_tag>_3d.png          -- isometric surface render
  - <output_dir>/<model_tag>_3d.html         -- interactive Plotly render

Run from FluidVLA-main/ :

    # FluidVLA
    python experiments/step1b_medical_msd/infer_msd.py ^
        --task Task09_Spleen ^
        --data_dir ./data/step1b_medical_msd/Task09_Spleen ^
        --model_type fluidvla ^
        --checkpoint ./checkpoints/fluidvla/Task09_Spleen/best_fluidvla.pt ^
        --case spleen_1.nii.gz ^
        --output_dir ./inference_outputs/Task09_Spleen

    # U-Net Std
    python experiments/step1b_medical_msd/infer_msd.py ^
        --task Task09_Spleen ^
        --data_dir ./data/step1b_medical_msd/Task09_Spleen ^
        --model_type unet3d_std ^
        --checkpoint ./checkpoints/unet3d/Task09_Spleen/best_unet3d_std.pt ^
        --case spleen_1.nii.gz ^
        --output_dir ./inference_outputs/Task09_Spleen
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.step1b_medical_msd.msd_dataset import (
    get_task_meta, load_volume, load_label, center_crop_or_pad, apply_crop_mode
)


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_fluidvla(checkpoint_path: str, device: torch.device):
    from fluidvla.core import FluidBotMedical3D
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt.get("args", {})
    model = FluidBotMedical3D(
        in_channels=ckpt.get("in_channels", a.get("in_channels", 4)),
        n_classes=ckpt.get("num_classes",  a.get("num_classes", 2)),
        d_model=a.get("d_model", 32),
        n_layers=a.get("n_layers", 2),
        patch_size=a.get("patch_size", 2),
        dilations=a.get("dilations", [1, 2, 4]),
        max_steps=a.get("max_steps", 6),
        dt=a.get("dt", 0.1),
        epsilon=a.get("epsilon", 0.08),
        use_pde=not a.get("no_pde", False),
        norm_type=a.get("norm_type", "rmsnorm"),
        norm_every=a.get("norm_every", 2),
        local_memory_dhw=(a.get("local_memory_d", 4), a.get("local_memory_h", 4), a.get("local_memory_w", 4)),
        signed_diffusion=a.get("signed_diffusion", False),
        diffusion_scale=a.get("diffusion_scale", 0.08),
        stop_patience=a.get("stop_patience", 2),
        min_steps=a.get("min_steps", 3),
        stop_probe_dhw=(a.get("stop_probe_d", 4), a.get("stop_probe_h", 8), a.get("stop_probe_w", 8)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("best_val_dice", float("nan")), ckpt.get("n_params", 0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, features=32):
        super().__init__()
        f = features
        self.enc1=ConvBlock(in_channels,f); self.enc2=ConvBlock(f,f*2); self.enc3=ConvBlock(f*2,f*4)
        self.pool=nn.MaxPool3d(2); self.bottleneck=ConvBlock(f*4,f*8)
        self.up3=nn.ConvTranspose3d(f*8,f*4,2,stride=2); self.dec3=ConvBlock(f*8,f*4)
        self.up2=nn.ConvTranspose3d(f*4,f*2,2,stride=2); self.dec2=ConvBlock(f*4,f*2)
        self.up1=nn.ConvTranspose3d(f*2,f,2,stride=2);   self.dec1=ConvBlock(f*2,f)
        self.head=nn.Conv3d(f,num_classes,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        b=self.bottleneck(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.head(d1)

def load_unet3d(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Detect features from first conv weight shape
    key = next(k for k in ckpt["model"] if "enc1" in k and "weight" in k and "block.0" in k)
    features = ckpt["model"][key].shape[0]
    in_ch    = ckpt["model"][key].shape[1]
    n_cls_key = next(k for k in ckpt["model"] if "head" in k and "weight" in k)
    n_cls = ckpt["model"][n_cls_key].shape[0]
    model = UNet3D(in_channels=in_ch, num_classes=n_cls, features=features).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("best_val_dice", float("nan")), ckpt.get("n_params", 0)


MODEL_LOADERS = {
    "fluidvla":   load_fluidvla,
    "unet3d_tiny":load_unet3d,
    "unet3d_std": load_unet3d,
}


EVENT_PREFIX = "__FLUIDVLA_EVENT__"


def emit_event(event: str, **payload):
    print(f"{EVENT_PREFIX}{json.dumps({'event': event, **payload})}", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, model_type, x_np, device):
    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(x)
    if device.type == "cuda": torch.cuda.synchronize()
    lat_ms = (time.time() - t0) * 1000.0

    if model_type == "fluidvla":
        logits = out["logits"]
        info = out.get("info", [])
        steps = sum(i["steps_used"] for i in info)/max(len(info),1) if info else float("nan")
        turb  = sum(i["final_turbulence"] for i in info)/max(len(info),1) if info else float("nan")
    else:
        logits = out
        steps, turb = float("nan"), float("nan")

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred, lat_ms, steps, turb


def dice_np(pred, gt, eps=1e-6):
    inter = ((pred==1)&(gt==1)).sum()
    denom = (pred==1).sum()+(gt==1).sum()
    return (2*inter+eps)/(denom+eps)


# ── Visualizations ────────────────────────────────────────────────────────────

def _best_z(y_np):
    """Axial slice with most foreground voxels."""
    sums = y_np.sum(axis=(0,1)) if y_np.ndim == 3 else y_np.sum(axis=(0,1))
    return int(np.argmax(sums)) if sums.max() > 0 else y_np.shape[2]//2


def plot_slice(x_np, y_np, pred, lat_ms, steps, turb, label, case_name, out_path):
    z = _best_z(y_np)
    mri   = x_np[0, :, :, z]
    gt_sl = y_np[:, :, z]
    pr_sl = pred[:, :, z]
    dice  = dice_np(pred, y_np)

    extra = f"  steps={steps:.1f}  turb={turb:.4f}" if not np.isnan(steps) else ""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{case_name}  |  {label}  |  lat={lat_ms:.1f}ms  |  dice={dice:.4f}{extra}", fontsize=10)
    for ax, img, title, cmap in zip(
        axes,
        [mri, gt_sl, pr_sl, None],
        ["MRI modality 0", "Ground truth", "Prediction", "Overlay"],
        ["gray","gray","gray",None]
    ):
        if title == "Overlay":
            ax.imshow(mri, cmap="gray")
            ov = np.zeros((*mri.shape, 4))
            ov[pr_sl > 0] = [1, 0.2, 0.2, 0.5]
            ax.imshow(ov)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9); ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out_path}")


def plot_multislice(x_np, y_np, pred, lat_ms, label, case_name, out_path, n_slices=5):
    D = x_np.shape[3]
    z_slices = np.linspace(int(D*0.1), int(D*0.9), n_slices, dtype=int)
    dice = dice_np(pred, y_np)

    fig, axes = plt.subplots(n_slices, 4, figsize=(20, 4*n_slices))
    fig.suptitle(f"{case_name}  |  {label}  |  lat={lat_ms:.1f}ms  |  dice={dice:.4f}", fontsize=10)
    for row, z in enumerate(z_slices):
        mri   = x_np[0, :, :, z]
        gt_sl = y_np[:, :, z]
        pr_sl = pred[:, :, z]
        axes[row,0].imshow(mri,   cmap="gray"); axes[row,0].set_ylabel(f"z={z}", fontsize=8)
        axes[row,1].imshow(gt_sl, cmap="gray")
        axes[row,2].imshow(pr_sl, cmap="gray")
        axes[row,3].imshow(mri,   cmap="gray")
        ov = np.zeros((*mri.shape, 4)); ov[pr_sl>0] = [1,0.2,0.2,0.5]
        axes[row,3].imshow(ov)
        if row == 0:
            for col, t in enumerate(["MRI","Ground truth","Prediction","Overlay"]):
                axes[0,col].set_title(t, fontsize=9)
    for ax in axes.flat: ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out_path}")


def plot_3d_png(y_np, pred, lat_ms, case_name, label, out_path, downsample=2):
    try:
        from skimage import measure
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        print("  [skip 3D PNG] skimage not available")
        return

    ds = max(1, downsample)
    gt_ds = (y_np > 0)[::ds, ::ds, ::ds]
    pr_ds = (pred > 0)[::ds, ::ds, ::ds]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{case_name}\n{label} | lat={lat_ms:.1f}ms", fontsize=9)

    for vol, color, alpha, name in [
        (gt_ds, "green", 0.25, "Ground truth"),
        (pr_ds, "red",   0.45, "Prediction"),
    ]:
        if vol.sum() == 0: continue
        try:
            verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=0.5)
            mesh = Poly3DCollection(verts[faces], alpha=alpha)
            mesh.set_facecolor(color); mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
            ax.set_xlim(0, vol.shape[0]); ax.set_ylim(0, vol.shape[1]); ax.set_zlim(0, vol.shape[2])
        except Exception: pass

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], color="green", lw=3, label="Ground truth"),
        Line2D([0],[0], color="red",   lw=3, label="Prediction"),
    ])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out_path}")


def plot_3d_html(y_np, pred, lat_ms, steps, turb, case_name, label, out_path, downsample=2):
    try:
        import plotly.graph_objects as go
        from skimage import measure
    except ImportError:
        print("  [skip 3D HTML] plotly or skimage not available")
        return

    ds = max(1, downsample)
    gt_ds = (y_np > 0)[::ds, ::ds, ::ds]
    pr_ds = (pred > 0)[::ds, ::ds, ::ds]

    fig = go.Figure()
    for vol, color, name, opacity in [
        (gt_ds, "rgb(50,200,50)",  "Ground truth", 0.30),
        (pr_ds, "rgb(230,50,50)",  "Prediction",   0.50),
    ]:
        if vol.sum() == 0: continue
        try:
            verts, faces, _, _ = measure.marching_cubes(vol.astype(np.float32), level=0.5)
            x, y, z = verts.T; i, j, k = faces.T
            fig.add_trace(go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,
                                    color=color, opacity=opacity, name=name, showlegend=True))
        except Exception: pass

    extra = f" | steps={steps:.1f} | turb={turb:.4f}" if not np.isnan(steps) else ""
    fig.update_layout(
        title=f"<b>{case_name}</b> — {label}<br>lat={lat_ms:.1f}ms{extra}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            orientation="v",
            bgcolor="rgba(8,12,16,0.68)",
            bordercolor="rgba(255,255,255,0.16)",
            borderwidth=1,
            font=dict(color="white"),
        ),
        margin=dict(r=10, l=10, b=10, t=60),
    )
    out_html = Path(out_path).with_suffix(".html")
    fig.write_html(str(out_html))
    print(f"  Saved: {out_html}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",        required=True)
    ap.add_argument("--data_dir",    required=True)
    ap.add_argument("--model_type",  required=True, choices=list(MODEL_LOADERS.keys()))
    ap.add_argument("--checkpoint",  required=True)
    ap.add_argument("--case",        required=True, help="Filename in imagesTr/, e.g. spleen_1.nii.gz")
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--depth",  type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--width",  type=int, default=128)
    ap.add_argument("--downsample", type=int, default=2, help="3D surface downsampling factor")
    ap.add_argument("--crop_mode", choices=["center", "foreground"], default="center")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop = (args.depth, args.height, args.width)
    pipeline_t0 = time.perf_counter()
    phase_timings = {}

    in_ch, n_cls_full, _, _ = get_task_meta(args.data_dir)
    is_multimodal = in_ch > 1

    # Load case
    from pathlib import Path as P
    img_path = str(P(args.data_dir) / "imagesTr" / args.case)
    lbl_path = str(P(args.data_dir) / "labelsTr" / args.case)
    emit_event("progress", phase="load_case", label="Loading volume", progress=8)
    phase_t0 = time.perf_counter()
    print(f"\n[Loading]  {args.case}  (in_ch={in_ch})")
    x_np = center_crop_or_pad(load_volume(img_path, is_multimodal), crop)
    y_np = center_crop_or_pad(load_label(lbl_path, binary=True),    crop)
    phase_timings["load_case_ms"] = (time.perf_counter() - phase_t0) * 1000.0

    # Load model
    loader = MODEL_LOADERS[args.model_type]
    emit_event("progress", phase="load_model", label="Loading checkpoint", progress=24)
    phase_t0 = time.perf_counter()
    print(f"[Model]    {args.model_type}  ->  {args.checkpoint}")
    model, best_val_dice, n_params = loader(args.checkpoint, device)
    phase_timings["load_model_ms"] = (time.perf_counter() - phase_t0) * 1000.0
    label = f"{args.model_type}  |  val_dice={best_val_dice:.4f}  |  {n_params:,} params"

    # Inference
    emit_event("progress", phase="run_model", label="Running model inference", progress=42)
    phase_t0 = time.perf_counter()
    pred, lat_ms, steps, turb = run_inference(model, args.model_type, x_np, device)
    phase_timings["run_model_ms"] = (time.perf_counter() - phase_t0) * 1000.0
    inf_dice = dice_np(pred, y_np)
    case_stem = args.case.replace(".nii.gz", "")
    prefix = str(out_dir / f"{args.model_type}_{case_stem}")
    emit_event(
        "metrics",
        model_latency_ms=lat_ms,
        inference_dice=float(inf_dice),
        steps=None if np.isnan(steps) else float(steps),
        final_turbulence=None if np.isnan(turb) else float(turb),
    )

    print(f"\n[Results]  lat={lat_ms:.1f}ms  dice={inf_dice:.4f}"
          f"  steps={steps:.1f}  turb={turb:.4f}")

    # Generate all outputs
    emit_event("progress", phase="render_slice", label="Rendering central slice", progress=58)
    phase_t0 = time.perf_counter()
    plot_slice(x_np, y_np, pred, lat_ms, steps, turb, label, args.case,
               f"{prefix}_slice.png")
    phase_timings["render_slice_ms"] = (time.perf_counter() - phase_t0) * 1000.0

    emit_event("progress", phase="render_multislice", label="Rendering multi-slice", progress=70)
    phase_t0 = time.perf_counter()
    plot_multislice(x_np, y_np, pred, lat_ms, label, args.case,
                    f"{prefix}_multislice.png")
    phase_timings["render_multislice_ms"] = (time.perf_counter() - phase_t0) * 1000.0

    emit_event("progress", phase="render_3d_png", label="Rendering 3D PNG", progress=82)
    phase_t0 = time.perf_counter()
    plot_3d_png(y_np, pred, lat_ms, args.case, label, f"{prefix}_3d.png",
                downsample=args.downsample)
    phase_timings["render_3d_png_ms"] = (time.perf_counter() - phase_t0) * 1000.0

    emit_event("progress", phase="render_3d_html", label="Rendering interactive 3D", progress=93)
    phase_t0 = time.perf_counter()
    plot_3d_html(y_np, pred, lat_ms, steps, turb, args.case, label,
                 f"{prefix}_3d.html", downsample=args.downsample)
    phase_timings["render_3d_html_ms"] = (time.perf_counter() - phase_t0) * 1000.0

    emit_event("progress", phase="write_json", label="Finalizing", progress=98)
    phase_t0 = time.perf_counter()
    result = {
        "task": args.task, "case": args.case,
        "model_type": args.model_type, "checkpoint": args.checkpoint,
        "best_val_dice": best_val_dice, "inference_dice": float(inf_dice),
        "latency_ms": lat_ms,
        "model_latency_ms": lat_ms,
        "pipeline_latency_ms": None,
        "steps": steps, "final_turbulence": turb,
        "n_params": n_params,
        "phase_timings_ms": phase_timings,
    }
    phase_timings["write_json_ms"] = (time.perf_counter() - phase_t0) * 1000.0
    pipeline_ms = (time.perf_counter() - pipeline_t0) * 1000.0
    result["pipeline_latency_ms"] = pipeline_ms
    result["phase_timings_ms"] = phase_timings
    json_path = f"{prefix}_result.json"
    Path(json_path).write_text(json.dumps(result, indent=2))
    print(f"\n[JSON]     {json_path}")
    print(json.dumps(result, indent=2))
    emit_event("summary", progress=100, label="Done", result=result)


if __name__ == "__main__":
    main()
