"""
msd_dataset.py
==============
Universal NIfTI dataset loader for all 10 MSD tasks.

Handles the two volume formats present across MSD tasks:
  - 3D mono-modal:  (H, W, D)       -> (1, D, H, W)
  - 3D multi-modal: (H, W, D, C)    -> (C, D, H, W)   [BrainTumour, Prostate]

Adds crop modes for difficult small-structure tasks:
  - center      : old behaviour, deterministic center crop
  - foreground  : crop centered on non-zero label bbox (fallback center)
  - mixed       : foreground-biased training crop, center fallback

All tasks support binary mode (foreground vs background).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

TASK_META = {
    "Task01_BrainTumour":   (4, 4, (128, 128, 128), "MRI 4-modality brain tumour"),
    "Task02_Heart":         (1, 2, (128, 128, 128), "MRI cardiac left-atrium"),
    "Task03_Liver":         (1, 3, (128, 128, 128), "CT liver + tumour"),
    "Task04_Hippocampus":   (1, 3, (128, 128, 128), "MRI hippocampus"),
    "Task05_Prostate":      (2, 3, (128, 128, 128), "MRI prostate zones"),
    "Task06_Lung":          (1, 2, (128, 128, 128), "CT lung nodule"),
    "Task07_Pancreas":      (1, 3, (128, 128, 128), "CT pancreas + tumour"),
    "Task08_HepaticVessel": (1, 3, (128, 128, 128), "CT hepatic vessel + tumour"),
    "Task09_Spleen":        (1, 2, (128, 128, 128), "CT spleen"),
    "Task10_Colon":         (1, 2, (128, 128, 128), "CT colon cancer"),
}

def get_task_meta(data_dir: str):
    name = Path(data_dir).name
    for key, meta in TASK_META.items():
        if key in name or name in key:
            return meta
    raise ValueError(f"Cannot infer task from '{name}'. Expected one of: {list(TASK_META.keys())}")

def load_nifti(path: str) -> np.ndarray:
    data = nib.load(path).get_fdata().astype(np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

def normalize_modality(x: np.ndarray) -> np.ndarray:
    mask = x != 0
    if mask.any():
        x = (x - x[mask].mean()) / (x[mask].std() + 1e-6)
    return x

def center_crop_or_pad(x: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    td, th, tw = target
    if x.ndim == 4:
        c, d, h, w = x.shape
        out = np.zeros((c, td, th, tw), dtype=x.dtype)
        sd = max((d-td)//2, 0); sh = max((h-th)//2, 0); sw = max((w-tw)//2, 0)
        dd = max((td-d)//2, 0); dh = max((th-h)//2, 0); dw = max((tw-w)//2, 0)
        cd = min(d, td); ch = min(h, th); cw = min(w, tw)
        out[:, dd:dd+cd, dh:dh+ch, dw:dw+cw] = x[:, sd:sd+cd, sh:sh+ch, sw:sw+cw]
        return out
    d, h, w = x.shape
    out = np.zeros((td, th, tw), dtype=x.dtype)
    sd = max((d-td)//2, 0); sh = max((h-th)//2, 0); sw = max((w-tw)//2, 0)
    dd = max((td-d)//2, 0); dh = max((th-h)//2, 0); dw = max((tw-w)//2, 0)
    cd = min(d, td); ch = min(h, th); cw = min(w, tw)
    out[dd:dd+cd, dh:dh+ch, dw:dw+cw] = x[sd:sd+cd, sh:sh+ch, sw:sw+cw]
    return out

def load_volume(img_path: str, is_multimodal: bool) -> np.ndarray:
    x = load_nifti(img_path)
    if is_multimodal and x.ndim == 4 and x.shape[-1] <= 8:
        x = np.moveaxis(x, -1, 0)
        x = np.transpose(x, (0, 3, 1, 2))
    elif x.ndim == 3:
        x = np.transpose(x, (2, 0, 1))[None]
    elif x.ndim == 4 and not is_multimodal:
        x = np.transpose(x[:, :, :, 0], (2, 0, 1))[None]
    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")
    for c in range(x.shape[0]):
        x[c] = normalize_modality(x[c])
    return x

def load_label(lbl_path: str, binary: bool) -> np.ndarray:
    y = load_nifti(lbl_path).astype(np.int64)
    if y.ndim == 3:
        y = np.transpose(y, (2, 0, 1))
    elif y.ndim == 4:
        y = np.transpose(y[:, :, :, 0], (2, 0, 1))
    if binary:
        y = (y > 0).astype(np.int64)
    return y

def _crop_or_pad_from_start(x: np.ndarray, target: Tuple[int, int, int], start: Tuple[int, int, int]) -> np.ndarray:
    td, th, tw = target
    sd, sh, sw = start
    if x.ndim == 4:
        c, d, h, w = x.shape
        out = np.zeros((c, td, th, tw), dtype=x.dtype)
        src_d0, src_h0, src_w0 = max(sd, 0), max(sh, 0), max(sw, 0)
        src_d1, src_h1, src_w1 = min(sd + td, d), min(sh + th, h), min(sw + tw, w)
        dst_d0, dst_h0, dst_w0 = max(-sd, 0), max(-sh, 0), max(-sw, 0)
        cd, ch, cw = max(src_d1 - src_d0, 0), max(src_h1 - src_h0, 0), max(src_w1 - src_w0, 0)
        if cd > 0 and ch > 0 and cw > 0:
            out[:, dst_d0:dst_d0+cd, dst_h0:dst_h0+ch, dst_w0:dst_w0+cw] = x[:, src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
        return out
    d, h, w = x.shape
    out = np.zeros((td, th, tw), dtype=x.dtype)
    src_d0, src_h0, src_w0 = max(sd, 0), max(sh, 0), max(sw, 0)
    src_d1, src_h1, src_w1 = min(sd + td, d), min(sh + th, h), min(sw + tw, w)
    dst_d0, dst_h0, dst_w0 = max(-sd, 0), max(-sh, 0), max(-sw, 0)
    cd, ch, cw = max(src_d1 - src_d0, 0), max(src_h1 - src_h0, 0), max(src_w1 - src_w0, 0)
    if cd > 0 and ch > 0 and cw > 0:
        out[dst_d0:dst_d0+cd, dst_h0:dst_h0+ch, dst_w0:dst_w0+cw] = x[src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
    return out

def foreground_crop_or_pad(x: np.ndarray, y: np.ndarray, target: Tuple[int, int, int], jitter_frac: float = 0.0, rng: Optional[np.random.RandomState] = None):
    coords = np.argwhere(y > 0)
    if coords.size == 0:
        return center_crop_or_pad(x, target), center_crop_or_pad(y, target)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = ((mins + maxs) / 2.0).astype(np.float32)
    if rng is not None and jitter_frac > 0:
        center += rng.uniform(-np.array(target) * jitter_frac, np.array(target) * jitter_frac)
    start = tuple(int(round(center[i] - target[i] / 2)) for i in range(3))
    return _crop_or_pad_from_start(x, target, start), _crop_or_pad_from_start(y, target, start)

def apply_crop_mode(x: np.ndarray, y: np.ndarray, target: Tuple[int, int, int], crop_mode: str = "center", split: str = "train", rng: Optional[np.random.RandomState] = None, fg_prob: float = 0.75, jitter_frac: float = 0.10):
    mode = (crop_mode or "center").lower()
    if mode == "center":
        return center_crop_or_pad(x, target), center_crop_or_pad(y, target)
    if mode == "foreground":
        return foreground_crop_or_pad(x, y, target, jitter_frac=jitter_frac if split == "train" else 0.0, rng=rng)
    if mode == "mixed":
        if split == "train" and rng is not None and rng.rand() < fg_prob:
            return foreground_crop_or_pad(x, y, target, jitter_frac=jitter_frac, rng=rng)
        if split != "train" and np.any(y > 0):
            return foreground_crop_or_pad(x, y, target, jitter_frac=0.0, rng=None)
        return center_crop_or_pad(x, target), center_crop_or_pad(y, target)
    raise ValueError(f"Unknown crop_mode: {crop_mode}")

class MSDDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", val_ratio: float = 0.2, max_samples: Optional[int] = None, crop_shape: Optional[Tuple[int, int, int]] = None, binary: bool = True, seed: int = 42, crop_mode: str = "center", foreground_prob: float = 0.75, jitter_frac: float = 0.10):
        self.data_dir = Path(data_dir)
        self.binary = binary
        self.split = split
        self.seed = seed
        self.crop_mode = crop_mode
        self.foreground_prob = foreground_prob
        self.jitter_frac = jitter_frac
        in_ch, n_cls, default_crop, _desc = get_task_meta(str(data_dir))
        self.in_channels = in_ch
        self.n_classes = 2 if binary else n_cls
        self.crop_shape = crop_shape or default_crop
        self.is_multimodal = (in_ch > 1)
        img_dir = self.data_dir / "imagesTr"
        lbl_dir = self.data_dir / "labelsTr"
        pairs: List[Tuple[Path, Path]] = []
        for img in sorted(img_dir.glob("*.nii.gz")):
            if img.name.startswith("._"):
                continue
            lbl = lbl_dir / img.name
            if lbl.exists():
                pairs.append((img, lbl))
        rng = np.random.RandomState(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        if split == "train":
            pairs = pairs[n_val:]
        elif split == "val":
            pairs = pairs[:n_val]
        else:
            raise ValueError("split must be 'train' or 'val'")
        if max_samples is not None:
            pairs = pairs[:max_samples]
        self.pairs = pairs
        print(f"[MSDDataset] {self.data_dir.name} | {split} | {len(pairs)} samples | in_ch={in_ch} | n_classes={self.n_classes} | crop={self.crop_shape} | crop_mode={self.crop_mode}")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        x = load_volume(str(img_path), self.is_multimodal)
        y = load_label(str(lbl_path), self.binary)
        rng = np.random.RandomState(self.seed * 1000003 + idx * 9176 + (0 if self.split == "train" else 1))
        x, y = apply_crop_mode(x, y, self.crop_shape, crop_mode=self.crop_mode, split=self.split, rng=rng, fg_prob=self.foreground_prob, jitter_frac=self.jitter_frac)
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()
