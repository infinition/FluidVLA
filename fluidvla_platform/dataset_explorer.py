from __future__ import annotations

import base64
import gzip
import io
import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    import nibabel as nib
except ImportError:
    nib = None


ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("path must stay inside the repository root") from exc
    return path


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _unique_existing_dirs(paths: list[str]) -> list[Path]:
    results: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        try:
            path = _resolve_repo_path(raw_path)
        except ValueError:
            continue
        key = str(path)
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        results.append(path)
    return results


def _discover_child_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.is_dir():
        return []
    return [child for child in sorted(base_dir.iterdir()) if child.is_dir()]


def _dataset_dir_candidates() -> list[Path]:
    candidates = _unique_existing_dirs([
        "data/step0_mnist",
        "data/step1_video",
        "data/step1b_medical_msd",
        "data/step2_sim",
        "data/step2a_synthetic",
        "data/step2c_isaac",
        "data/step2d_so101_urdf",
        "data/step3_lerobot",
    ])
    data_root = ROOT / "data"
    if data_root.is_dir():
        candidates.extend(_discover_child_dirs(data_root))
    candidates.extend(_discover_child_dirs(ROOT / "data" / "step1b_medical_msd"))
    return _unique_existing_dirs([str(path) for path in candidates])


def _checkpoint_dir_candidates() -> list[Path]:
    candidates = _unique_existing_dirs([
        "checkpoints/step0",
        "checkpoints/step1",
        "checkpoints/step2_sim",
        "checkpoints/step2a_synthetic",
        "checkpoints/step2c_isaac",
        "checkpoints/step2d_so101_urdf",
        "checkpoints/step3_lerobot",
        "checkpoints/fluidvla",
        "checkpoints/unet3d",
        "checkpoints",
    ])
    ckpt_root = ROOT / "checkpoints"
    if ckpt_root.is_dir():
        candidates.extend(_discover_child_dirs(ckpt_root))
    return _unique_existing_dirs([str(path) for path in candidates])


def _collect_dataset_files(selected_dir: Path) -> list[Path]:
    patterns = [
        "*.npz",
        "*.npy",
        "*.nii",
        "*.nii.gz",
        "*ubyte*",
        "data_batch_*",
        "test_batch*",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(selected_dir.glob(pattern)))
    if files:
        return sorted(set(files))
    for pattern in patterns:
        files.extend(sorted(selected_dir.rglob(pattern)))
    return sorted(set(files))


def _collect_checkpoint_files(selected_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("*.pt", "*.pth", "*.json"):
        files.extend(sorted(selected_dir.glob(pattern)))
        files.extend(sorted(selected_dir.rglob(pattern)))
    return sorted(set(files))


def _format_dir_entry(path: Path, kind: str) -> dict:
    collector = _collect_dataset_files if kind == "dataset" else _collect_checkpoint_files
    files = collector(path)
    return {
        "path": str(path),
        "relative_path": _relpath(path),
        "name": path.name,
        "kind": kind,
        "file_count": len(files),
        "preview": [_relpath(item) for item in files[:6]],
    }


def build_explorer_overview() -> dict:
    dataset_dirs = [_format_dir_entry(path, "dataset") for path in _dataset_dir_candidates()]
    checkpoint_dirs = [_format_dir_entry(path, "checkpoint") for path in _checkpoint_dir_candidates()]
    default_dataset = next((item["path"] for item in dataset_dirs if item["file_count"] > 0), dataset_dirs[0]["path"] if dataset_dirs else None)
    default_checkpoint = next((item["path"] for item in checkpoint_dirs if item["file_count"] > 0), checkpoint_dirs[0]["path"] if checkpoint_dirs else None)
    return {
        "dataset_dirs": dataset_dirs,
        "checkpoint_dirs": checkpoint_dirs,
        "default_dataset_dir": default_dataset,
        "default_checkpoint_dir": default_checkpoint,
    }


def list_explorer_dir(kind: str, directory: str) -> dict:
    if kind not in {"dataset", "checkpoint"}:
        raise ValueError("kind must be dataset or checkpoint")
    resolved_dir = _resolve_repo_path(directory)
    if not resolved_dir.is_dir():
        raise ValueError(f"directory not found: {resolved_dir}")
    collector = _collect_dataset_files if kind == "dataset" else _collect_checkpoint_files
    files = collector(resolved_dir)
    return {
        "kind": kind,
        "directory": _format_dir_entry(resolved_dir, kind),
        "files": [_format_file_entry(path, resolved_dir, kind) for path in files],
    }


def _format_file_entry(path: Path, root: Path, kind: str) -> dict:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        file_type = "nii.gz"
    else:
        file_type = path.suffix.lower().lstrip(".") or "file"
    category = file_type
    if file_type in {"npz", "npy", "nii", "nii.gz"} or "ubyte" in name or "batch" in name:
        category = "dataset"
    if file_type in {"pt", "pth"}:
        category = "checkpoint"
    if file_type == "json":
        category = "log"
    return {
        "path": str(path),
        "relative_path": _relpath(path),
        "display_name": _relative_display(path, root),
        "file_type": file_type,
        "category": category,
        "kind": kind,
        "size_kb": round(path.stat().st_size / 1024, 1),
    }


def _relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _read_idx(filename: Path) -> np.ndarray:
    open_fn = gzip.open if str(filename).endswith(".gz") else open
    with open_fn(filename, "rb") as handle:
        _, _, dims = struct.unpack(">HBB", handle.read(4))
        shape = tuple(struct.unpack(">I", handle.read(4))[0] for _ in range(dims))
        return np.frombuffer(handle.read(), dtype=np.uint8).reshape(shape)


def _load_cifar_batch(file_path: Path):
    with file_path.open("rb") as handle:
        data_dict = pickle.load(handle, encoding="bytes")
    data = data_dict[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data_dict[b"labels"]
    return data, labels


def _load_cifar_meta(file_path: Path) -> list[str]:
    with file_path.open("rb") as handle:
        data_dict = pickle.load(handle, encoding="bytes")
    return [label.decode("utf-8") for label in data_dict[b"label_names"]]


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(np.min(arr)) if arr.size else 0.0
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax <= vmin:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    arr = (arr - vmin) / (vmax - vmin)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _encode_png(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        pil = Image.fromarray(array.astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(array.astype(np.uint8))
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _episode_frame_to_image(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[:, 0, :, :] if arr.shape[0] == 3 else arr[0]
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 2:
        return _normalize_to_uint8(arr)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _array_stats(array: np.ndarray) -> dict:
    arr = np.asarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.nanmin(arr)) if arr.size else None,
        "max": float(np.nanmax(arr)) if arr.size else None,
    }


def _extract_nifti_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :]
    if axis == 1:
        return volume[:, index, :]
    return volume[:, :, index]


def _safe_get_zooms(header, ndim: int):
    try:
        return [float(value) for value in header.get_zooms()[:ndim]]
    except Exception:
        return None


def _inspect_episode(data_dict: dict, options: dict) -> dict:
    frames = np.asarray(data_dict.get("frames", np.array([])))
    actions = np.asarray(data_dict.get("actions", np.array([])))
    proprios = np.asarray(data_dict.get("proprios", np.array([])))
    reward_data = data_dict.get("reward", [0])
    reward = float(reward_data[0]) if len(reward_data) else 0.0
    step_count = int(len(actions))
    step = int(options.get("step", 0) or 0)
    if step_count:
        step = max(0, min(step, step_count - 1))
    frame = frames[step] if step_count and len(frames) > step else None
    preview = None
    if frame is not None:
        preview = _encode_png(_episode_frame_to_image(frame))
    return {
        "viewer": "episode",
        "summary": {
            "steps": step_count,
            "reward": reward,
            "frame_shape": list(frames.shape[1:]) if frames.ndim > 1 else [],
            "action_shape": list(actions.shape),
            "proprio_shape": list(proprios.shape),
        },
        "controls": {
            "step": {"min": 0, "max": max(0, step_count - 1), "value": step},
        },
        "preview": {
            "image_base64": preview,
            "action_values": actions[step].tolist() if step_count and actions.ndim >= 2 else [],
            "proprio_values": proprios[step].tolist() if len(proprios) > step else [],
        },
    }


def _inspect_nifti(file_path: Path, options: dict) -> dict:
    if nib is None:
        raise ImportError("nibabel is not installed")
    image = nib.load(str(file_path))
    volume = image.get_fdata(dtype=np.float32)
    ndim = volume.ndim
    if ndim not in {3, 4}:
        return {
            "viewer": "nifti",
            "summary": {
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
                "zooms": _safe_get_zooms(image.header, ndim),
                "ndim": ndim,
            },
            "controls": {},
            "preview": {"message": f"Unsupported NIfTI ndim={ndim}"},
        }
    volume_index = int(options.get("volume_index", 0) or 0)
    axis = int(options.get("axis", 2) or 2)
    axis = max(0, min(axis, 2))
    active = volume
    if ndim == 4:
        max_volume = volume.shape[-1] - 1
        volume_index = max(0, min(volume_index, max_volume))
        active = volume[..., volume_index]
    max_slice = active.shape[axis] - 1
    slice_index = int(options.get("slice_index", max_slice // 2) or 0)
    slice_index = max(0, min(slice_index, max_slice))
    slice_2d = _extract_nifti_slice(active, axis, slice_index)
    return {
        "viewer": "nifti",
        "summary": {
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "zooms": _safe_get_zooms(image.header, ndim),
            "ndim": ndim,
            "min": float(np.nanmin(volume)) if volume.size else None,
            "max": float(np.nanmax(volume)) if volume.size else None,
        },
        "controls": {
            "axis": {"value": axis, "options": [0, 1, 2]},
            "slice_index": {"min": 0, "max": max_slice, "value": slice_index},
            "volume_index": {"min": 0, "max": volume.shape[-1] - 1, "value": volume_index} if ndim == 4 else None,
        },
        "preview": {
            "image_base64": _encode_png(_normalize_to_uint8(slice_2d)),
            "slice_shape": list(slice_2d.shape),
        },
    }


def _inspect_mnist(file_path: Path, options: dict) -> dict:
    data = _read_idx(file_path)
    sample_index = int(options.get("sample_index", 0) or 0)
    max_index = len(data) - 1
    sample_index = max(0, min(sample_index, max_index)) if max_index >= 0 else 0
    preview = None
    if data.ndim == 3 and len(data):
        preview = _encode_png(np.asarray(data[sample_index], dtype=np.uint8))
    return {
        "viewer": "mnist",
        "summary": {"shape": list(data.shape), "dtype": str(data.dtype)},
        "controls": {"sample_index": {"min": 0, "max": max(0, max_index), "value": sample_index}},
        "preview": {
            "image_base64": preview,
            "label_values": data[:100].tolist() if data.ndim == 1 else None,
        },
    }


def _inspect_cifar(file_path: Path, options: dict) -> dict:
    images, labels = _load_cifar_batch(file_path)
    sample_index = int(options.get("sample_index", 0) or 0)
    max_index = images.shape[0] - 1
    sample_index = max(0, min(sample_index, max_index)) if max_index >= 0 else 0
    meta_file = file_path.parent / "batches.meta"
    label_names = _load_cifar_meta(meta_file) if meta_file.exists() else None
    return {
        "viewer": "cifar",
        "summary": {"shape": list(images.shape), "dtype": str(images.dtype)},
        "controls": {"sample_index": {"min": 0, "max": max(0, max_index), "value": sample_index}},
        "preview": {
            "image_base64": _encode_png(images[sample_index]),
            "label": int(labels[sample_index]),
            "label_name": label_names[labels[sample_index]] if label_names else None,
        },
    }


def _extract_numeric_json_series(payload) -> dict:
    if not isinstance(payload, list) or not payload:
        return {}
    first = payload[0]
    if not isinstance(first, dict):
        return {}
    numeric_keys = [key for key, value in first.items() if isinstance(value, (int, float)) and key != "epoch"]
    epochs = [item.get("epoch", index + 1) for index, item in enumerate(payload)]
    return {
        key: [item.get(key, 0) for item in payload]
        for key in numeric_keys[:8]
    } | {"epoch": epochs}


def _inspect_json(file_path: Path) -> dict:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    preview_rows = payload[:20] if isinstance(payload, list) else payload
    return {
        "viewer": "json",
        "summary": {
            "kind": type(payload).__name__,
            "items": len(payload) if isinstance(payload, list) else None,
            "keys": list(payload[0].keys())[:20] if isinstance(payload, list) and payload and isinstance(payload[0], dict) else list(payload.keys())[:20] if isinstance(payload, dict) else [],
        },
        "preview": {
            "data": preview_rows,
            "series": _extract_numeric_json_series(payload),
        },
    }


def _inspect_checkpoint(file_path: Path) -> dict:
    if torch is None:
        raise ImportError("torch is not installed")
    checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
    top_level_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []
    state_dict = None
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    layers = []
    if isinstance(state_dict, dict):
        for name, tensor in list(state_dict.items())[:100]:
            layers.append({"layer": name, "shape": list(tensor.shape)})
    return {
        "viewer": "checkpoint",
        "summary": {
            "keys": top_level_keys[:20],
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "val_mse": checkpoint.get("val_mse") if isinstance(checkpoint, dict) else None,
            "layer_count": len(state_dict) if isinstance(state_dict, dict) else 0,
        },
        "preview": {
            "config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
            "layers": layers,
        },
    }


def _inspect_numpy(file_path: Path, options: dict) -> dict:
    if file_path.suffix.lower() == ".npz":
        data = dict(np.load(file_path, allow_pickle=True))
        if "frames" in data and "actions" in data:
            return _inspect_episode(data, options)
        arrays = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = _array_stats(value)
        return {
            "viewer": "numpy_bundle",
            "summary": {"keys": list(data.keys())[:20]},
            "preview": {"arrays": arrays},
        }
    array = np.load(file_path, allow_pickle=True)
    preview = None
    if array.ndim == 2:
        preview = _encode_png(_normalize_to_uint8(array))
    elif array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        preview = _encode_png(np.asarray(array, dtype=np.uint8) if array.dtype == np.uint8 else _episode_frame_to_image(array))
    return {
        "viewer": "numpy",
        "summary": _array_stats(array),
        "preview": {
            "image_base64": preview,
            "rows": array[:20].tolist() if array.ndim <= 2 else None,
        },
    }


def inspect_explorer_file(path_str: str, options: dict | None = None) -> dict:
    options = options or {}
    file_path = _resolve_repo_path(path_str)
    if not file_path.is_file():
        raise ValueError(f"file not found: {file_path}")
    name = file_path.name.lower()
    if file_path.suffix.lower() in {".npz", ".npy"}:
        payload = _inspect_numpy(file_path, options)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        payload = _inspect_nifti(file_path, options)
    elif "ubyte" in name:
        payload = _inspect_mnist(file_path, options)
    elif "batch" in name and "meta" not in name:
        payload = _inspect_cifar(file_path, options)
    elif file_path.suffix.lower() == ".json":
        payload = _inspect_json(file_path)
    elif file_path.suffix.lower() in {".pt", ".pth"}:
        payload = _inspect_checkpoint(file_path)
    else:
        payload = {
            "viewer": "unsupported",
            "summary": {"message": "unsupported file type"},
            "preview": {},
        }
    payload.update({
        "path": str(file_path),
        "relative_path": _relpath(file_path),
        "name": file_path.name,
        "size_kb": round(file_path.stat().st_size / 1024, 1),
    })
    return payload