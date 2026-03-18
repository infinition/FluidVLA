#!/usr/bin/env python3
"""
fluidvla_server.py
==================
FluidVLA Platform — Standalone web server.
No external web framework required. Pure Python stdlib + PyTorch.

Usage (run from FluidVLA-main/) :
    python fluidvla_server.py [--port 7860] [--host 0.0.0.0]

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import http.server
import json
import logging
import mimetypes
import os
import queue
import re
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fluidvla_platform.dataset_explorer import (
    build_explorer_overview,
    inspect_explorer_file,
    list_explorer_dir,
)
PLATFORM_DIR = ROOT / "fluidvla_platform"
STATIC = ROOT / "fluidvla_static"
INTERACTIVE_PAGE = PLATFORM_DIR / "interactive.html"
MEDICAL_STEP0_RESULTS = ROOT / "experiments" / "step1b_medical_msd" / "medical_step0_results_exhaustive.md"
SCOREBOARD_FILE = PLATFORM_DIR / "scoreboard.json"
CHECKPOINTS = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"
MEDICAL_DATA_DIR = DATA_DIR / "step1b_medical_msd"
INFERENCE_DIR = ROOT / "inference_outputs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("FluidVLA")

# ── WebSocket helpers ─────────────────────────────────────────────────────────

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
EVENT_PREFIX = "__FLUIDVLA_EVENT__"

def ws_handshake(rfile, wfile, headers):
    key = headers.get("Sec-WebSocket-Key", "").strip()
    accept = base64.b64encode(
        hashlib.sha1((key + WS_GUID).encode()).digest()
    ).decode()
    try:
        wfile.write(
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
        log.debug("Client disconnected during websocket handshake: %s", exc)

def ws_recv(sock):
    """Read one WebSocket frame. Returns (opcode, data) or (None, None) on close."""
    try:
        header = b""
        while len(header) < 2:
            chunk = sock.recv(2 - len(header))
            if not chunk: return None, None
            header += chunk
        b0, b1 = header
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack(">H", sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", sock.recv(8))[0]
        mask = sock.recv(4) if masked else b"\x00\x00\x00\x00"
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk: return None, None
            data += chunk
        if masked:
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        return opcode, data
    except Exception:
        return None, None

def ws_send(sock, data: str):
    """Send a text WebSocket frame."""
    try:
        payload = data.encode("utf-8")
        n = len(payload)
        if n < 126:
            header = bytes([0x81, n])
        elif n < 65536:
            header = bytes([0x81, 126]) + struct.pack(">H", n)
        else:
            header = bytes([0x81, 127]) + struct.pack(">Q", n)
        sock.sendall(header + payload)
    except Exception:
        pass


# ── Global state ──────────────────────────────────────────────────────────────

ws_clients: List[socket.socket] = []
ws_lock = threading.Lock()
training_jobs: Dict[str, dict] = {}   # job_id -> {process, log_queue, status}
job_lock = threading.Lock()


def broadcast(msg: dict):
    """Broadcast JSON message to all WebSocket clients."""
    text = json.dumps(msg)
    dead = []
    with ws_lock:
        for sock in ws_clients:
            try:
                ws_send(sock, text)
            except Exception:
                dead.append(sock)
        for d in dead:
            ws_clients.remove(d)


def emit(job_id: str, event: str, data: dict):
    broadcast({"job_id": job_id, "event": event, **data})


# ── Dataset / checkpoint discovery ───────────────────────────────────────────

TASK_META = {
    "Task01_BrainTumour":   {"in_channels": 4, "desc": "MRI Brain Tumour", "default_crop": [128,128,128]},
    "Task02_Heart":         {"in_channels": 1, "desc": "MRI Heart",        "default_crop": [128,128,128]},
    "Task03_Liver":         {"in_channels": 1, "desc": "CT Liver",         "default_crop": [128,128,128]},
    "Task04_Hippocampus":   {"in_channels": 1, "desc": "MRI Hippocampus",  "default_crop": [64,64,64]},
    "Task05_Prostate":      {"in_channels": 2, "desc": "MRI Prostate",     "default_crop": [128,128,64]},
    "Task06_Lung":          {"in_channels": 1, "desc": "CT Lung",          "default_crop": [128,128,128]},
    "Task07_Pancreas":      {"in_channels": 1, "desc": "CT Pancreas",      "default_crop": [128,128,128]},
    "Task08_HepaticVessel": {"in_channels": 1, "desc": "CT Hepatic Vessel","default_crop": [128,128,128]},
    "Task09_Spleen":        {"in_channels": 1, "desc": "CT Spleen",        "default_crop": [128,128,128]},
    "Task10_Colon":         {"in_channels": 1, "desc": "CT Colon",         "default_crop": [128,128,128]},
}

TASK_TRAIN_PRESETS = {
    "Task01_BrainTumour":   {"crop_mode": "center", "epochs": 5,  "max_train_samples": 16, "max_val_samples": 4, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task02_Heart":         {"crop_mode": "center", "epochs": 8,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task03_Liver":         {"crop_mode": "center", "epochs": 8,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task04_Hippocampus":   {"crop_mode": "mixed",  "epochs":10,  "max_train_samples": 32, "max_val_samples": 8, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task05_Prostate":      {"crop_mode": "mixed",  "epochs":10,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task06_Lung":          {"crop_mode": "center", "epochs": 8,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task07_Pancreas":      {"crop_mode": "mixed",  "epochs":10,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task08_HepaticVessel": {"crop_mode": "mixed",  "epochs":10,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task09_Spleen":        {"crop_mode": "center", "epochs": 6,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
    "Task10_Colon":         {"crop_mode": "mixed",  "epochs":10,  "max_train_samples": 24, "max_val_samples": 6, "d_model": 32, "n_layers": 2, "max_steps": 6, "diffusion_scale": 0.08},
}

EXPERIMENTS = {
    "step0_mnist": {
        "title": "Step 0 | Image Classification",
        "kicker": "Vision Core",
        "category": "vision",
        "status": "active",
        "summary": "Validates the PDE core on MNIST and CIFAR-10 before video.",
        "description": "First validation brick of the FluidVLA spatial core. This step measures convergence, training stability, and ability to learn useful visual features without attention.",
        "docs": "experiments/step0_mnist/README.md",
        "dataset_roots": ["data"],
        "checkpoint_roots": ["checkpoints/step0"],
        "output_roots": ["checkpoints/step0"],
        "data_patterns": ["MNIST/**", "CIFAR*", "cifar-10*"],
        "actions": [
            {
                "id": "train",
                "label": "Train classifier",
                "kind": "job",
                "parser": "step0_train",
                "params": [
                    {"name": "dataset", "label": "Dataset", "type": "select", "default": "mnist", "options": ["mnist", "cifar10"]},
                    {"name": "model", "label": "Model", "type": "select", "default": "small", "options": ["tiny", "small", "base"]},
                    {"name": "epochs", "label": "Epochs", "type": "number", "default": 20, "min": 1, "max": 200},
                    {"name": "batch_size", "label": "Batch size", "type": "number", "default": 128, "min": 1, "max": 1024},
                    {"name": "lr", "label": "Learning rate", "type": "number", "default": 0.0003, "step": 0.0001},
                ],
            }
        ],
    },
    "step1_video": {
        "title": "Step 1 | Video Prediction",
        "kicker": "Spatio-temporal PDE",
        "category": "video",
        "status": "active",
        "summary": "Video prediction and adaptive compute on Moving MNIST.",
        "description": "This step tests spatio-temporal diffusion, memory growth with temporal length, and dynamic reduction of PDE steps at inference.",
        "docs": "experiments/step1_video/README.md",
        "dataset_roots": ["data/step1_video"],
        "checkpoint_roots": ["checkpoints/step1"],
        "output_roots": ["checkpoints/step1", "data/step1_video"],
        "actions": [
            {
                "id": "train",
                "label": "Train video model",
                "kind": "job",
                "parser": "step1_video_train",
                "params": [
                    {"name": "epochs", "label": "Epochs", "type": "number", "default": 10, "min": 1, "max": 200},
                    {"name": "batch_size", "label": "Batch size", "type": "number", "default": 16, "min": 1, "max": 256},
                    {"name": "seq_len", "label": "Sequence length", "type": "number", "default": 10, "min": 4, "max": 32},
                    {"name": "d_model", "label": "d_model", "type": "number", "default": 64, "min": 16, "max": 256},
                    {"name": "n_layers", "label": "n_layers", "type": "number", "default": 3, "min": 1, "max": 8},
                    {"name": "max_train_samples", "label": "Max train samples", "type": "number", "default": 512, "min": 1},
                    {"name": "max_test_samples", "label": "Max test samples", "type": "number", "default": 128, "min": 1},
                    {"name": "epsilon", "label": "Epsilon", "type": "number", "default": 0.2, "step": 0.01},
                    {"name": "no_pde", "label": "Disable PDE", "type": "boolean", "default": False},
                ],
            }
        ],
    },
    "step1b_medical_msd": {
        "title": "Step 1b | Medical MSD",
        "kicker": "3D Medical Segmentation",
        "category": "medical",
        "status": "active",
        "summary": "Multi-task MSD pipeline, training and 3D NIfTI inference.",
        "description": "Most mature medical branch of the repo. Covers MSD datasets, FluidVLA and U-Net 3D checkpoints, 3D inference, and HTML/PNG output rendering.",
        "docs": "experiments/step1b_medical_msd/README.md",
        "dataset_roots": ["data/step1b_medical_msd"],
        "checkpoint_roots": ["checkpoints/fluidvla", "checkpoints/unet3d"],
        "output_roots": ["inference_outputs"],
        "actions": [
            {
                "id": "train",
                "label": "Train medical model",
                "kind": "job",
                "parser": "medical_train",
                "params": [
                    {"name": "task", "label": "Task", "type": "select", "default": "Task01_BrainTumour", "options": sorted(TASK_META.keys())},
                    {"name": "model_type", "label": "Model", "type": "select", "default": "fluidvla", "options": ["fluidvla", "unet3d"]},
                    {"name": "epochs", "label": "Epochs", "type": "number", "default": 10, "min": 1, "max": 1000},
                    {"name": "batch_size", "label": "Batch size", "type": "number", "default": 1, "min": 1, "max": 8},
                    {"name": "lr", "label": "Learning rate", "type": "number", "default": 0.001, "step": 0.0001, "min": 0.00001, "max": 0.01},
                    {"name": "max_train_samples", "label": "Max train samples (0=all)", "type": "number", "default": 64, "min": 0},
                    {"name": "max_val_samples", "label": "Max val samples (0=all)", "type": "number", "default": 16, "min": 0},
                    {"name": "d_model", "label": "d_model (feature width)", "type": "number", "default": 32, "min": 16, "max": 128},
                    {"name": "n_layers", "label": "PDE layers", "type": "number", "default": 2, "min": 1, "max": 6},
                    {"name": "max_steps", "label": "PDE steps/layer", "type": "number", "default": 6, "min": 2, "max": 16},
                    {"name": "depth", "label": "Crop depth", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "height", "label": "Crop height", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "width", "label": "Crop width", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "crop_mode", "label": "Crop mode", "type": "select", "default": "mixed", "options": ["center", "foreground", "mixed"]},
                    {"name": "binary", "label": "Binary segmentation", "type": "boolean", "default": False},
                    {"name": "no_pde", "label": "Disable PDE", "type": "boolean", "default": False},
                ],
            },
            {
                "id": "infer",
                "label": "Run inference",
                "kind": "medical_infer",
                "parser": "medical_infer",
                "params": [
                    {"name": "task", "label": "Task", "type": "select", "default": "Task01_BrainTumour", "options": sorted(TASK_META.keys())},
                    {"name": "model_type", "label": "Model", "type": "select", "default": "fluidvla", "options": ["fluidvla", "unet3d_std", "unet3d_tiny"]},
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "case", "label": "Case filename", "type": "text", "default": ""},
                    {"name": "depth", "label": "Depth", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "height", "label": "Height", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "width", "label": "Width", "type": "number", "default": 128, "min": 32, "max": 256},
                    {"name": "crop_mode", "label": "Crop mode", "type": "select", "default": "center", "options": ["center", "foreground"]},
                ],
            },
        ],
    },
    "step2_sim": {
        "title": "Step 2 | Sim / Isaac",
        "kicker": "Vision-Action Sim",
        "category": "robotics",
        "status": "active",
        "summary": "Demo collection, imitation learning, and Isaac evaluation.",
        "description": "This experiment covers synthetic or Isaac data collection, vision-action policy training, and debug outputs such as debug_frames.png or eval_results.json.",
        "docs": "experiments/step2_sim/README.md",
        "dataset_roots": ["data/step2_sim", "data/step2c_isaac"],
        "checkpoint_roots": ["checkpoints/step2_sim", "checkpoints/step2c_isaac"],
        "output_roots": ["data/step2_sim", "data/step2c_isaac", "checkpoints/step2_sim", "checkpoints/step2c_isaac"],
        "actions": [
            {
                "id": "collect_synth",
                "label": "Collect synthetic demos",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "episodes", "label": "Episodes", "type": "number", "default": 200, "min": 1},
                    {"name": "image_size", "label": "Image size", "type": "number", "default": 64, "min": 32, "max": 512},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./data/step2_sim"},
                ],
            },
            {
                "id": "collect_isaac",
                "label": "Collect Isaac demos",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "episodes", "label": "Episodes", "type": "number", "default": 100, "min": 1},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./data/step2c_isaac"},
                    {"name": "show_gui", "label": "Show GUI", "type": "boolean", "default": False},
                ],
            },
            {
                "id": "train",
                "label": "Train policy",
                "kind": "job",
                "parser": "step2_sim_train",
                "params": [
                    {"name": "dataset", "label": "Dataset dir", "type": "text", "default": "./data/step2_sim"},
                    {"name": "epochs", "label": "Epochs", "type": "number", "default": 20, "min": 1, "max": 200},
                    {"name": "batch_size", "label": "Batch size", "type": "number", "default": 16, "min": 1, "max": 256},
                    {"name": "d_model", "label": "d_model", "type": "number", "default": 256, "min": 32, "max": 512},
                    {"name": "n_layers", "label": "n_layers", "type": "number", "default": 4, "min": 1, "max": 8},
                    {"name": "eq_weight", "label": "Eq weight", "type": "number", "default": 0.01, "step": 0.01},
                    {"name": "epsilon", "label": "Epsilon", "type": "number", "default": 0.02, "step": 0.01},
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./checkpoints/step2_sim"},
                ],
            },
            {
                "id": "eval",
                "label": "Evaluate checkpoint",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "save_dir", "label": "Result dir", "type": "text", "default": "./data/step2c_isaac"},
                    {"name": "show_gui", "label": "Show GUI", "type": "boolean", "default": False},
                ],
            },
        ],
    },
    "step2a_synthetic": {
        "title": "Step 2a | Synthetic Pick & Place",
        "kicker": "Synthetic Imitation",
        "category": "robotics",
        "status": "active",
        "summary": "Pure synthetic imitation learning pipeline.",
        "description": "This step serves as a validation gate before Isaac. The repo currently exposes mainly training on pre-generated datasets, which is explicitly flagged in the platform.",
        "docs": "experiments/step2a_synthetic/README.md",
        "dataset_roots": ["data/step2a_synthetic"],
        "checkpoint_roots": ["checkpoints/step2a_synthetic"],
        "output_roots": ["checkpoints/step2a_synthetic", "data/step2a_synthetic"],
        "actions": [
            {
                "id": "train",
                "label": "Train synthetic policy",
                "kind": "job",
                "parser": "step2a_train",
                "params": [
                    {"name": "dataset", "label": "Dataset dir", "type": "text", "default": "./data/step2a_synthetic"},
                    {"name": "epochs", "label": "Epochs", "type": "number", "default": 20, "min": 1, "max": 200},
                    {"name": "batch_size", "label": "Batch size", "type": "number", "default": 32, "min": 1, "max": 256},
                    {"name": "d_model", "label": "d_model", "type": "number", "default": 128, "min": 32, "max": 512},
                    {"name": "n_layers", "label": "n_layers", "type": "number", "default": 3, "min": 1, "max": 8},
                    {"name": "eq_weight", "label": "Eq weight", "type": "number", "default": 0.1, "step": 0.01},
                    {"name": "epsilon", "label": "Epsilon", "type": "number", "default": 0.02, "step": 0.01},
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "no_pde", "label": "Disable PDE", "type": "boolean", "default": False},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./checkpoints/step2a_synthetic"},
                ],
            }
        ],
    },
    "step2d_so101_urdf": {
        "title": "Step 2d | SO-101 URDF Viewer",
        "kicker": "Embodiment Viewer",
        "category": "robotics",
        "status": "active",
        "summary": "Demo-oriented URDF viewer for Isaac Sim and FluidVLA trajectories.",
        "description": "Visualization and demo brick for loading the SO-101, replaying a Step 2 checkpoint, and making the simulation-to-embodiment transition more tangible.",
        "docs": "experiments/step2d_so101_urdf/README.md",
        "dataset_roots": [],
        "checkpoint_roots": ["checkpoints/step2d_so101_urdf", "checkpoints/step2c_isaac", "checkpoints/step2a_synthetic"],
        "output_roots": ["checkpoints/step2d_so101_urdf"],
        "actions": [
            {
                "id": "viewer",
                "label": "Launch URDF viewer",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "urdf", "label": "URDF path", "type": "text", "default": ""},
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "n_steps", "label": "n_steps", "type": "number", "default": 200, "min": 1, "max": 5000},
                    {"name": "image_size", "label": "Image size", "type": "number", "default": 64, "min": 32, "max": 512},
                    {"name": "show_gui", "label": "Show GUI", "type": "boolean", "default": False},
                    {"name": "random_weights", "label": "Random weights", "type": "boolean", "default": False},
                    {"name": "fallback_rerun", "label": "Fallback rerun", "type": "boolean", "default": False},
                    {"name": "no_pde", "label": "Disable PDE", "type": "boolean", "default": False},
                ],
            }
        ],
    },
    "step3_lerobot": {
        "title": "Step 3 | LeRobot / Real Hardware",
        "kicker": "Edge Robotics",
        "category": "robotics",
        "status": "active",
        "summary": "Collection, benchmark, and inference for the real SO-101 arm.",
        "description": "Last mile of the vision-action pipeline. This step combines episode collection, latency benchmarking, and real-time inference with a mock_robot mode to keep a safe test surface.",
        "docs": "experiments/step3_lerobot/README.md",
        "dataset_roots": ["data/step3_lerobot"],
        "checkpoint_roots": ["checkpoints/step3_lerobot"],
        "output_roots": ["data/step3_lerobot", "checkpoints/step3_lerobot"],
        "actions": [
            {
                "id": "benchmark",
                "label": "Benchmark latency",
                "kind": "job",
                "parser": "step3_benchmark",
                "params": [
                    {"name": "camera_id", "label": "Camera id", "type": "number", "default": 0, "min": 0},
                    {"name": "mock_robot", "label": "Mock robot", "type": "boolean", "default": True},
                ],
            },
            {
                "id": "collect",
                "label": "Collect episodes",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "task", "label": "Task", "type": "text", "default": "pick_place"},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./data/step3_lerobot"},
                    {"name": "camera_id", "label": "Camera id", "type": "number", "default": 0, "min": 0},
                    {"name": "robot_port", "label": "Robot port", "type": "text", "default": "/dev/ttyUSB0"},
                    {"name": "mock_robot", "label": "Mock robot", "type": "boolean", "default": True},
                ],
            },
            {
                "id": "infer",
                "label": "Run hardware inference",
                "kind": "job",
                "parser": "generic",
                "params": [
                    {"name": "checkpoint", "label": "Checkpoint path", "type": "text", "default": ""},
                    {"name": "task", "label": "Task", "type": "text", "default": "pick_place"},
                    {"name": "save_dir", "label": "Save dir", "type": "text", "default": "./data/step3_lerobot"},
                    {"name": "camera_id", "label": "Camera id", "type": "number", "default": 0, "min": 0},
                    {"name": "robot_port", "label": "Robot port", "type": "text", "default": "/dev/ttyUSB0"},
                    {"name": "mock_robot", "label": "Mock robot", "type": "boolean", "default": True},
                ],
            },
        ],
    },
}

AUTO_SCOREBOARD_HINTS = {
    "step0_mnist": {"metric": "test_acc", "direction": "max", "unit": "accuracy"},
    "step1_video": {"metric": "best_test_mse", "direction": "min", "unit": "mse"},
    "step1b_medical_msd": {"metric": "best_val_dice", "direction": "max", "unit": "dice"},
    "step2_sim": {"metric": "val_mse", "direction": "min", "unit": "mse"},
    "step2a_synthetic": {"metric": "val_mse", "direction": "min", "unit": "mse"},
    "step2d_so101_urdf": {"metric": "artifact", "direction": "max", "unit": ""},
    "step3_lerobot": {"metric": "episodes", "direction": "max", "unit": "episodes"},
}


def default_task_data_dir(task_name: str) -> str:
    medical_path = MEDICAL_DATA_DIR / task_name
    legacy_path = DATA_DIR / task_name
    if medical_path.exists():
        return f"./data/step1b_medical_msd/{task_name}"
    if legacy_path.exists():
        return f"./data/{task_name}"
    return f"./data/step1b_medical_msd/{task_name}"

def scan_datasets() -> list:
    if not DATA_DIR.exists():
        return []
    results = []
    for task_name, meta in TASK_META.items():
        task_path = MEDICAL_DATA_DIR / task_name
        if not task_path.exists():
            task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue
        img_dir = task_path / "imagesTr"
        lbl_dir = task_path / "labelsTr"
        cases = []
        if img_dir.exists():
            cases = sorted([f.name for f in img_dir.glob("*.nii.gz")
                            if not f.name.startswith("._")])
        results.append({
            "task": task_name,
            "path": str(task_path),
            "desc": meta["desc"],
            "in_channels": meta["in_channels"],
            "default_crop": meta["default_crop"],
            "n_cases": len(cases),
            "cases": cases[:20],  # first 20 for UI
            "train_preset": TASK_TRAIN_PRESETS.get(task_name, {}),
        })
    return results


def scan_checkpoints() -> list:
    if not CHECKPOINTS.exists():
        return []
    results = []
    for model_type in ["fluidvla", "unet3d"]:
        base = CHECKPOINTS / model_type
        if not base.exists():
            continue
        for task_dir in sorted(base.iterdir()):
            if not task_dir.is_dir():
                continue
            for ckpt in task_dir.glob("*.pt"):
                resolved_model_type = model_type
                ckpt_name = ckpt.name.lower()
                if model_type == "unet3d":
                    if "tiny" in ckpt_name:
                        resolved_model_type = "unet3d_tiny"
                    elif "std" in ckpt_name:
                        resolved_model_type = "unet3d_std"

                meta = {"model_type": resolved_model_type, "task": task_dir.name,
                        "filename": ckpt.name, "path": str(ckpt),
                        "size_mb": round(ckpt.stat().st_size / 1e6, 1),
                        "best_val_dice": None}
                # Try to read best_val_dice from sibling json
                history_candidates = [
                    ckpt.with_suffix(".json"),
                    task_dir / "history.json",
                    task_dir / "history_std.json",
                    task_dir / "history_tiny.json",
                    task_dir / "results.json",
                ]
                for history_f in history_candidates:
                    if not history_f.exists():
                        continue
                    try:
                        h = json.loads(history_f.read_text())
                        if isinstance(h, list) and h:
                            meta["best_val_dice"] = round(max(r.get("val_dice", r.get("best_val_dice", 0)) for r in h), 4)
                            break
                        if isinstance(h, dict):
                            score = h.get("best_val_dice")
                            if score is not None:
                                meta["best_val_dice"] = round(float(score), 4)
                                break
                    except Exception:
                        pass
                results.append(meta)
    # Also scan old-style checkpoints/step_medical0/
    for ckpt in CHECKPOINTS.rglob("*.pt"):
        if "fluidvla" in str(ckpt) or "unet3d" in str(ckpt):
            continue
        results.append({
            "model_type": "fluidvla",
            "task": ckpt.parent.name,
            "filename": ckpt.name,
            "path": str(ckpt),
            "size_mb": round(ckpt.stat().st_size / 1e6, 1),
            "best_val_dice": None,
        })
    return results


def scan_inference_outputs() -> list:
    if not INFERENCE_DIR.exists():
        return []
    results = []
    for task_dir in sorted(INFERENCE_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        files = {"task": task_dir.name, "items": []}
        for f in sorted(task_dir.iterdir()):
            if f.suffix in (".png", ".html", ".json"):
                files["items"].append({
                    "name": f.name, "path": str(f),
                    "type": f.suffix[1:],
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
        if files["items"]:
            results.append(files)
    return results


SUPPORTED_INFER_MODEL_TYPES = {"fluidvla", "unet3d_std", "unet3d_tiny"}


def normalize_infer_model_type(model_type: str, checkpoint: str = "") -> str:
    resolved = (model_type or "fluidvla").strip().lower()
    if resolved != "unet3d":
        return resolved

    ckpt_name = Path(checkpoint).name.lower()
    if "tiny" in ckpt_name:
        return "unet3d_tiny"
    return "unet3d_std"


def infer_checkpoint_model_type(checkpoint: Path) -> Optional[str]:
    path_lower = str(checkpoint).replace("\\", "/").lower()
    name_lower = checkpoint.name.lower()

    if "/fluidvla/" in path_lower or "fluidvla" in name_lower:
        return "fluidvla"
    if "/unet3d/" in path_lower or "unet" in name_lower:
        if "tiny" in name_lower:
            return "unet3d_tiny"
        if "std" in name_lower:
            return "unet3d_std"
        return "unet3d_std"
    return None


def validate_checkpoint_for_task(checkpoint: str, task: str, model_type: str) -> Optional[str]:
    if not checkpoint:
        return "checkpoint is required"

    ckpt_path = Path(checkpoint).expanduser()
    if not ckpt_path.exists() or not ckpt_path.is_file():
        return f"checkpoint not found: {ckpt_path}"
    if ckpt_path.suffix.lower() != ".pt":
        return f"checkpoint must be a .pt file: {ckpt_path.name}"

    resolved_model_type = normalize_infer_model_type(model_type, checkpoint)
    if resolved_model_type not in SUPPORTED_INFER_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_INFER_MODEL_TYPES))
        return f"unsupported model_type '{model_type}'. Expected one of: {supported}"

    inferred_model_type = infer_checkpoint_model_type(ckpt_path)
    if inferred_model_type and inferred_model_type != resolved_model_type:
        return (
            f"checkpoint/model mismatch: checkpoint looks like {inferred_model_type}, "
            f"but request asked for {resolved_model_type}"
        )

    parent_task = ckpt_path.parent.name
    if task and re.match(r"^Task\d+_", parent_task) and parent_task != task:
        return f"checkpoint/task mismatch: checkpoint is under {parent_task}, requested task is {task}"

    return None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relpath_str(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_scoreboard_file():
    if SCOREBOARD_FILE.exists():
        return
    SCOREBOARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    initial = {
        "version": 1,
        "entries": [
            {
                "id": "medical-step0-dice",
                "experiment_id": "step1b_medical_msd",
                "title": "Task01 BrainTumour | Best Val Dice PDE ON",
                "metric_name": "best_val_dice",
                "value": 0.9177,
                "unit": "dice",
                "direction": "max",
                "source": "README.md / experiments/step1b_medical_msd/medical_step0_results_exhaustive.md",
                "notes": "16 train / 4 val / 5 epochs, calibrated post-fix result.",
                "updated_at": iso_now(),
            },
            {
                "id": "medical-step0-latency",
                "experiment_id": "step1b_medical_msd",
                "title": "BRATS_001 central slice latency",
                "metric_name": "latency_ms",
                "value": 44.16,
                "unit": "ms",
                "direction": "min",
                "source": "README.md / experiments/step1b_medical_msd/medical_step0_results_exhaustive.md",
                "notes": "Central-slice inference, 6/6 steps, turbulence 0.2218.",
                "updated_at": iso_now(),
            },
            {
                "id": "medical-step0-pde-gain",
                "experiment_id": "step1b_medical_msd",
                "title": "PDE gain vs no PDE",
                "metric_name": "delta_dice",
                "value": 0.0048,
                "unit": "dice",
                "direction": "max",
                "source": "README.md / experiments/step1b_medical_msd/medical_step0_results_exhaustive.md",
                "notes": "Controlled PDE OFF 0.9129 vs PDE ON 0.9177 comparison.",
                "updated_at": iso_now(),
            },
        ],
    }
    SCOREBOARD_FILE.write_text(json.dumps(initial, indent=2), encoding="utf-8")


def load_manual_scoreboard() -> list:
    ensure_scoreboard_file()
    payload = read_json_file(SCOREBOARD_FILE)
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return entries


def save_manual_scoreboard(entries: list):
    ensure_scoreboard_file()
    SCOREBOARD_FILE.write_text(
        json.dumps({"version": 1, "entries": entries}, indent=2),
        encoding="utf-8",
    )


def upsert_manual_scoreboard(entry: dict) -> dict:
    entries = load_manual_scoreboard()
    now = iso_now()
    clean = {
        "id": entry.get("id") or f"manual-{int(time.time() * 1000)}",
        "experiment_id": entry.get("experiment_id", "step1b_medical_msd"),
        "title": entry.get("title", "Untitled metric").strip() or "Untitled metric",
        "metric_name": entry.get("metric_name", "metric").strip() or "metric",
        "value": float(entry.get("value", 0)),
        "unit": str(entry.get("unit", "")).strip(),
        "direction": entry.get("direction", "max"),
        "source": str(entry.get("source", "manual")).strip() or "manual",
        "notes": str(entry.get("notes", "")).strip(),
        "updated_at": now,
    }
    replaced = False
    for index, existing in enumerate(entries):
        if existing.get("id") == clean["id"]:
            entries[index] = clean
            replaced = True
            break
    if not replaced:
        entries.append(clean)
    save_manual_scoreboard(entries)
    return clean


def delete_manual_scoreboard(entry_id: str) -> bool:
    entries = load_manual_scoreboard()
    next_entries = [entry for entry in entries if entry.get("id") != entry_id]
    if len(next_entries) == len(entries):
        return False
    save_manual_scoreboard(next_entries)
    return True


def count_files(path: Path, predicate=None) -> int:
    if not path.exists():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if predicate and not predicate(candidate):
            continue
        total += 1
    return total


def collect_media(paths: list[Path], limit: int = 24) -> list:
    media_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".webm", ".mp4", ".html", ".json"}
    items = []
    seen = set()
    for root in paths:
        if not root.exists():
            continue
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_file():
                continue
            name_lower = candidate.name.lower()
            suffix = candidate.suffix.lower()
            is_nii = name_lower.endswith(".nii.gz")
            if suffix not in media_suffixes and not is_nii:
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            items.append({
                "name": candidate.name,
                "path": str(candidate),
                "relative_path": relpath_str(candidate),
                "type": "nii.gz" if is_nii else suffix.lstrip("."),
                "size_kb": round(candidate.stat().st_size / 1024, 1),
                "root": relpath_str(root),
            })
            if len(items) >= limit:
                return items
    return items


def summarize_path(path: Path, label: str, kind: str, predicate=None) -> dict:
    exists = path.exists()
    preview = []
    if exists:
        for candidate in sorted(path.rglob("*")):
            if not candidate.is_file():
                continue
            if predicate and not predicate(candidate):
                continue
            preview.append(relpath_str(candidate))
            if len(preview) >= 6:
                break
    return {
        "label": label,
        "kind": kind,
        "path": str(path),
        "relative_path": relpath_str(path),
        "exists": exists,
        "file_count": count_files(path, predicate) if exists else 0,
        "preview": preview,
    }


def history_best_value(records, key: str, direction: str):
    values = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            value = record.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                pass
    if not values:
        return None
    return max(values) if direction == "max" else min(values)


def build_auto_scoreboard() -> list:
    entries = []

    step0_dir = CHECKPOINTS / "step0"
    if step0_dir.exists():
        for history_file in sorted(step0_dir.glob("history_*.json")):
            history = read_json_file(history_file)
            best = history_best_value(history, "test_acc", "max")
            if best is None:
                continue
            dataset_name = history_file.stem.replace("history_", "")
            entries.append({
                "id": f"auto-step0-{dataset_name}",
                "experiment_id": "step0_mnist",
                "title": f"{dataset_name.upper()} | Best test accuracy",
                "metric_name": "test_acc",
                "value": round(best, 4),
                "unit": "accuracy",
                "direction": "max",
                "source": relpath_str(history_file),
                "notes": "Auto-discovered from history JSON.",
                "updated_at": iso_now(),
                "auto": True,
            })

    step1_dir = CHECKPOINTS / "step1"
    if step1_dir.exists():
        summary = read_json_file(step1_dir / "summary_video.json")
        if isinstance(summary, dict) and summary.get("best_test_mse") is not None:
            entries.append({
                "id": "auto-step1-summary",
                "experiment_id": "step1_video",
                "title": "Video | Best test MSE",
                "metric_name": "best_test_mse",
                "value": round(float(summary["best_test_mse"]), 5),
                "unit": "mse",
                "direction": "min",
                "source": relpath_str(step1_dir / "summary_video.json"),
                "notes": "Auto-discovered from summary_video.json.",
                "updated_at": iso_now(),
                "auto": True,
            })

    for ckpt in scan_checkpoints():
        if not ckpt.get("task") or ckpt.get("best_val_dice") is None:
            continue
        entries.append({
            "id": f"auto-medical-{ckpt['model_type']}-{ckpt['task']}-{ckpt['filename']}",
            "experiment_id": "step1b_medical_msd",
            "title": f"{ckpt['task']} | {ckpt['model_type']}",
            "metric_name": "best_val_dice",
            "value": round(float(ckpt["best_val_dice"]), 4),
            "unit": "dice",
            "direction": "max",
            "source": relpath_str(Path(ckpt["path"])),
            "notes": "Auto-discovered from checkpoint sidecar/history.",
            "updated_at": iso_now(),
            "auto": True,
        })

    for exp_id, ckpt_dir_name in (("step2_sim", "step2_sim"), ("step2_sim", "step2c_isaac"), ("step2a_synthetic", "step2a_synthetic")):
        history_path = CHECKPOINTS / ckpt_dir_name / "history.json"
        history = read_json_file(history_path)
        best = history_best_value(history, "val_mse", "min")
        if best is None:
            continue
        entries.append({
            "id": f"auto-{exp_id}-{ckpt_dir_name}",
            "experiment_id": exp_id,
            "title": f"{ckpt_dir_name} | Best val MSE",
            "metric_name": "val_mse",
            "value": round(best, 5),
            "unit": "mse",
            "direction": "min",
            "source": relpath_str(history_path),
            "notes": "Auto-discovered from history.json.",
            "updated_at": iso_now(),
            "auto": True,
        })

    step3_dir = DATA_DIR / "step3_lerobot"
    index_file = step3_dir / "dataset_index.json"
    index = read_json_file(index_file)
    if isinstance(index, dict):
        total_episodes = index.get("total_episodes") or len(index.get("episodes", []))
        try:
            total_episodes = int(total_episodes)
        except Exception:
            total_episodes = 0
        entries.append({
            "id": "auto-step3-episodes",
            "experiment_id": "step3_lerobot",
            "title": "LeRobot | Collected episodes",
            "metric_name": "episodes",
            "value": total_episodes,
            "unit": "episodes",
            "direction": "max",
            "source": relpath_str(index_file),
            "notes": "Auto-discovered from dataset_index.json.",
            "updated_at": iso_now(),
            "auto": True,
        })

    return entries


def build_scoreboard_payload() -> dict:
    manual_entries = load_manual_scoreboard()
    auto_entries = build_auto_scoreboard()
    merged = manual_entries + auto_entries
    grouped = {}
    for exp_id in EXPERIMENTS:
        grouped[exp_id] = []
    for entry in merged:
        grouped.setdefault(entry.get("experiment_id", "unknown"), []).append(entry)
    for values in grouped.values():
        values.sort(key=lambda item: (item.get("auto", False), item.get("title", "")))
    return {"manual": manual_entries, "auto": auto_entries, "grouped": grouped}


def experiment_paths(config: dict, key: str) -> list[Path]:
    return [ROOT / rel for rel in config.get(key, [])]


def discover_experiment_state(exp_id: str, config: dict) -> dict:
    dataset_paths = experiment_paths(config, "dataset_roots")
    checkpoint_paths = experiment_paths(config, "checkpoint_roots")
    output_paths = experiment_paths(config, "output_roots")

    data_summaries = []
    for path in dataset_paths:
        data_summaries.append(summarize_path(path, path.name or relpath_str(path), "dataset"))

    checkpoint_summaries = []
    for path in checkpoint_paths:
        checkpoint_summaries.append(
            summarize_path(path, path.name or relpath_str(path), "checkpoint", lambda candidate: candidate.suffix.lower() == ".pt")
        )

    output_summaries = []
    for path in output_paths:
        output_summaries.append(summarize_path(path, path.name or relpath_str(path), "output"))

    media = collect_media(output_paths + checkpoint_paths)
    dataset_files = sum(item["file_count"] for item in data_summaries)
    checkpoint_files = sum(item["file_count"] for item in checkpoint_summaries)
    output_files = sum(item["file_count"] for item in output_summaries)

    return {
        "id": exp_id,
        "title": config["title"],
        "kicker": config["kicker"],
        "category": config["category"],
        "status": config["status"],
        "summary": config["summary"],
        "description": config["description"],
        "docs": config["docs"],
        "actions": config.get("actions", []),
        "paths": {
            "datasets": data_summaries,
            "checkpoints": checkpoint_summaries,
            "outputs": output_summaries,
        },
        "counts": {
            "dataset_files": dataset_files,
            "checkpoint_files": checkpoint_files,
            "output_files": output_files,
            "media_items": len(media),
        },
        "media": media,
    }


def build_experiments_payload() -> list:
    return [discover_experiment_state(exp_id, config) for exp_id, config in EXPERIMENTS.items()]


def parse_job_metrics(parser_key: str, line: str) -> Optional[dict]:
    patterns = {
        "step0_train": [
            (r"Test\s+\|\s+Acc:\s+([0-9.]+)%", lambda match: {"metric": "test_acc", "value": float(match.group(1)), "unit": "%"}),
        ],
        "step1_video_train": [
            (r"Test MSE:\s*([0-9.]+).*?Test Steps:\s*([0-9.]+)", lambda match: {"metric": "test_mse", "value": float(match.group(1)), "steps": float(match.group(2)), "unit": "mse"}),
        ],
        "medical_train": [
            (r"val_dice=(\d+\.\d+)", lambda match: {"metric": "val_dice", "value": float(match.group(1)), "unit": "dice"}),
        ],
        "step2_sim_train": [
            (r"MSE:\s*([0-9.]+).*?Steps\(eval\):\s*([0-9.]+)", lambda match: {"metric": "val_mse", "value": float(match.group(1)), "steps": float(match.group(2)), "unit": "mse"}),
        ],
        "step2a_train": [
            (r"Val MSE:([0-9.]+).*?Steps\(eval\):([0-9.]+).*?Lat:([0-9.]+)ms", lambda match: {"metric": "val_mse", "value": float(match.group(1)), "steps": float(match.group(2)), "latency_ms": float(match.group(3)), "unit": "mse"}),
        ],
        "step3_benchmark": [
            (r"Loop mean:\s*([0-9.]+)ms\s*\|\s*p95:\s*([0-9.]+)ms", lambda match: {"metric": "loop_mean_ms", "value": float(match.group(1)), "p95_ms": float(match.group(2)), "unit": "ms"}),
        ],
    }
    for regex, builder in patterns.get(parser_key, []):
        match = re.search(regex, line)
        if match:
            return builder(match)
    return None


def start_command_job(job_id: str, cmd: list, cwd: str, parser_key: str = "generic", meta: Optional[dict] = None):
    meta = meta or {}

    def _run():
        with job_lock:
            training_jobs[job_id]["status"] = "running"
        emit(job_id, "job_start", {"cmd": " ".join(cmd), **meta})
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=cwd,
                env=env,
            )
            with job_lock:
                training_jobs[job_id]["process"] = proc
            for line in proc.stdout:
                line = line.rstrip()
                emit(job_id, "job_log", {"line": line, **meta})
                metric = parse_job_metrics(parser_key, line)
                if metric:
                    emit(job_id, "job_metric", {**metric, **meta})
            proc.wait()
            status = "done" if proc.returncode == 0 else "error"
            with job_lock:
                training_jobs[job_id]["status"] = status
            emit(job_id, "job_end", {"status": status, "returncode": proc.returncode, **meta})
        except Exception as exc:
            with job_lock:
                training_jobs[job_id]["status"] = "error"
            emit(job_id, "job_end", {"status": "error", "msg": str(exc), **meta})

    threading.Thread(target=_run, daemon=True).start()


def _bool_param(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_experiment_command(experiment_id: str, action_id: str, params: dict) -> tuple[list, str, dict]:
    if experiment_id == "step0_mnist" and action_id == "train":
        cmd = [
            sys.executable,
            "experiments/step0_mnist/train_step0.py",
            "--dataset", str(params.get("dataset", "mnist")),
            "--model", str(params.get("model", "small")),
            "--epochs", str(int(params.get("epochs", 20))),
            "--batch_size", str(int(params.get("batch_size", 128))),
            "--lr", str(float(params.get("lr", 3e-4))),
            "--data_dir", str(params.get("data_dir", "./data")),
            "--save_dir", str(params.get("save_dir", "./checkpoints/step0")),
        ]
        return cmd, "step0_train", {"experiment_id": experiment_id, "action_id": action_id, "dataset": str(params.get("dataset", "mnist"))}

    if experiment_id == "step1_video" and action_id == "train":
        cmd = [
            sys.executable,
            "experiments/step1_video/train_step1_video.py",
            "--data_dir", str(params.get("data_dir", "./data/step1_video")),
            "--save_dir", str(params.get("save_dir", "./checkpoints/step1")),
            "--epochs", str(int(params.get("epochs", 10))),
            "--batch_size", str(int(params.get("batch_size", 16))),
            "--seq_len", str(int(params.get("seq_len", 10))),
            "--d_model", str(int(params.get("d_model", 64))),
            "--n_layers", str(int(params.get("n_layers", 3))),
            "--max_train_samples", str(int(params.get("max_train_samples", 512))),
            "--max_test_samples", str(int(params.get("max_test_samples", 128))),
            "--epsilon", str(float(params.get("epsilon", 0.2))),
            "--workers", str(int(params.get("workers", 0))),
        ]
        if _bool_param(params.get("no_pde"), False):
            cmd.append("--no_pde")
        return cmd, "step1_video_train", {"experiment_id": experiment_id, "action_id": action_id, "task": "MovingMNIST"}

    if experiment_id == "step1b_medical_msd" and action_id == "train":
        task = str(params.get("task", "Task01_BrainTumour"))
        model_type = str(params.get("model_type", "fluidvla"))
        data_dir = str(params.get("data_dir", default_task_data_dir(task)))
        if model_type == "fluidvla":
            cmd = [
                sys.executable,
                "experiments/step1b_medical_msd/train_fluidvla_msd.py",
                "--task", task,
                "--data_dir", data_dir,
                "--save_dir", str(params.get("save_dir", "./checkpoints/fluidvla")),
                "--epochs", str(int(params.get("epochs", 10))),
                "--batch_size", str(int(params.get("batch_size", 1))),
                "--lr", str(float(params.get("lr", 1e-3))),
                "--max_train_samples", str(int(params.get("max_train_samples", 64))),
                "--max_val_samples", str(int(params.get("max_val_samples", 16))),
                "--d_model", str(int(params.get("d_model", 32))),
                "--n_layers", str(int(params.get("n_layers", 2))),
                "--max_steps", str(int(params.get("max_steps", 6))),
                "--depth", str(int(params.get("depth", 128))),
                "--height", str(int(params.get("height", 128))),
                "--width", str(int(params.get("width", 128))),
                "--crop_mode", str(params.get("crop_mode", "mixed")),
            ]
            if _bool_param(params.get("binary"), False):
                cmd.append("--binary")
            else:
                cmd.append("--no-binary")
            if _bool_param(params.get("no_pde"), False):
                cmd.append("--no_pde")
        else:
            cmd = [
                sys.executable,
                "experiments/step1b_medical_msd/train_unet3d_msd.py",
                "--task", task,
                "--data_dir", data_dir,
                "--save_dir", str(params.get("save_dir", "./checkpoints/unet3d")),
                "--epochs", str(int(params.get("epochs", 5))),
                "--batch_size", str(int(params.get("batch_size", 1))),
                "--max_train_samples", str(int(params.get("max_train_samples", 16))),
                "--max_val_samples", str(int(params.get("max_val_samples", 4))),
                "--depth", str(int(params.get("depth", 128))),
                "--height", str(int(params.get("height", 128))),
                "--width", str(int(params.get("width", 128))),
                "--crop_mode", str(params.get("crop_mode", "center")),
            ]
            if _bool_param(params.get("binary"), True):
                cmd.append("--binary")
            else:
                cmd.append("--no-binary")
        return cmd, "medical_train", {"experiment_id": experiment_id, "action_id": action_id, "task": task}

    if experiment_id == "step2_sim" and action_id == "collect_synth":
        cmd = [
            sys.executable,
            "experiments/step2_sim/isaac_env.py",
            "--mode", "synthetic",
            "--episodes", str(int(params.get("episodes", 200))),
            "--image_size", str(int(params.get("image_size", 64))),
            "--save_dir", str(params.get("save_dir", "./data/step2_sim")),
        ]
        return cmd, "generic", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step2_sim" and action_id == "collect_isaac":
        cmd = [
            sys.executable,
            "experiments/step2_sim/isaac_env.py",
            "--mode", "collect",
            "--episodes", str(int(params.get("episodes", 100))),
            "--save_dir", str(params.get("save_dir", "./data/step2c_isaac")),
        ]
        if _bool_param(params.get("show_gui"), False):
            cmd.append("--show_gui")
        return cmd, "generic", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step2_sim" and action_id == "train":
        cmd = [
            sys.executable,
            "experiments/step2_sim/train_step2.py",
            "--dataset", str(params.get("dataset", "./data/step2_sim")),
            "--epochs", str(int(params.get("epochs", 20))),
            "--batch_size", str(int(params.get("batch_size", 16))),
            "--lr", str(float(params.get("lr", 3e-4))),
            "--d_model", str(int(params.get("d_model", 256))),
            "--n_layers", str(int(params.get("n_layers", 4))),
            "--max_steps", str(int(params.get("max_steps", 12))),
            "--epsilon", str(float(params.get("epsilon", 0.02))),
            "--eq_weight", str(float(params.get("eq_weight", 0.01))),
            "--save_dir", str(params.get("save_dir", "./checkpoints/step2_sim")),
        ]
        checkpoint = str(params.get("checkpoint", "")).strip()
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        return cmd, "step2_sim_train", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step2_sim" and action_id == "eval":
        checkpoint = str(params.get("checkpoint", "")).strip()
        if not checkpoint:
            raise ValueError("checkpoint is required for step2_sim eval")
        cmd = [
            sys.executable,
            "experiments/step2_sim/isaac_env.py",
            "--mode", "eval",
            "--checkpoint", checkpoint,
            "--save_dir", str(params.get("save_dir", "./data/step2c_isaac")),
        ]
        if _bool_param(params.get("show_gui"), False):
            cmd.append("--show_gui")
        return cmd, "generic", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step2a_synthetic" and action_id == "train":
        cmd = [
            sys.executable,
            "experiments/step2a_synthetic/train_step2a.py",
            "--dataset", str(params.get("dataset", "./data/step2a_synthetic")),
            "--save_dir", str(params.get("save_dir", "./checkpoints/step2a_synthetic")),
            "--epochs", str(int(params.get("epochs", 20))),
            "--batch_size", str(int(params.get("batch_size", 32))),
            "--lr", str(float(params.get("lr", 3e-4))),
            "--d_model", str(int(params.get("d_model", 128))),
            "--n_layers", str(int(params.get("n_layers", 3))),
            "--max_steps", str(int(params.get("max_steps", 12))),
            "--eq_weight", str(float(params.get("eq_weight", 0.1))),
            "--epsilon", str(float(params.get("epsilon", 0.02))),
        ]
        checkpoint = str(params.get("checkpoint", "")).strip()
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        if _bool_param(params.get("no_pde"), False):
            cmd.append("--no_pde")
        return cmd, "step2a_train", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step2d_so101_urdf" and action_id == "viewer":
        urdf = str(params.get("urdf", "")).strip()
        if not urdf:
            raise ValueError("urdf path is required for the SO-101 viewer")
        cmd = [
            sys.executable,
            "experiments/step2d_so101_urdf/so101_urdf_viewer.py",
            "--urdf", urdf,
            "--n_steps", str(int(params.get("n_steps", 200))),
            "--image_size", str(int(params.get("image_size", 64))),
            "--save_dir", str(params.get("save_dir", "./checkpoints/step2d_so101_urdf")),
        ]
        checkpoint = str(params.get("checkpoint", "")).strip()
        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])
        if _bool_param(params.get("show_gui"), False):
            cmd.append("--show_gui")
        if _bool_param(params.get("random_weights"), False):
            cmd.append("--random_weights")
        if _bool_param(params.get("fallback_rerun"), False):
            cmd.append("--fallback_rerun")
        if _bool_param(params.get("no_pde"), False):
            cmd.append("--no_pde")
        return cmd, "generic", {"experiment_id": experiment_id, "action_id": action_id}

    if experiment_id == "step3_lerobot" and action_id in {"benchmark", "collect", "infer"}:
        mode = {"benchmark": "benchmark", "collect": "collect", "infer": "infer"}[action_id]
        cmd = [
            sys.executable,
            "experiments/step3_lerobot/lerobot_inference.py",
            "--mode", mode,
            "--camera_id", str(int(params.get("camera_id", 0))),
            "--robot_port", str(params.get("robot_port", "/dev/ttyUSB0")),
            "--task", str(params.get("task", "pick_place")),
            "--save_dir", str(params.get("save_dir", "./data/step3_lerobot")),
        ]
        if _bool_param(params.get("mock_robot"), True):
            cmd.append("--mock_robot")
        checkpoint = str(params.get("checkpoint", "")).strip()
        if mode == "infer":
            if not checkpoint:
                raise ValueError("checkpoint is required for step3 infer")
            cmd.extend(["--checkpoint", checkpoint])
        return cmd, "step3_benchmark" if mode == "benchmark" else "generic", {"experiment_id": experiment_id, "action_id": action_id}

    raise ValueError(f"Unsupported action: {experiment_id}/{action_id}")


# ── Training job manager ──────────────────────────────────────────────────────

def start_training_job(job_id: str, cmd: list, cwd: str):
    """Launch a training subprocess, stream stdout to WebSocket clients."""
    def _run():
        with job_lock:
            training_jobs[job_id]["status"] = "running"
        emit(job_id, "train_start", {"cmd": " ".join(cmd)})
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=cwd, env=env
            )
            with job_lock:
                training_jobs[job_id]["process"] = proc

            for line in proc.stdout:
                line = line.rstrip()
                emit(job_id, "train_log", {"line": line})
                # Parse epoch metrics from log line
                m = re.search(r"val_dice=(\d+\.\d+)", line)
                if m:
                    emit(job_id, "train_metric",
                         {"val_dice": float(m.group(1))})

            proc.wait()
            status = "done" if proc.returncode == 0 else "error"
            emit(job_id, "train_end",
                 {"status": status, "returncode": proc.returncode})
        except Exception as e:
            emit(job_id, "train_end", {"status": "error", "msg": str(e)})
        finally:
            with job_lock:
                training_jobs[job_id]["status"] = "done"

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def stop_job(job_id: str):
    with job_lock:
        job = training_jobs.get(job_id)
        if job and job.get("process"):
            try:
                job["process"].terminate()
            except Exception:
                pass
            job["status"] = "stopped"
    emit(job_id, "train_end", {"status": "stopped"})


# ── Inference runner ──────────────────────────────────────────────────────────

def run_inference_task(job_id: str, params: dict):
    """Run the canonical medical MSD inference pipeline in a thread."""
    def _run():
        emit(job_id, "infer_start", params)
        try:
            out_dir = Path(params.get("output_dir", str(INFERENCE_DIR / params["task"])))
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(ROOT / "experiments" / "step1b_medical_msd" / "infer_msd.py"),
                "--task",       params["task"],
                "--data_dir",   params["data_dir"],
                "--model_type", params["model_type"],
                "--checkpoint", params["checkpoint"],
                "--case",       params["case"],
                "--output_dir", str(out_dir),
                "--depth",  str(params.get("depth",  128)),
                "--height", str(params.get("height", 128)),
                "--width",  str(params.get("width",  128)),
                "--crop_mode", params.get("crop_mode", "center"),
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, cwd=str(ROOT)
            )
            summary = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith(EVENT_PREFIX):
                    try:
                        payload = json.loads(line[len(EVENT_PREFIX):])
                        event = payload.pop("event", "")
                        if event == "progress":
                            emit(job_id, "infer_progress", payload)
                        elif event == "metrics":
                            emit(job_id, "infer_metrics", payload)
                        elif event == "summary":
                            summary = payload.get("result")
                            emit(job_id, "infer_progress", {"progress": payload.get("progress", 100), "label": payload.get("label", "Termine")})
                        continue
                    except Exception:
                        pass
                emit(job_id, "infer_log", {"line": line})
            proc.wait()
            if proc.returncode == 0:
                # Collect produced files
                stem = params["case"].replace(".nii.gz", "")
                prefix = f"{params['model_type']}_{stem}"
                files = []
                for f in out_dir.glob(f"{prefix}*"):
                    files.append({"name": f.name, "path": str(f),
                                  "type": f.suffix[1:]})
                emit(job_id, "infer_done", {"files": files, "summary": summary})
            else:
                emit(job_id, "infer_error", {"returncode": proc.returncode})
        except Exception as e:
            emit(job_id, "infer_error", {"msg": str(e), "tb": traceback.format_exc()})

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ── HTTP / WebSocket handler ──────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress default access log

    # ── routing ───────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # WebSocket upgrade
        if (self.headers.get("Upgrade", "").lower() == "websocket"
                and path == "/ws"):
            self._handle_ws()
            return

        # API
        if path.startswith("/api/"):
            self._handle_api_get(path, parsed)
            return

        # Static files
        if path.startswith("/static/"):
            self._serve_static(path[8:])
            return

        # Serve file by absolute path (for inference output images)
        if path == "/file" or path.startswith("/file/"):
            self._serve_file(parsed, path[6:] if path.startswith("/file/") else None)
            return

        if path == "/interactive":
            self._serve_interactive()
            return

        if path == "/medical-step0-results":
            self._serve_medical_step0_results()
            return

        # Fallback → SPA index
        self._serve_index()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {}

        if path.startswith("/api/"):
            self._handle_api_post(path, data)
        else:
            self._json(404, {"error": "not found"})

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def _handle_ws(self):
        key = self.headers.get("Sec-WebSocket-Key", "")
        accept = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode()).digest()
        ).decode()
        try:
            self.wfile.write(
                ("HTTP/1.1 101 Switching Protocols\r\n"
                 "Upgrade: websocket\r\n"
                 "Connection: Upgrade\r\n"
                 f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode()
            )
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
            self.close_connection = True
            log.debug("Client disconnected before websocket upgrade completed: %s", exc)
            return
        sock = self.connection
        with ws_lock:
            ws_clients.append(sock)
        # Send initial state
        ws_send(sock, json.dumps({"event": "connected",
                                  "datasets": scan_datasets(),
                                  "checkpoints": scan_checkpoints(),
                                  "outputs": scan_inference_outputs(),
                                  "experiments": build_experiments_payload(),
                                  "scoreboard": build_scoreboard_payload(),
                                  "dataset_explorer": build_explorer_overview()}))
        try:
            while True:
                opcode, data = ws_recv(sock)
                if opcode is None or opcode == 8:
                    break
                if opcode == 1:
                    try:
                        msg = json.loads(data.decode())
                        self._handle_ws_msg(sock, msg)
                    except Exception:
                        pass
        finally:
            with ws_lock:
                if sock in ws_clients:
                    ws_clients.remove(sock)

    def _handle_ws_msg(self, sock, msg: dict):
        action = msg.get("action")
        if action == "ping":
            ws_send(sock, json.dumps({"event": "pong"}))
        elif action == "refresh":
            ws_send(sock, json.dumps({
                "event": "state",
                "datasets": scan_datasets(),
                "checkpoints": scan_checkpoints(),
                "outputs": scan_inference_outputs(),
                "experiments": build_experiments_payload(),
                "scoreboard": build_scoreboard_payload(),
                "dataset_explorer": build_explorer_overview(),
                "jobs": {jid: {"status": j["status"]}
                         for jid, j in training_jobs.items()},
            }))

    # ── REST API GET ──────────────────────────────────────────────────────────

    def _handle_api_get(self, path, parsed):
        qs = parse_qs(parsed.query)

        if path == "/api/state":
            self._json(200, {
                "datasets":    scan_datasets(),
                "checkpoints": scan_checkpoints(),
                "outputs":     scan_inference_outputs(),
                "experiments": build_experiments_payload(),
                "scoreboard":  build_scoreboard_payload(),
                "dataset_explorer": build_explorer_overview(),
                "jobs": {jid: {"status": j["status"]}
                         for jid, j in training_jobs.items()},
            })

        elif path == "/api/experiments":
            self._json(200, build_experiments_payload())

        elif path == "/api/scoreboard":
            self._json(200, build_scoreboard_payload())

        elif path == "/api/dataset-explorer":
            self._json(200, build_explorer_overview())

        elif path == "/api/dataset-explorer/list":
            kind = qs.get("kind", ["dataset"])[0]
            directory = qs.get("dir", [""])[0]
            try:
                self._json(200, list_explorer_dir(kind, directory))
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        elif path == "/api/dataset-explorer/inspect":
            target_path = qs.get("path", [""])[0]
            options = {
                "step": qs.get("step", [None])[0],
                "axis": qs.get("axis", [None])[0],
                "slice_index": qs.get("slice_index", [None])[0],
                "volume_index": qs.get("volume_index", [None])[0],
                "sample_index": qs.get("sample_index", [None])[0],
            }
            options = {key: value for key, value in options.items() if value not in (None, "")}
            try:
                self._json(200, inspect_explorer_file(target_path, options))
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        elif path == "/api/datasets":
            self._json(200, scan_datasets())

        elif path == "/api/checkpoints":
            self._json(200, scan_checkpoints())

        elif path == "/api/outputs":
            self._json(200, scan_inference_outputs())

        elif path == "/api/image":
            # /api/image?path=<abs_path>
            img_path = Path(qs.get("path", [""])[0])
            if img_path.exists() and img_path.suffix in (".png", ".jpg"):
                data = base64.b64encode(img_path.read_bytes()).decode()
                self._json(200, {"data": data, "mime": "image/png"})
            else:
                self._json(404, {"error": "not found"})

        elif path == "/api/jobs":
            self._json(200, {
                jid: {"status": j["status"]}
                for jid, j in training_jobs.items()
            })

        else:
            self._json(404, {"error": "unknown endpoint"})

    # ── REST API POST ─────────────────────────────────────────────────────────

    def _handle_api_post(self, path, data: dict):

        if path == "/api/train":
            job_id = f"train_{int(time.time()*1000)}"
            task    = data.get("task", "Task09_Spleen")
            model   = data.get("model_type", "fluidvla")
            epochs  = int(data.get("epochs", 5))
            batch   = int(data.get("batch_size", 1))
            samples = int(data.get("max_train_samples", 16))
            val_s   = int(data.get("max_val_samples", 4))
            d, h, w = data.get("depth",128), data.get("height",128), data.get("width",128)
            binary  = data.get("binary", True)
            data_dir = data.get("data_dir", default_task_data_dir(task))
            crop_mode = data.get("crop_mode", TASK_TRAIN_PRESETS.get(task, {}).get("crop_mode", "center"))

            if model == "fluidvla":
                script = "experiments/step1b_medical_msd/train_fluidvla_msd.py"
                cmd = [sys.executable, script,
                       "--task", task, "--data_dir", data_dir,
                       "--epochs", str(epochs), "--batch_size", str(batch),
                       "--max_train_samples", str(samples),
                       "--max_val_samples", str(val_s),
                       "--depth", str(d), "--height", str(h), "--width", str(w),
                       "--d_model",   str(data.get("d_model",   32)),
                       "--n_layers",  str(data.get("n_layers",  2)),
                       "--max_steps", str(data.get("max_steps", 6)),
                       "--diffusion_scale", str(data.get("diffusion_scale", 0.08)),
                       "--lr",        str(data.get("lr", 1e-3)),
                       "--crop_mode", str(crop_mode),
                       ]
                if binary: cmd.append("--binary")
                if data.get("no_pde"): cmd.append("--no_pde")
            else:
                script = "experiments/step1b_medical_msd/train_unet3d_msd.py"
                cmd = [sys.executable, script,
                       "--task", task, "--data_dir", data_dir,
                       "--epochs", str(epochs), "--batch_size", str(batch),
                       "--max_train_samples", str(samples),
                       "--max_val_samples", str(val_s),
                       "--depth", str(d), "--height", str(h), "--width", str(w),
                       "--crop_mode", str(crop_mode),
                       ]
                if binary: cmd.append("--binary")

            with job_lock:
                training_jobs[job_id] = {"status": "queued", "process": None}

            start_training_job(job_id, cmd, str(ROOT))
            self._json(200, {"job_id": job_id, "status": "started"})

        elif path == "/api/experiments/run":
            experiment_id = data.get("experiment_id", "")
            action_id = data.get("action_id", "")
            params = data.get("params", {}) or {}
            if experiment_id not in EXPERIMENTS:
                self._json(400, {"error": f"unknown experiment_id: {experiment_id}"})
                return

            if experiment_id == "step1b_medical_msd" and action_id == "infer":
                task = str(params.get("task", "Task01_BrainTumour"))
                model_type = normalize_infer_model_type(params.get("model_type", "fluidvla"), params.get("checkpoint", ""))
                checkpoint = str(params.get("checkpoint", ""))
                err = validate_checkpoint_for_task(checkpoint, task, model_type)
                if err:
                    self._json(400, {"error": err})
                    return
                case_name = str(params.get("case", "")).strip()
                if not case_name:
                    self._json(400, {"error": "case is required for medical inference"})
                    return
                job_id = f"exp_{experiment_id}_{action_id}_{int(time.time()*1000)}"
                with job_lock:
                    training_jobs[job_id] = {"status": "running", "process": None}
                infer_params = {
                    "task": task,
                    "data_dir": str(params.get("data_dir", default_task_data_dir(task))),
                    "model_type": model_type,
                    "checkpoint": checkpoint,
                    "case": case_name,
                    "output_dir": str(params.get("output_dir", str(INFERENCE_DIR / task))),
                    "depth": int(params.get("depth", 128)),
                    "height": int(params.get("height", 128)),
                    "width": int(params.get("width", 128)),
                    "crop_mode": str(params.get("crop_mode", "center")),
                }
                run_inference_task(job_id, infer_params)
                self._json(200, {"job_id": job_id, "status": "started"})
                return

            try:
                cmd, parser_key, meta = build_experiment_command(experiment_id, action_id, params)
            except Exception as exc:
                self._json(400, {"error": str(exc)})
                return

            job_id = f"exp_{experiment_id}_{action_id}_{int(time.time()*1000)}"
            with job_lock:
                training_jobs[job_id] = {"status": "queued", "process": None, **meta}
            start_command_job(job_id, cmd, str(ROOT), parser_key, meta)
            self._json(200, {"job_id": job_id, "status": "started"})

        elif path == "/api/scoreboard":
            try:
                entry = upsert_manual_scoreboard(data)
            except Exception as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"entry": entry, "scoreboard": build_scoreboard_payload()})

        elif path == "/api/scoreboard/delete":
            entry_id = str(data.get("id", "")).strip()
            if not entry_id:
                self._json(400, {"error": "id is required"})
                return
            removed = delete_manual_scoreboard(entry_id)
            if not removed:
                self._json(404, {"error": f"scoreboard entry not found: {entry_id}"})
                return
            self._json(200, {"deleted": entry_id, "scoreboard": build_scoreboard_payload()})

        elif path == "/api/train/multi":
            # Train multiple tasks sequentially on one model
            tasks   = data.get("tasks", [])
            model   = data.get("model_type", "fluidvla")
            job_id  = f"multi_{int(time.time()*1000)}"
            epochs  = int(data.get("epochs", 5))
            batch   = int(data.get("batch_size", 1))
            samples = int(data.get("max_train_samples", 16))
            val_s   = int(data.get("max_val_samples", 4))
            lr      = str(data.get("lr", 1e-3))
            binary  = data.get("binary", True)
            d       = int(data.get("depth", 128))
            h       = int(data.get("height", 128))
            w       = int(data.get("width", 128))
            crop_mode = data.get("crop_mode", "center")

            def _multi_run():
                emit(job_id, "train_start", {"tasks": tasks, "model": model})
                for task in tasks:
                    d_dir = data.get("data_dir", default_task_data_dir(task))
                    if model == "fluidvla":
                        script = "experiments/step1b_medical_msd/train_fluidvla_msd.py"
                        cmd = [sys.executable, script,
                               "--task", task, "--data_dir", d_dir,
                               "--epochs", str(epochs),
                               "--batch_size", str(batch),
                               "--max_train_samples", str(samples),
                               "--max_val_samples", str(val_s),
                               "--depth", str(d), "--height", str(h), "--width", str(w),
                               "--d_model", str(data.get("d_model", 32)),
                               "--n_layers", str(data.get("n_layers", 2)),
                               "--max_steps", str(data.get("max_steps", 6)),
                               "--diffusion_scale", str(data.get("diffusion_scale", 0.08)),
                               "--lr", lr, "--crop_mode", str(crop_mode)]
                        if binary: cmd.append("--binary")
                        if data.get("no_pde"): cmd.append("--no_pde")
                    else:
                        script = "experiments/step1b_medical_msd/train_unet3d_msd.py"
                        cmd = [sys.executable, script,
                               "--task", task, "--data_dir", d_dir,
                               "--epochs", str(epochs),
                               "--batch_size", str(batch),
                               "--max_train_samples", str(samples),
                               "--max_val_samples", str(val_s),
                               "--depth", str(d), "--height", str(h), "--width", str(w), "--crop_mode", str(crop_mode)]
                        if binary: cmd.append("--binary")
                    emit(job_id, "train_log",
                         {"line": f"\n=== Starting {task} ===\n"})
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, cwd=str(ROOT))
                    with job_lock:
                        training_jobs[job_id]["process"] = proc
                    for line in proc.stdout:
                        emit(job_id, "train_log", {"line": line.rstrip()})
                        m = re.search(r"val_dice=(\d+\.\d+)", line)
                        if m:
                            emit(job_id, "train_metric",
                                 {"task": task, "val_dice": float(m.group(1))})
                    proc.wait()
                emit(job_id, "train_end", {"status": "done", "tasks": tasks})
                with job_lock:
                    training_jobs[job_id]["status"] = "done"

            with job_lock:
                training_jobs[job_id] = {"status": "running", "process": None}
            threading.Thread(target=_multi_run, daemon=True).start()
            self._json(200, {"job_id": job_id, "status": "started"})

        elif path == "/api/stop":
            job_id = data.get("job_id")
            if job_id:
                stop_job(job_id)
                self._json(200, {"stopped": job_id})
            else:
                self._json(400, {"error": "no job_id"})

        elif path == "/api/infer":
            task = data.get("task", "Task01_BrainTumour")
            model_type = normalize_infer_model_type(data.get("model_type", "fluidvla"), data.get("checkpoint", ""))
            ckpt = data.get("checkpoint", "")
            err = validate_checkpoint_for_task(ckpt, task, model_type)
            if err:
                self._json(400, {"error": err}); return
            job_id = f"infer_{int(time.time()*1000)}"
            with job_lock:
                training_jobs[job_id] = {"status": "running", "process": None}
            params = dict(data)
            params["model_type"] = model_type
            params.setdefault("data_dir", default_task_data_dir(task))
            run_inference_task(job_id, params)
            self._json(200, {"job_id": job_id})

        elif path == "/api/infer/upload":
            # Upload NIfTI + checkpoint path, run inference, return base64 PNG
            # data: {nifti_b64, filename, checkpoint, model_type, task, ...}
            import tempfile
            nifti_b64 = data.get("nifti_b64", "")
            filename  = data.get("filename", "uploaded.nii.gz")
            checkpoint = data.get("checkpoint", "")
            model_type = normalize_infer_model_type(data.get("model_type", "fluidvla"), checkpoint)
            task       = data.get("task", "Task01_BrainTumour")
            err = validate_checkpoint_for_task(checkpoint, task, model_type)
            if err:
                self._json(400, {"error": err}); return

            tmp_dir = Path(tempfile.mkdtemp())
            # Create fake MSD structure
            img_dir = tmp_dir / "imagesTr"; img_dir.mkdir()
            lbl_dir = tmp_dir / "labelsTr"; lbl_dir.mkdir()
            nifti_bytes = base64.b64decode(nifti_b64)
            nifti_path = img_dir / filename
            nifti_path.write_bytes(nifti_bytes)
            # Create dummy label
            import numpy as np
            try:
                import nibabel as nib
                dummy = nib.Nifti1Image(np.zeros((10,10,10), dtype=np.int16), np.eye(4))
                nib.save(dummy, str(lbl_dir / filename))
            except Exception as e:
                self._json(500, {"error": f"nibabel needed: {e}"}); return

            job_id = f"upload_{int(time.time()*1000)}"
            with job_lock:
                training_jobs[job_id] = {"status": "running", "process": None}

            out_dir = INFERENCE_DIR / "uploads" / job_id
            out_dir.mkdir(parents=True, exist_ok=True)

            params = {
                "task":        task,
                "data_dir":    str(tmp_dir),
                "model_type":  model_type,
                "checkpoint":  checkpoint,
                "case":        filename,
                "output_dir":  str(out_dir),
                "depth":  data.get("depth",  128),
                "height": data.get("height", 128),
                "width":  data.get("width",  128),
                "crop_mode": data.get("crop_mode", "center"),
            }
            run_inference_task(job_id, params)
            self._json(200, {"job_id": job_id})

        else:
            self._json(404, {"error": "unknown endpoint"})

    # ── Static / index ────────────────────────────────────────────────────────

    def _write_response_bytes(self, body):
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
            win_error = getattr(exc, "winerror", None)
            if isinstance(exc, OSError) and win_error not in {None, 10053, 10054}:
                raise
            self.close_connection = True
            log.debug("Client disconnected while sending response: %s", exc)

    def _serve_index(self):
        index = STATIC / "index.html"
        if index.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self._write_response_bytes(index.read_bytes())
        else:
            self.send_response(404)
            self.end_headers()
            self._write_response_bytes("<h1>404 - index.html not found</h1>".encode("utf-8"))

    def _serve_static(self, rel_path):
        full = STATIC / rel_path
        if full.exists() and full.is_file():
            mime, _ = mimetypes.guess_type(str(full))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self._write_response_bytes(full.read_bytes())
        else:
            self._json(404, {"error": "not found"})

    def _serve_interactive(self):
        if INTERACTIVE_PAGE.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self._write_response_bytes(INTERACTIVE_PAGE.read_bytes())
        else:
            self._json(404, {"error": "interactive page not found"})

    def _serve_medical_step0_results(self):
        if MEDICAL_STEP0_RESULTS.exists():
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.end_headers()
            self._write_response_bytes(MEDICAL_STEP0_RESULTS.read_bytes())
        else:
            self._json(404, {"error": "medical step0 results not found"})

    def _serve_file(self, parsed, raw_path=None):
        """Serve an arbitrary file (inference output images)."""
        path_arg = parse_qs(parsed.query).get("path", [None])[0]
        download = parse_qs(parsed.query).get("download", [""])[0].lower() in {"1", "true", "yes"}
        candidate = path_arg if path_arg is not None else raw_path
        if not candidate:
            self._json(400, {"error": "missing path"})
            return

        candidate = unquote(candidate)
        if re.match(r"^/[A-Za-z]:[/\\]", candidate):
            candidate = candidate[1:]

        p = Path(candidate)
        if p.exists() and p.is_file():
            mime, _ = mimetypes.guess_type(str(p))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
            self.end_headers()
            self._write_response_bytes(p.read_bytes())
        else:
            self._json(404, {"error": "not found"})

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._write_response_bytes(body)


# ── Threaded server ───────────────────────────────────────────────────────────

class ThreadedServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    STATIC.mkdir(exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    INFERENCE_DIR.mkdir(exist_ok=True)

    server = ThreadedServer((args.host, args.port), Handler)
    log.info(f"FluidVLA Platform running on http://{args.host}:{args.port}")
    log.info(f"Open http://localhost:{args.port} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
