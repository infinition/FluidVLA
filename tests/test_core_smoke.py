from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from fluidvla.core import FluidBotClassifier, FluidBotMedical3D, FluidBotVLA, FluidBotVideo


def test_core_public_api_imports():
    assert FluidBotClassifier.__name__ == "FluidBotClassifier"
    assert FluidBotVideo.__name__ == "FluidBotVideo"
    assert FluidBotVLA.__name__ == "FluidBotVLA"
    assert FluidBotMedical3D.__name__ == "FluidBotMedical3D"


def test_classifier_smoke_forward():
    model = FluidBotClassifier(in_channels=1, num_classes=10, d_model=16, n_layers=1, max_steps=2, min_steps=1)
    x = torch.randn(2, 1, 32, 32)
    logits, info = model(x)

    assert logits.shape == (2, 10)
    assert "avg_steps" in info
    assert "layer_steps" in info


def test_video_smoke_forward():
    model = FluidBotVideo(
        in_channels=1,
        d_model=16,
        n_layers=1,
        max_steps=2,
        min_steps=1,
        patch_size=4,
        num_classes=5,
    )
    x = torch.randn(2, 1, 4, 32, 32)
    out = model(x)

    assert out["features"].shape[:3] == (2, 16, 4)
    assert out["logits"].shape == (2, 5)
    assert len(out["info"]) == 1


def test_vla_smoke_forward():
    model = FluidBotVLA(
        in_channels=3,
        d_model=16,
        n_layers=1,
        patch_size=4,
        action_dim=7,
        proprio_dim=8,
        max_steps=2,
        min_steps=1,
        n_frames=4,
    )
    frames = torch.randn(2, 3, 4, 32, 32)
    proprio = torch.randn(2, 8)
    out = model(frames, proprio)

    assert out["actions"].shape == (2, 7)
    assert out["features"].shape == (2, 16)
    assert len(out["info"]) == 1


def test_medical_smoke_forward():
    model = FluidBotMedical3D(
        in_channels=4,
        n_classes=3,
        d_model=16,
        n_layers=1,
        patch_size=2,
        max_steps=2,
        min_steps=1,
    )
    x = torch.randn(1, 4, 16, 32, 32)
    out = model(x)

    assert out["logits"].shape == (1, 3, 16, 32, 32)
    assert out["features"].shape[1] == 16
    assert len(out["info"]) == 1
