"""Compatibility facade for high-level FluidVLA models."""

try:
    from .video_models import FluidBotVideo
    from .vision_models import FluidBotClassifier, PatchEmbed
    from .vla_models import ActionHead, SpatialActionHead, FluidBotVLA
except ImportError:
    from video_models import FluidBotVideo
    from vision_models import FluidBotClassifier, PatchEmbed
    from vla_models import ActionHead, SpatialActionHead, FluidBotVLA

__all__ = [
    "ActionHead",
    "FluidBotClassifier",
    "FluidBotVLA",
    "FluidBotVideo",
    "PatchEmbed",
    "SpatialActionHead",
]


if __name__ == "__main__":
    print("fluid_model.py is now a compatibility facade.")
    print("Use src.core for stable imports or the split implementation modules in src/core.")