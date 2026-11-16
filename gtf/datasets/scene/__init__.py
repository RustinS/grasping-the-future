"""Scene preprocessing: per-source depth loaders, the scene feature pipeline, DINOv3 features.

Exports:
    - AnchorDepthResult, load_anchor_depth: per-source depth + RGB + masks of one canonical frame
    - prepare_scene_cloud, batch_fps_indices, finish_scene_features: the scene feature pipeline
    - DinoV3Extractor: DINOv3 ViT-L/16 patch features for the semantic channel
"""

from gtf.datasets.scene.depth_sources import AnchorDepthResult, load_anchor_depth
from gtf.datasets.scene.dinov3_semantic import DinoV3Extractor
from gtf.datasets.scene.scene_pipeline import batch_fps_indices, finish_scene_features, prepare_scene_cloud

__all__ = [
    "AnchorDepthResult",
    "DinoV3Extractor",
    "batch_fps_indices",
    "finish_scene_features",
    "load_anchor_depth",
    "prepare_scene_cloud",
]
