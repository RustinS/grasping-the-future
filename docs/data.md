# Data format and conventions

## A segment

Every source (the motion-capture loaders in `gtf/datasets/` and the extraction pipeline in `gtf/extraction/`) writes one
folder per segment, `results/<segment>/`, with the trajectory in `canonical/canonical.npz` and its description in
`canonical/metadata.json`. The trajectory holds two MANO hands and one rigid object at the segment's keyframes (about 10
per second for extracted video, the dataset's rate thinned to 30 Hz for motion capture).

| Key | Shape | Meaning |
|---|---|---|
| `frame_indices` | (T,) int | Video frame of each keyframe |
| `frame_times_s` | (T,) float64 | Time of each keyframe in seconds (decoded video timestamps), written by the extraction pipeline; the model reads time steps from it, and uses a uniform grid at `fps` / median frame step where it is absent (motion capture) |
| `fps` | () | Frame rate of the source video |
| `hand_{left,right}_orient` | (T, 3) | MANO global orientation, axis-angle |
| `hand_{left,right}_pose` | (T, 45) | MANO finger articulation, 15 joints in axis-angle |
| `hand_{left,right}_cam_t` | (T, 3) | MANO translation (manopth convention), meters |
| `hand_{left,right}_betas` | (10,) | MANO shape |
| `hand_{left,right}_valid` | (T,) bool | Whether the hand is present at the keyframe; absent frames are zero-filled |
| `hand_{left,right}_verts` | (T, 778, 3) | MANO vertices (for inspection; the model recomputes them) |
| `obj_canonical_verts`, `obj_faces` | (V, 3), (F, 3) | Object mesh in its canonical frame, unit scale |
| `obj_scale` | (3,) | Per-axis scale of the canonical mesh, meters |
| `obj_rotation`, `obj_translation` | (T, 3), (T, 3) | Object pose, axis-angle and meters: posed vertices = R(obj_rotation) · (obj_canonical_verts · obj_scale) + obj_translation |
| `obj_verts` | (T, V, 3) | Posed object vertices (for inspection) |
| `gravity_direction` | (3,) | Unit gravity vector in the camera frame (optional; `metadata.json`: `has_gravity`) |
| `support_plane_height` | () | Height of the support surface along gravity, meters (optional; `has_support_plane`) |

Motion-capture segments with a moving camera also store world-frame fields (`*_world`), the per-frame camera
extrinsics (`camera_extrinsics`, world to camera) and gravity in the world frame (`gravity_world`); the model re-anchors
them to the camera of the present frame at training and evaluation time.

## Your own segments

`python -m gtf.predict` runs the model on segments that did not come from our loaders or pipeline. Each segment is a
folder `DIR/<segment>/` with

- `canonical/canonical.npz` with every key of the table above (the two `gravity`/`support` keys are optional; with a
  gravity vector the model works in a gravity-aligned frame, as in training);
- `canonical/metadata.json` with at least `{"segment_id": "<segment>", "source_dataset": "<any name>", "has_gravity":
  true|false, "has_support_plane": true|false}`;
- `video.mp4`, the clip whose frames `frame_indices` index (the scene input: MoGe-2 depth and DINOv3 features of each
  keyframe).

The command builds the manifest, the scene caches and the object embeddings next to the segments on its first run
(README, "Inference on new data").

## Conventions

- **Camera frame.** Positions are in the OpenCV camera frame of the segment's anchor camera: x right, y down, z forward,
  meters. Gravity is stored per segment (`metadata.json`: `has_gravity`, `gravity_source`; the vector in the canonical
  or the scene cache) and the model rotates every window into a gravity-aligned frame.
- **Left hands.** `mano_fk_verts(..., "left")` evaluates the right MANO model and mirrors the result by
  M = diag(-1, 1, 1): left-hand orientation and translation live in mirrored coordinates. A rigid transform (R, t)
  applied to left-hand parameters must therefore be conjugated (R → M R M, t → M t) and pivot around the right model's
  root joint: use `gtf.utils.hand_rigid.transform_hand_rigid`, never a raw rotation.
- **Shaped pivots.** MANO's global orientation rotates about the root joint J0(betas) of the shaped hand; every
  re-expression of hand parameters uses the shaped root joint.
- **Penetration and contact** use the winding-number sign of the object mesh (`gtf/extraction/gates/hoi_contact.py`).
- **Interaction windows.** A segment can export several trajectories: children named `<segment>__iwN` (interaction
  windows) and `<segment>__shotN` (shots after a cut) are segments of their own.

## The corpus

`scripts/build_corpus.sh CONFIG` runs these stages in order (each can be run alone):

| Stage | Command | Output |
|---|---|---|
| `ingest` | `python -m gtf.datasets.<name> --start-from convert` for the eight datasets | `results/<segment>/canonical/` |
| `split` | `python -m gtf.datasets.split_interaction_segments` | `__iwN` interaction windows of long segments |
| `manifest` | `python -m gtf.datasets.build_manifest` | `datasets/canonical_manifest.json` |
| `pca` | `python -m gtf.datasets.scene.fit_pca` (kept when the file exists) | `datasets/scene_dinov3_pca.pt` |
| `scene` | `python -m gtf.datasets.scene.preprocess` | per-frame scene caches `results/<segment>/canonical/scene_cache_v2.h5` |
| `gravity` | `python -m gtf.datasets.estimate_scene_gravity` | gravity of segments without a native one (from their scene); their scene caches are removed |
| `scene` | again | rebuilds the removed scene caches with the new gravity |
| `manifest2` | `python -m gtf.datasets.build_manifest` | the manifest with the scene caches |
| `uni3d` | `python -m gtf.datasets.build_uni3d_cache --config CONFIG` | object embeddings at `data.uni3d_cache_path` |
| `stats` | `python -m gtf.data.compute_stats --config CONFIG` | normalization statistics at `data.stats_path` |

Action100M segments come from the extraction pipeline, run into the same `results/` folder before `manifest`; the
quality gates decide which of them enter the manifest.

**Scene caches.** For every keyframe a scene cache stores a point cloud of the scene around the hands and the object
(from sensor depth where the dataset has it, otherwise from MoGe-2), cropped and centered, with DINOv3 features reduced
by the PCA basis, the distances of each point to the object and the hands, and the support plane. The scene encoder
(LitePT) reads the row of the last fully observed frame.

**Normalization statistics** are window-sampled: they are computed over windows drawn the way training draws them
(same window length, sampler and task mix), so a change of the corpus or of the task mix needs new statistics.
