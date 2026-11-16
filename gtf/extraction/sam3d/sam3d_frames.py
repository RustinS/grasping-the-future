"""Per-frame SAM 3D work: reconstruction or pose for one frame, overlays, the MoGe depth cache, per-segment metadata."""

import os
import json
import numpy as np
import torch
from PIL import Image as PILImage
from pathlib import Path
from gtf.utils import rprint as print
from gtf.utils.npz import savez_compressed_level
from gtf.extraction.sam3d.sam3d_inference import run_inference, run_pose_only_inference, run_stage1_pose_batch


def render_gaussian_overlay(output, original_rgb, intrinsics):
    """Render Gaussian splat overlay. Returns (H, W, 3) uint8."""
    from copy import deepcopy

    from pytorch3d.transforms import (
        matrix_to_quaternion,
        quaternion_multiply,
        quaternion_to_matrix,
    )
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    from sam3d_objects.model.backbone.tdfy_dit.renderers import GaussianRenderer
    from sam3d_objects.pipeline.layout_post_optimization_utils import (
        flip_coords_pytorch3d_to_opencv,
    )

    gs = deepcopy(output["gaussian"][0])
    H, W = original_rgb.shape[:2]
    device = gs._xyz.device

    # Transform to camera space (following get_gs_transformed)
    R = quaternion_to_matrix(output["rotation"])
    tfm = compose_transform(scale=output["scale"], rotation=R, translation=output["translation"])

    gs.from_xyz(tfm.transform_points(gs.get_xyz.unsqueeze(0))[0])
    scale_factor = output["scale"][0]
    gs._scaling = gs._scaling + torch.log(scale_factor.expand_as(gs._scaling))

    # Extract pure rotation from composed transform and rotate orientations
    tfm_matrix = tfm.get_matrix()[0]
    rot_matrix = tfm_matrix[:3, :3]
    pure_rot = rot_matrix / rot_matrix.norm(dim=0)[None, :]
    rot_quat = matrix_to_quaternion(pure_rot[None])
    gs.from_rotation(quaternion_multiply(rot_quat.expand_as(gs.get_rotation), gs.get_rotation))

    flip_coords_pytorch3d_to_opencv(gs)

    # Render at square resolution with intrinsics adjusted for square canvas
    resolution = max(H, W)
    pad_top = (resolution - H) // 2
    pad_left = (resolution - W) // 2
    sq_intrinsics = intrinsics.clone().to(device)
    sq_intrinsics[0, 0] *= W / resolution
    sq_intrinsics[1, 1] *= H / resolution
    sq_intrinsics[0, 2] = (sq_intrinsics[0, 2] * W + pad_left) / resolution
    sq_intrinsics[1, 2] = (sq_intrinsics[1, 2] * H + pad_top) / resolution

    renderer = GaussianRenderer(
        {
            "resolution": resolution,
            "near": 0.01,
            "far": 100.0,
            "ssaa": 1,
            "bg_color": (0.0, 0.0, 0.0),
            "backend": "gsplat",
        }
    )
    extrinsics = torch.eye(4, device=device)
    rendered = renderer.render(gs, extrinsics, sq_intrinsics)

    color = rendered["color"][:, pad_top : pad_top + H, pad_left : pad_left + W]
    alpha = rendered["alpha"][:, pad_top : pad_top + H, pad_left : pad_left + W]
    orig_t = torch.from_numpy(original_rgb).permute(2, 0, 1).float().to(device) / 255.0
    composite = alpha * color + (1 - alpha) * orig_t
    return (composite.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def render_mesh_overlay(output, original_rgb, intrinsics):
    """Render mesh overlay. Returns (H, W, 3) uint8."""
    from pytorch3d.renderer import (
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.renderer.cameras import PerspectiveCameras
    from pytorch3d.structures import Meshes
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    glb = output.get("glb")
    if glb is None:
        return original_rgb.copy()

    H, W = original_rgb.shape[:2]
    device = "cuda"

    verts = torch.tensor(glb.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(glb.faces, dtype=torch.int64, device=device)

    # Undo to_glb() z-up→y-up rotation so pose decoder outputs align
    glb_to_raw = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    verts = verts @ glb_to_raw

    R = quaternion_to_matrix(output["rotation"])
    l2c = compose_transform(scale=output["scale"], rotation=R, translation=output["translation"])
    verts_camera = l2c.transform_points(verts.unsqueeze(0))[0]

    if hasattr(glb.visual, "vertex_colors") and glb.visual.vertex_colors is not None:
        vc = torch.tensor(glb.visual.vertex_colors[:, :3], dtype=torch.float32, device=device) / 255.0
    else:
        vc = torch.ones_like(verts_camera) * 0.7
    textures = TexturesVertex(verts_features=[vc])
    meshes = Meshes(verts=[verts_camera], faces=[faces], textures=textures)

    R_cam = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
    T_cam = torch.zeros(1, 3, device=device)

    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx * W, fy * H]], device=device),
        principal_point=torch.tensor([[cx * W, cy * H]], device=device),
        R=R_cam,
        T=T_cam,
        image_size=torch.tensor([[H, W]], device=device),
        in_ndc=False,
        device=device,
    )

    raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    rendered = renderer(meshes)[0].detach().cpu().numpy()
    color_mesh = rendered[..., :3]
    alpha_mesh = rendered[..., 3:4]
    orig_f = original_rgb.astype(np.float32) / 255.0
    composite = alpha_mesh * color_mesh + (1 - alpha_mesh) * orig_f
    return (composite * 255).clip(0, 255).astype(np.uint8)


def render_overlay_frame(output, original_rgb, intrinsics):
    """Side-by-side: Original | Gaussian overlay | Mesh overlay."""
    try:
        gs_overlay = render_gaussian_overlay(output, original_rgb, intrinsics)
    except Exception as e:
        print(f"    WARNING: Gaussian overlay failed: {e}")
        gs_overlay = original_rgb.copy()

    try:
        mesh_overlay = render_mesh_overlay(output, original_rgb, intrinsics)
    except Exception as e:
        print(f"    WARNING: Mesh overlay failed: {e}")
        mesh_overlay = original_rgb.copy()

    return np.concatenate([original_rgb, gs_overlay, mesh_overlay], axis=1)


def _extract_pose_dict(output, extra_keys=()):
    """Extract rotation/translation/scale (+ optional extra keys) as JSON-serializable dict."""
    pose = {}
    for key in ("rotation", "translation", "scale"):
        if key in output and output[key] is not None:
            val = output[key]
            if hasattr(val, "detach"):
                val = val.detach().cpu()
            if hasattr(val, "numpy"):
                val = val.numpy()
            pose[key] = val.tolist()
    for key in extra_keys:
        if key in output:
            val = output[key]
            pose[key] = val.item() if hasattr(val, "item") else val
    return pose


class MogeCacheCollector:
    """Collect keyframe MoGe arrays and commit one atomic segment-level NPZ."""

    def __init__(self, sam3d_dir):
        self.path = Path(sam3d_dir) / "moge_keyframes.npz"
        self.records = {}

    def add(self, frame_index, depth, intrinsics_px, mask):
        self.records[int(frame_index)] = {
            "depth": depth,
            "intrinsics_px": intrinsics_px,
            "mask": mask,
        }

    def commit(self):
        if not self.records:
            self.path.unlink(missing_ok=True)
            return
        arrays = {
            "frame_indices": np.asarray(sorted(self.records), dtype=np.int32),
        }
        for frame_index, record in sorted(self.records.items()):
            for key, value in record.items():
                arrays[f"{key}_{frame_index:05d}"] = value
        savez_compressed_level(
            self.path,
            compresslevel=1,
            allow_pickle=False,
            **arrays,
        )


def _save_moge_cache(
    frame_dir,
    output,
    image_shape,
    *,
    frame_index=None,
    collector=None,
):
    """Save raw MoGe results for exact reuse by object pose."""
    depth = output.get("moge_depth")
    mask = output.get("moge_mask")
    intrinsics = output.get("intrinsics")
    if depth is None or mask is None or intrinsics is None:
        return False

    def as_numpy(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    H, W = image_shape[:2]
    depth = np.squeeze(as_numpy(depth)).astype(np.float32, copy=False)
    mask = np.squeeze(as_numpy(mask)).astype(np.uint8, copy=False)
    intrinsics = np.squeeze(as_numpy(intrinsics)).astype(np.float32, copy=True)
    if depth.shape != (H, W) or mask.shape != (H, W) or intrinsics.shape != (3, 3):
        print(f"    WARNING: not caching MoGe output with shapes depth={depth.shape}, mask={mask.shape}, K={intrinsics.shape}")
        return False
    intrinsics[0, 0] *= W
    intrinsics[1, 1] *= H
    intrinsics[0, 2] *= W
    intrinsics[1, 2] *= H
    if collector is not None and frame_index is not None:
        collector.add(frame_index, depth, intrinsics, mask)
    else:
        savez_compressed_level(
            os.path.join(frame_dir, "moge.npz"),
            compresslevel=1,
            allow_pickle=False,
            depth=depth,
            intrinsics_px=intrinsics,
            mask=mask,
            frame_shape=np.asarray([H, W], dtype=np.int32),
        )
    return True


def process_frame(
    inference_obj,
    frame_rgb,
    mask,
    frame_dir,
    seed=42,
    visualize=False,
    frame_index=None,
    moge_collector=None,
):
    """Process a single frame through the full SAM3D pipeline (reconstruction and pose); writes splat.ply,
    mesh.glb and pose.json into frame_dir.

    Returns (pose_dict, overlay_frame_or_None, output_dict).
    """
    os.makedirs(frame_dir, exist_ok=True)

    output = run_inference(inference_obj, frame_rgb, mask, seed=seed)

    gs = output.get("gs") or (output["gaussian"][0] if "gaussian" in output else None)
    if gs is not None:
        gs.save_ply(os.path.join(frame_dir, "splat.ply"))
    glb = output.get("glb")
    if glb is not None:
        glb.export(os.path.join(frame_dir, "mesh.glb"))

    pose = _extract_pose_dict(output)
    with open(os.path.join(frame_dir, "pose.json"), "w") as f:
        json.dump(pose, f, indent=2)
    _save_moge_cache(
        frame_dir,
        output,
        frame_rgb.shape,
        frame_index=frame_index,
        collector=moge_collector,
    )

    overlay = render_overlay_frame(output, frame_rgb, output["intrinsics"]) if visualize else None
    return pose, overlay, output


def process_pose_frame(
    pipeline,
    frame_rgb,
    mask,
    frame_dir,
    frozen_gs,
    frozen_glb,
    seed=42,
    visualize=False,
    stage1_steps=20,
    frame_index=None,
    moge_collector=None,
):
    """Process a pose-only frame: Stage 1 pose of this frame, rendered with the anchor's frozen reconstruction.

    Its pose is returned (and stored in the segment metadata), not written to frame_dir.
    """
    os.makedirs(frame_dir, exist_ok=True)

    for stale in ("splat.ply", "mesh.glb", "pose.json"):
        stale_path = os.path.join(frame_dir, stale)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    rgba = np.concatenate([frame_rgb[..., :3], (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
    output = run_pose_only_inference(pipeline, rgba, seed=seed, stage1_steps=stage1_steps)

    output["gaussian"] = [frozen_gs]
    output["glb"] = frozen_glb

    pose = _extract_pose_dict(output)
    _save_moge_cache(
        frame_dir,
        output,
        frame_rgb.shape,
        frame_index=frame_index,
        collector=moge_collector,
    )

    overlay = render_overlay_frame(output, frame_rgb, output["intrinsics"]) if visualize else None
    return pose, overlay


def process_pose_frames_batch(
    pipeline,
    frame_items,
    seed=42,
    stage1_steps=10,
    moge_collector=None,
):
    """Process pose-only frames in one Stage 1 call (like process_pose_frame, without overlays).

    Used only when visualization is disabled: the overlays need the frozen geometry per frame and use the
    singleton path (process_pose_frame).
    """
    rgba_batch = []
    for item in frame_items:
        os.makedirs(item["frame_dir"], exist_ok=True)
        for stale in ("splat.ply", "mesh.glb", "pose.json"):
            stale_path = os.path.join(item["frame_dir"], stale)
            if os.path.exists(stale_path):
                os.remove(stale_path)
        rgba_batch.append(
            np.concatenate(
                [
                    item["frame_rgb"][..., :3],
                    (item["frame_mask"].astype(np.uint8) * 255)[..., None],
                ],
                axis=-1,
            )
        )

    outputs = run_stage1_pose_batch(pipeline, rgba_batch, seed=seed, stage1_steps=stage1_steps)
    poses = []
    for item, output in zip(frame_items, outputs):
        pose = _extract_pose_dict(output)
        _save_moge_cache(
            item["frame_dir"],
            output,
            item["frame_rgb"].shape,
            frame_index=item["frame_index"],
            collector=moge_collector,
        )
        poses.append(pose)
    return poses


def _load_frame(vr, masks, frame_idx, total_frames):
    """Load and prepare a frame + mask pair. Returns (frame_rgb, frame_mask, mask_pixels) or None."""
    if frame_idx >= total_frames:
        return None
    # decord returns torch tensors once another stage has set its bridge, else NDArrays
    _frame = vr[frame_idx]
    frame_rgb = _frame.numpy() if hasattr(_frame, "numpy") else _frame.asnumpy()  # (H, W, 3) uint8
    frame_mask = masks[frame_idx]  # (H, W) uint8
    fh, fw = frame_rgb.shape[:2]
    mh, mw = frame_mask.shape[:2]
    if (fh, fw) != (mh, mw):
        mask_pil = PILImage.fromarray(frame_mask)
        mask_pil = mask_pil.resize((fw, fh), PILImage.NEAREST)
        frame_mask = np.array(mask_pil)
    mask_pixels = int(frame_mask.sum())
    return frame_rgb, frame_mask, mask_pixels


def _save_overlay_video(overlays, overlay_frame_indices, out_dir, select_fps):
    """Sort overlays by frame index, pad to even dims, write MP4. Returns (path_or_None, sorted_indices, count)."""
    if not overlays:
        return None, overlay_frame_indices, 0
    sorted_pairs = sorted(zip(overlay_frame_indices, overlays), key=lambda x: x[0])
    sorted_indices = [p[0] for p in sorted_pairs]
    import imageio.v3 as iio

    combined = np.stack([p[1] for p in sorted_pairs], axis=0)
    h, w = combined.shape[1:3]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        combined = np.pad(combined, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    vis_path = os.path.join(out_dir, "vis_overlay.mp4")
    overlay_fps = min(select_fps, 10)
    iio.imwrite(vis_path, combined, fps=overlay_fps, codec="libx264")
    print(f"  Saved overlay video: {vis_path} ({overlay_fps:.0f} fps)")
    return vis_path, sorted_indices, len(sorted_pairs)


def _save_segment_metadata(
    sam3d_dir,
    results_list,
    total_frames,
    video_fps,
    select_fps,
    keyframe_indices,
    num_succeeded,
    num_failed,
    overlay_path,
    overlay_frame_indices,
    num_overlay_frames,
    seed,
    anchor_frame_index=None,
    tracking_mode=True,
    extra_fields=None,
):
    """Write metadata.json for a segment."""
    meta = {
        "total_video_frames": total_frames,
        "video_fps": round(video_fps, 2),
        "select_fps": select_fps,
        "num_keyframes_selected": len(keyframe_indices),
        "num_keyframes_succeeded": num_succeeded,
        "num_keyframes_failed": num_failed,
        "keyframe_indices": keyframe_indices,
        "anchor_frame_index": anchor_frame_index,
        "tracking_mode": tracking_mode,
        "seed": seed,
        "vis_overlay_frames": num_overlay_frames,
        "vis_overlay_path": "vis_overlay.mp4" if overlay_path else None,
        "vis_overlay_frame_indices": overlay_frame_indices,
        "frames": results_list,
    }
    if extra_fields:
        meta.update(extra_fields)
    metadata_path = Path(sam3d_dir) / "metadata.json"
    temporary = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(meta, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def _compact_pose_tree(sam3d_dir, results, anchor_index):
    """Remove the frame directories of pose-only keyframes once metadata.json holds their poses."""
    for frame_index, result in results.items():
        frame_dir = Path(sam3d_dir) / f"frame_{frame_index:04d}"
        if frame_index != anchor_index and str(result.get("mode", "")).startswith("pose_only"):
            (frame_dir / "pose.json").unlink(missing_ok=True)
            try:
                frame_dir.rmdir()
            except OSError:
                pass
