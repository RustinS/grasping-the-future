"""Shared helpers of the extraction pipeline: paths, logging, segment I/O, prompts, depth alignment and rendering."""

import json
import os
import pickle
from datetime import datetime
from functools import lru_cache

import numpy as np
from rich.console import Console

from gtf.paths import REPO_ROOT, mano_model_path

from .action_validation import (
    ActionValidationResolution,
    reject_action_validation as _reject_action_validation,
    resolve_action_validation as _resolve_action_validation,
)

# ====================================================================
# Constants
# ====================================================================

ROOT = str(REPO_ROOT)
# Per-segment results; ACTION100M_RESULTS_DIR (absolute) redirects every stage.
_RESULTS_OVERRIDE = os.environ.get("ACTION100M_RESULTS_DIR")
if _RESULTS_OVERRIDE is not None and not os.path.isabs(_RESULTS_OVERRIDE):
    raise RuntimeError("ACTION100M_RESULTS_DIR must be an absolute path")
RESULTS_DIR = os.path.realpath(_RESULTS_OVERRIDE or os.path.join(ROOT, "results"))
# Cut Action100M clips and their manifest.json, written by the fetch step.
SEGMENTS_DIR = os.path.join(ROOT, "segments")

PANEL_W, PANEL_H = 640, 360

HAND_COLOR = (1.0, 1.0, 0.9)
HAND_COLOR_LEFT = (0.9, 1.0, 1.0)

GLB_TO_RAW = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

console = Console()

_MANO_FACES_CACHE = {}
_RENDERER_CACHE = {}  # keyed by (width, height), reused across render calls


# ====================================================================
# Logging
# ====================================================================


def rprint(*args, no_extra: bool = False, **kwargs) -> None:
    """Rich print with a '[YYYY-MM-DD HH:MM:SS] ' prefix (no prefix with ``no_extra``)."""
    prefix = "" if no_extra else f"[[bold green]{datetime.now():%Y-%m-%d %H:%M:%S}[/bold green]] "

    if args and all(isinstance(a, (str, bytes)) for a in args):
        msg = " ".join(a.decode() if isinstance(a, bytes) else a for a in args)
        console.print(f"{prefix}{msg}", **kwargs)
    else:
        console.print(prefix)
        for obj in args:
            console.print(obj, **kwargs)


# ====================================================================
# Data loading
# ====================================================================


def load_manifest():
    """Load SEGMENTS_DIR/manifest.json and return the 'succeeded' entries list."""
    with open(os.path.join(SEGMENTS_DIR, "manifest.json")) as f:
        entries = json.load(f)["succeeded"]
    rprint(f"Loaded {len(entries)} succeeded segments from manifest")
    return entries


def resolve_action_validation(name, obj=None) -> ActionValidationResolution:
    """Return the shared fail-closed action resolution for ``name``."""

    return _resolve_action_validation(
        os.path.join(RESULTS_DIR, name), requested_object=obj
    )


def is_action_valid(name, obj=None):
    """Check validation through the shared fail-closed resolver."""

    return resolve_action_validation(name, obj).valid


def get_segment_object(name):
    """Return the segment's resolved object name (the metadata noun before validation has run)."""

    return resolve_action_validation(name).object_name


def reject_action(name, reason, obj=None) -> tuple[str, ...]:
    """Persist a rejection through the shared action-row transaction."""

    return _reject_action_validation(
        os.path.join(RESULTS_DIR, name), reason, requested_object=obj
    )


# ──────────────────────────────────────────────────────────────────────────────
# Object-agnostic front end (the pipeline runs without a fixed --object)
#
# Validation asks whether any object is manipulated; the discovered noun is a soft
# label used for pointing and the canonical metadata. Pointing names the noun when it
# is specific enough to ground a point, and uses an agnostic prompt otherwise.
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_PROMPT_AGNOSTIC = "Is an object being manipulated by hand in this video? Answer yes or no and explain briefly."
FRAME_PROMPT_AGNOSTIC = (
    "Is an object being manipulated by hand in this image? Answer only yes or no."
)

POINTING_PROMPT_NAMED = "point to the {object} in hand"
POINTING_PROMPT_FALLBACK = "point to the object held in the hand"

# Nouns too vague to ground a Molmo2 point -> pointing uses the agnostic fallback.
GENERIC_OBJECT_NAMES = frozenset(
    {
        "object",
        "objects",
        "item",
        "items",
        "thing",
        "things",
        "stuff",
        "something",
        "tool",
        "tools",
        "device",
        "devices",
        "equipment",
        "utensil",
    }
)


def is_groundable_object_name(obj):
    """True if *obj* is a specific noun usable in a Molmo2 pointing prompt.

    False for None/empty, the GENERIC_OBJECT_NAMES set, or an absurdly long string
    (a malformed discovery response). Multi-word specifics ("spray can") are groundable.
    """
    if not obj:
        return False
    name = obj.strip().lower()
    if not name or name in GENERIC_OBJECT_NAMES:
        return False
    if len(name.split()) > 4:
        return False
    return True


def build_pointing_prompt(obj):
    """Named pointing prompt when *obj* is groundable, agnostic fallback otherwise."""
    if is_groundable_object_name(obj):
        return POINTING_PROMPT_NAMED.format(object=obj)
    return POINTING_PROMPT_FALLBACK


def load_mano_faces():
    """Load MANO hand faces (right and left) with watertight closure. Cached after first call."""
    if "right" in _MANO_FACES_CACHE:
        return _MANO_FACES_CACHE["right"], _MANO_FACES_CACHE["left"]

    with open(mano_model_path("right"), "rb") as f:
        mano = pickle.load(f, encoding="latin1")
    faces = mano["f"].astype(np.int64)

    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ],
        dtype=np.int64,
    )
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    _MANO_FACES_CACHE["right"] = faces_right
    _MANO_FACES_CACHE["left"] = faces_left
    return faces_right, faces_left


def load_hand_data(segment_dir):
    """Load the hand stage's output (WiLoR, in `hands/`): verts, joints, intrinsics, validity per hand."""
    import torch

    packed_dir = os.path.join(segment_dir, "hands")

    det_valid = {}
    det_pkl_path = os.path.join(segment_dir, "hands", "det.pkl")
    if os.path.exists(det_pkl_path):
        with open(det_pkl_path, "rb") as f:
            det_data = pickle.load(f)
        for entry in det_data:
            seq, valid_arr, frame_idx_arr = (
                entry.get("seq", ""),
                entry.get("valid"),
                entry.get("frame_idx"),
            )
            if valid_arr is None or frame_idx_arr is None:
                continue
            for side in ("left", "right"):
                if seq.endswith(f"_{side}"):
                    det_valid[side] = {
                        int(frame_idx_arr[i]) for i, v in enumerate(valid_arr) if v
                    }

    result = {}
    for side in ("left", "right"):
        pt_path = os.path.join(packed_dir, f"{side}.pt")
        npz_path = os.path.join(packed_dir, f"{side}.npz")
        if not os.path.exists(pt_path) or not os.path.exists(npz_path):
            continue

        pt = torch.load(pt_path, map_location="cpu")
        frame_names = np.load(npz_path, allow_pickle=False)["frame_names"]

        # Keep first occurrence of each frame index
        frame_to_idx = {int(fn): i for i, fn in reversed(list(enumerate(frame_names)))}

        result[side] = {
            "verts": pt["cHands"].numpy(),
            "joints": pt["cJoints"].numpy(),
            "intr": pt["intr"].numpy(),
            "img_size": pt["img_size"].numpy(),
            "frame_to_idx": frame_to_idx,
            "valid_frames": det_valid.get(side, set(frame_to_idx.keys())),
        }
    return result


def _moge_depth_path(segment_dir, frame_index):
    return os.path.join(
        segment_dir, "object_pose", "depth", f"{frame_index:05d}.npz"
    )


def load_moge_depth(segment_dir, frame_index):
    """Load MoGe depth map and intrinsics for a frame."""
    depth_path = _moge_depth_path(segment_dir, frame_index)
    if not os.path.exists(depth_path):
        return None, None
    with np.load(depth_path, allow_pickle=False) as data:
        return data["depth"], data["intrinsics_px"]


@lru_cache(maxsize=8192)
def _load_moge_intrinsics_generation(
    depth_path, device, inode, size, mtime_ns, ctime_ns
):
    """Load only the small NPZ member, bound to one exact file generation."""

    del device, inode, size, mtime_ns, ctime_ns  # fields intentionally bind the cache key
    with np.load(depth_path, allow_pickle=False) as data:
        return np.array(data["intrinsics_px"], copy=True)


def load_moge_intrinsics(segment_dir, frame_index):
    """Load only MoGe's 3x3 intrinsics without inflating the depth member.

    NPZ members are compressed independently, so reading only the intrinsics avoids
    decompressing the depth map. The stat-bound cache invalidates on atomic replacement
    or in-place mutation and returns a private 3x3 copy so callers cannot mutate it.
    """

    depth_path = _moge_depth_path(segment_dir, frame_index)
    try:
        stat = os.stat(depth_path)
    except FileNotFoundError:
        return None
    intrinsics = _load_moge_intrinsics_generation(
        os.path.realpath(depth_path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    return intrinsics.copy()


def load_sam3d_metadata(segment_dir, subdir="sam3d"):
    """Load SAM3D metadata.json (subdir="sam3d_c" = the temporal dual-init variant).
    Returns dict or None."""
    meta_path = os.path.join(segment_dir, subdir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def load_object_mesh(segment_dir, frame_index, max_faces=5000, subdir="sam3d"):
    """Load SAM3D GLB mesh for a frame, decimating if over max_faces. Returns trimesh or None."""
    import trimesh

    glb_path = os.path.join(segment_dir, subdir, f"frame_{frame_index:04d}", "mesh.glb")
    if not os.path.exists(glb_path):
        return None
    mesh = trimesh.load(glb_path, force="mesh")
    if max_faces and len(mesh.faces) > max_faces:
        target_reduction = 1.0 - max_faces / len(mesh.faces)
        vc = (
            mesh.visual.vertex_colors
            if hasattr(mesh.visual, "vertex_colors")
            and mesh.visual.vertex_colors is not None
            else None
        )
        decimated = mesh.simplify_quadric_decimation(target_reduction)
        # Decimation drops the vertex colors and trimesh fills in a constant default, so
        # colors are re-interpolated from the original mesh when it had real color variance.
        dvc = (
            decimated.visual.vertex_colors
            if hasattr(decimated.visual, "vertex_colors")
            and decimated.visual.vertex_colors is not None
            else None
        )
        if (
            vc is not None
            and np.asarray(vc)[:, :3].std() > 1e-6
            and (dvc is None or np.asarray(dvc)[:, :3].std() < 1e-6)
        ):
            from scipy.spatial import cKDTree

            tree = cKDTree(mesh.vertices)
            _, idx = tree.query(decimated.vertices)
            decimated.visual.vertex_colors = np.asarray(vc)[idx]
        mesh = decimated
    return mesh


# ====================================================================
# Coordinate transforms
# ====================================================================


def quaternion_to_matrix(q_wxyz):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def opencv_to_opengl(verts_cv):
    """Convert OpenCV camera-space verts to OpenGL (flip y, z)."""
    return verts_cv * np.array([1, -1, -1], dtype=np.float32)


# ====================================================================
# Calibration (hand-stage camera ↔ MoGe depth alignment)
# ====================================================================


def compute_scale_z(joints_21x3, depth_map, K_h, img_size_h, K_m):
    """Compute the z-scale between the hand-stage camera and MoGe depth from joint projections. Returns float or None."""
    H_m, W_m = depth_map.shape
    W_h, H_h = int(img_size_h[0]), int(img_size_h[1])

    fx_hn = K_h[0, 0] * W_m / W_h
    fy_hn = K_h[1, 1] * H_m / H_h
    cx_hn = K_h[0, 2] * W_m / W_h
    cy_hn = K_h[1, 2] * H_m / H_h

    joints = joints_21x3.astype(np.float64)
    z_j = joints[:, 2]
    valid_z = z_j > 0
    u = np.zeros(21)
    v = np.zeros(21)
    u[valid_z] = fx_hn * joints[valid_z, 0] / z_j[valid_z] + cx_hn
    v[valid_z] = fy_hn * joints[valid_z, 1] / z_j[valid_z] + cy_hn

    u_int = np.round(u).astype(int)
    v_int = np.round(v).astype(int)
    in_bounds = valid_z & (u_int >= 0) & (u_int < W_m) & (v_int >= 0) & (v_int < H_m)

    moge_z = np.zeros(21)
    moge_z[in_bounds] = depth_map[v_int[in_bounds], u_int[in_bounds]]

    valid = in_bounds & (moge_z > 0) & (z_j > 0)
    if valid.sum() < 5:
        return None

    ratios = moge_z[valid] / z_j[valid]

    # When the hand extends past the object, some joints project onto distant
    # background surfaces producing a bimodal ratio distribution.  Detect this
    # by finding the largest gap in the sorted ratios and keeping the lower
    # (near-camera) cluster which corresponds to joints on/near the object.
    sorted_r = np.sort(ratios)
    gaps = np.diff(sorted_r)
    max_gap_idx = int(np.argmax(gaps))
    max_gap = gaps[max_gap_idx]
    overall_range = sorted_r[-1] - sorted_r[0]
    if (
        overall_range > 0
        and max_gap / overall_range > 0.3
        and max_gap_idx >= 2
        and (len(sorted_r) - max_gap_idx - 1) >= 2
    ):
        ratios = sorted_r[: max_gap_idx + 1]

    if len(ratios) < 3:
        return None

    s = float(np.median(ratios))
    return s if 0.25 <= s <= 4.0 else None


def smooth_scales(
    raw_scales,
    max_deviation=1.5,
    local_interp_tol=0.12,
    local_jump_factor=4.0,
    smooth_passes=2,
):
    """Smooth per-frame scale factors.

    The raw MoGe/hand depth scale should vary slowly over a short segment.
    We first reject global outliers, then interpolate gaps, then suppress
    short-lived local spikes that still cause hand-depth popping.
    """
    frame_indices = sorted(raw_scales.keys())
    if not frame_indices:
        return raw_scales

    valid_scales = [s for fi in frame_indices if (s := raw_scales[fi]) is not None]
    if not valid_scales:
        return raw_scales

    segment_median = float(np.median(valid_scales))
    lo = segment_median / max_deviation
    hi = segment_median * max_deviation

    clean = np.array(
        [
            float(s) if (s := raw_scales[fi]) is not None and lo <= s <= hi else np.nan
            for fi in frame_indices
        ],
        dtype=np.float64,
    )
    fi_arr = np.asarray(frame_indices, dtype=np.float64)
    good = np.isfinite(clean)

    if not good.any():
        clean[:] = segment_median
    elif good.sum() == 1:
        clean[:] = clean[good][0]
    else:
        clean = np.interp(fi_arr, fi_arr[good], clean[good])

    for _ in range(smooth_passes):
        updated = clean.copy()
        if len(clean) > 2:
            step_med = float(np.median(np.abs(np.diff(clean))))
            spike_tol = max(
                local_interp_tol * segment_median,
                local_jump_factor * step_med,
                0.02 * segment_median,
            )
            for i in range(1, len(clean) - 1):
                alpha = (fi_arr[i] - fi_arr[i - 1]) / max(
                    fi_arr[i + 1] - fi_arr[i - 1], 1.0
                )
                pred = clean[i - 1] * (1 - alpha) + clean[i + 1] * alpha
                if abs(clean[i] - pred) > spike_tol:
                    updated[i] = pred
            updated[1:-1] = (
                0.25 * updated[:-2] + 0.5 * updated[1:-1] + 0.25 * updated[2:]
            )
        clean = updated

    return {fi: float(s) for fi, s in zip(frame_indices, clean)}


# ====================================================================
# Rendering primitives (lazy-import pyrender/trimesh/cv2)
# ====================================================================


def letterbox(image, target_w, target_h):
    """Resize image into target box preserving aspect ratio, pad with black."""
    import cv2

    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized[:, :, :3]
    return canvas


def create_raymond_lights():
    """Create 3-point directional lighting (Raymond setup)."""
    import pyrender

    sin_t, cos_t = np.sin(np.pi / 6), np.cos(np.pi / 6)
    phis = np.pi * np.array([0.0, 2 / 3, 4 / 3])
    nodes = []
    for phi in phis:
        z = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
        z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            )
        )
    return nodes


def render_scene_pyrender(meshes, fx, fy, cx, cy, W, H, cam_pose=None):
    """Render meshes with pyrender. Returns (color_rgba_float, depth). meshes: list of (verts, faces, color)."""
    import pyrender
    import trimesh

    if cam_pose is None:
        cam_pose = np.eye(4)
    key = (int(W), int(H))
    if key not in _RENDERER_CACHE:
        _RENDERER_CACHE[key] = pyrender.OffscreenRenderer(
            viewport_width=key[0], viewport_height=key[1], point_size=1.0
        )
    renderer = _RENDERER_CACHE[key]
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0.0], ambient_light=(0.3, 0.3, 0.3))
    for i, (verts, faces, color) in enumerate(meshes):
        if len(verts) == 0:
            continue
        color = np.asarray(color, dtype=np.float32)
        vc = np.tile(color, (len(verts), 1)) if color.ndim == 1 else color
        if vc.shape[1] == 3:
            vc = np.c_[vc, np.ones(len(vc), dtype=np.float32)]
        tm = trimesh.Trimesh(
            vertices=verts, faces=faces, vertex_colors=(vc * 255).astype(np.uint8)
        )
        scene.add(pyrender.Mesh.from_trimesh(tm), f"mesh_{i}")
    scene.add(
        pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=1e12),
        pose=cam_pose,
    )
    for node in create_raymond_lights():
        scene.add_node(node)
    color_img, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color_img.astype(np.float32) / 255.0, depth


def alpha_composite(fg_rgba, bg_rgb):
    """Alpha-composite RGBA foreground over RGB background (both float 0-1)."""
    alpha = fg_rgba[:, :, 3:4]
    return fg_rgba[:, :, :3] * alpha + bg_rgb * (1 - alpha)


def render_overlay(
    frame_rgb,
    hand_meshes_gl,
    obj_mesh_gl,
    fx_m,
    fy_m,
    cx_m,
    cy_m,
    W,
    H,
):
    """Render hand+object meshes overlaid on a video frame. Returns uint8 RGB image."""
    bg = frame_rgb.astype(np.float32) / 255.0
    meshes = list(hand_meshes_gl)
    if obj_mesh_gl is not None:
        meshes.append(obj_mesh_gl)

    if meshes:
        rgba, _ = render_scene_pyrender(meshes, fx_m, fy_m, cx_m, cy_m, W, H)
        bg = alpha_composite(rgba, bg)

    return (bg * 255).clip(0, 255).astype(np.uint8)


def render_3d_view(meshes_gl, cam_pose, render_w, render_h):
    """Render 3D view of meshes from a given camera pose. Returns uint8 RGB image."""
    empty = np.zeros((render_h, render_w, 3), dtype=np.uint8)
    if not meshes_gl:
        return empty

    all_verts = np.concatenate([m[0] for m in meshes_gl if len(m[0]) > 0], axis=0)
    if len(all_verts) == 0:
        return empty

    verts_cam = (all_verts - cam_pose[:3, 3]) @ cam_pose[:3, :3]
    z_vals = -verts_cam[:, 2]
    z_vals = z_vals[z_vals > 0.01]
    if len(z_vals) == 0:
        return empty

    z_center = (z_vals.min() + z_vals.max()) / 2
    extent = np.abs(verts_cam[:, :2]).max() * 1.3 + 0.01
    focal = max(z_center * min(render_w, render_h) / (2 * extent), 50.0)
    cx, cy = render_w / 2.0, render_h / 2.0
    rgba, _ = render_scene_pyrender(
        meshes_gl, focal, focal, cx, cy, render_w, render_h, cam_pose
    )
    return (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
