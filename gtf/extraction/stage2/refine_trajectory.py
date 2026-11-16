#!/usr/bin/env python3
"""Stage-2 trajectory refinement: optimized_mano/optimized.npz -> optimized_stage2/optimized.npz, per segment.

The chain runs in memory: recover the packed per-keyframe states (stage2_npz.recover_states), apply the steps,
rebuild the npz (stage2_npz.rebuild_npz):

  1. hand-rigidity object-rotation bridge (repair_hand_rigidity.py): where the object is occluded and held, its
     rotation steps follow the grasping hand's. Weighted by occlusion x contact (runs shorter than bridge_min_run
     dropped, run boundaries tapered), skipped when the camera rotates fast, and reverted when it adds rotation
     energy instead of removing drift.
  2. learned per-segment translation bias (denoiser_model.py, checkpoints/denoiser/v0.pt; production chain only,
     --no-bias skips it): one constant translation per entity, estimated over overlapping windows at a ~100 ms
     keyframe cadence, reduced to its along-ray (depth) component, shared by the object and every hand in contact
     with it, and capped at bias_max_mm.
  3. temporal filter (temporal_filter.py): hand wrist orientation + cam_t over +-1 keyframe; the object is left as is.
  4. articulation filter (articulation_filter.py): swing-twist clamp + per-joint Karcher smoothing of the fingers.
  5. depth-from-size repair (depth_repair.py).

The object is rebuilt by per-frame delta transport (stage2_npz.rebuild_npz), which keeps the per-keyframe SAM 3D scale
in obj_verts. Multi-window children ({seg}__iwN) are skipped: they share the parent's stage-2 chain.
No ground truth is used; fps is read from video.mp4.

Output: {segment}/optimized_stage2/optimized.npz (same schema as optimized_mano/optimized.npz; export_canonical and
quality_gate read it with opt_subdir="optimized_stage2") and {segment}/optimized_stage2/refine_report.json (gates,
bias magnitudes, residual diagnostics, and a transaction record that binds input and output by SHA-256; written
last, so a complete report marks a complete refinement).

Usage:
    uv run python -m gtf.extraction.stage2.refine_trajectory --segment <name>
    uv run python -m gtf.extraction.stage2.refine_trajectory --all              # every segment with optimized_mano
    uv run python -m gtf.extraction.stage2.refine_trajectory --segment <name> --no-bias --force
"""

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from gtf.paths import REPO_ROOT
from gtf.extraction.stage2.stage2_common import (
    OBJ,
    OBJ_ROT,
    OBJ_TRANS,
    RESULTS_DIR,
    STATE_DIM,
    WINDOW_T,
    center_states,
    compute_origin,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gtf.extraction import chains
from gtf.extraction.stage2.articulation_filter import apply_articulation_filter
from gtf.extraction.stage2.stage2_npz import rebuild_npz, recover_states
from gtf.extraction.stage2.repair_hand_rigidity import occlusion_weight, repair_chain
from gtf.extraction.stage2.temporal_filter import apply_temporal_filter
from gtf.utils.npz import savez_compressed_level
from gtf.utils.cli_settings import Setting, add_settings, apply_settings

DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "denoiser" / "v0.pt"
TRANS_COLS = ((0, slice(6, 9)), (1, slice(105, 108)), (2, slice(204, 207)))  # (valid col, trans slice)
ENTITY_NAMES = ("left", "right", "obj")
REFINE_TRANSACTION_FORMAT = "gtf.refine.v2"


class RefinePreconditionReject(RuntimeError):
    """The bound optimizer output cannot support temporal refinement."""

    def __init__(self, observed_keyframes: int):
        self.observed_keyframes = int(observed_keyframes)
        super().__init__(f"insufficient_temporal_support: {self.observed_keyframes} keyframe(s); at least 3 required")


def _safe_child_name(value: str, *, label: str) -> str:
    """Validate a single literal child name before any filesystem mutation."""

    if not isinstance(value, str) or not value or value in {".", ".."} or "\x00" in value or Path(value).is_absolute() or Path(value).parts != (value,):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Publish JSON last so an interrupted refinement is never resumable as complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def refine_outputs_complete(input_path: Path, output_path: Path, report_path: Path, segment: str) -> bool:
    """Validate the stage-2 NPZ plus its last-published transaction report.

    The transaction binds both files by content (REFINE_TRANSACTION_FORMAT). A report
    without a matching transaction counts as incomplete and is regenerated on resume.
    """
    try:
        if not input_path.is_file() or not output_path.is_file() or not report_path.is_file() or output_path.stat().st_mtime_ns <= input_path.stat().st_mtime_ns:
            return False
        with np.load(output_path, allow_pickle=True) as existing:
            required = {
                "frame_indices",
                "obj_verts",
                "canonical_verts",
                "obj_faces",
            }
            if not required.issubset(existing.files):
                return False
            frame_indices = np.asarray(existing["frame_indices"])
            if frame_indices.ndim != 1 or frame_indices.size == 0:
                return False
            frame_count = int(frame_indices.size)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("segment") != segment or int(report.get("K", -1)) != frame_count:
            return False
        transaction = report.get("transaction")
        if not isinstance(transaction, dict):
            return False
        input_stat = input_path.stat()
        output_stat = output_path.stat()
        return bool(
            transaction.get("format") == REFINE_TRANSACTION_FORMAT
            and transaction.get("input_sha256") == _sha256_file(input_path)
            and transaction.get("output_sha256") == _sha256_file(output_path)
            and transaction.get("input_size_bytes") == input_stat.st_size
            and transaction.get("input_mtime_ns") == input_stat.st_mtime_ns
            and transaction.get("output_size_bytes") == output_stat.st_size
            and transaction.get("frame_count") == frame_count
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        EOFError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return False


def video_fps(seg_dir: Path) -> float:
    """Probe fps from video.mp4 (30 when it cannot be read)."""
    import cv2

    vp = seg_dir / "video.mp4"
    if vp.exists():
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 0:
            return float(fps)
    return 30.0


def camera_rotation_speed(seg_dir: Path, fps: float) -> float | None:
    """Median camera rotation speed (deg/s) from object_pose/camera_poses.npz, for the hand-rigidity camera gate.
    None without camera poses."""
    cp = seg_dir / "object_pose" / "camera_poses.npz"
    if not cp.exists():
        return None
    c2w = np.load(cp, allow_pickle=True)["cam_to_world"]
    if len(c2w) < 2:
        return None
    Rrel = np.einsum("tij,tik->tjk", c2w[:-1, :3, :3], c2w[1:, :3, :3])
    tr = np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return float(np.median(np.degrees(np.arccos(tr)) * fps))


def load_denoiser(ckpt_path: Path, device: torch.device):
    """Load the trained denoiser + its normalization stats (default checkpoints/denoiser/v0.pt)."""
    from gtf.extraction.stage2.denoiser_model import TrajDenoiser

    ck = torch.load(ckpt_path, map_location=device)
    model = TrajDenoiser(d_model=ck["args"]["d_model"], n_layers=ck["args"]["n_layers"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = {k: v.to(device) for k, v in ck["stats"].items()}
    return model, stats


@torch.no_grad()
def bias_deltas(states: torch.Tensor, valid: torch.Tensor, model, stats, device, frame_stride: int = 1) -> dict[str, torch.Tensor]:
    """Per-segment per-entity constant translation deltas: the denoiser runs over
    overlapping windows (stride T/2, last window right-aligned; one window when K <=
    WINDOW_T), per-frame deltas are overlap-averaged, then collapsed to the valid-frame mean.

    frame_stride decimates the keyframes the model sees toward its trained ~100 ms cadence
    (extractions can have 33-67 ms steps); the output is a per-segment constant, so
    decimation only changes the model's view, never which frames get corrected."""
    sub = torch.arange(0, states.shape[0], frame_stride, device=device)
    s_sub, v_sub = states[sub], valid[sub]
    K = len(sub)
    if K <= WINDOW_T:
        starts = [0]
    else:
        starts = list(range(0, K - WINDOW_T, WINDOW_T // 2)) + [K - WINDOW_T]
    sum_d = torch.zeros(K, 9, device=device)
    cnt = torch.zeros(K, 3, device=device)
    for s in starts:
        w = slice(s, min(s + WINDOW_T, K))
        L = w.stop - w.start
        w_states = torch.zeros(1, WINDOW_T, STATE_DIM, device=device)
        w_valid = torch.zeros(1, WINDOW_T, 3, dtype=torch.bool, device=device)
        w_fv = torch.zeros(1, WINDOW_T, dtype=torch.bool, device=device)
        w_states[0, :L] = s_sub[w]
        w_valid[0, :L] = v_sub[w]
        w_fv[0, :L] = True
        origin = compute_origin(w_states, w_fv)
        noisy_n = (center_states(w_states, origin) - stats["mean"]) / stats["std"]
        out = model(noisy_n, w_valid, w_fv)
        states_dn = center_states(out * stats["std"] + stats["mean"], origin, sign=+1.0)[0, :L]
        for i, (col, sl) in enumerate(TRANS_COLS):
            m = v_sub[w][:, col].float().unsqueeze(-1)
            sum_d[w, 3 * i : 3 * i + 3] += (states_dn[:, sl] - s_sub[w][:, sl]) * m
            cnt[w, i] += m[:, 0]
    deltas = {}
    for i, (col, _) in enumerate(TRANS_COLS):
        rows = cnt[:, i] > 0
        per_frame = sum_d[rows, 3 * i : 3 * i + 3] / cnt[rows, i].unsqueeze(-1)
        deltas[ENTITY_NAMES[i]] = per_frame.mean(dim=0) if rows.any() else torch.zeros(3, device=device)
    return deltas


def condition_bridge_weight(w_frame: torch.Tensor, min_run: int = 3, ramp: int = 2) -> torch.Tensor:
    """Condition the hand-rigidity blend weight (per-step semantics follow repair_chain:
    step k bridges when w at k or k+1 is active).

    The raw occlusion x contact weight flickers (isolated 1-step activations); bridging
    those injects single noisy hand increments as object rotation, and every hard 0->1
    transition is a velocity kink. This only ever reduces the bridge (never extends it):
    zero active runs shorter than min_run frames, then taper each surviving run's boundary
    frames (linear ramp inside the run)."""
    w = w_frame.clone()
    K = len(w)
    k = 0
    while k < K:
        if w[k] <= 0:
            k += 1
            continue
        a = k
        while k < K and w[k] > 0:
            k += 1
        L = k - a
        if L < min_run:
            w[a:k] = 0.0
        elif ramp > 0:
            for j in range(L):
                w[a + j] *= min(1.0, (j + 1) / (ramp + 1), (L - j) / (ramp + 1))
    return w


def radial_project(d: torch.Tensor, p_mean: torch.Tensor) -> torch.Tensor:
    """Keep only the along-view-ray (depth) component of a bias delta.

    The correction targets the shared MoGe depth bias; the tangential (image-plane)
    component would fight the silhouette/keypoint alignment the optimizer already enforced
    (xy is image-constrained at any depth, z is the unconstrained channel). Moving along the
    ray through the entity keeps its projection fixed, so the silhouette IoU is preserved by
    construction. p_mean: the entity's mean camera-frame position."""
    u = p_mean / p_mean.norm().clamp(min=1e-6)
    return (d * u).sum() * u


def contact_tie_sides(opt: dict, thresh_m: float, min_frac: float = 0.1, min_valid: int = 3) -> dict[str, float]:
    """Per-side contact fraction over each side's own valid frames → {side: frac} for
    sides meeting the tie criterion.

    Every hand in contact must share the object's delta: a contacting hand with its own
    delta has its contact torn apart by the inter-entity bias difference. Each side is
    judged on its own valid frames, so a barely tracked hand cannot shadow the hand that
    actually grasps."""
    ov = opt["obj_verts"][:, ::10]
    out: dict[str, float] = {}
    for side in ("left", "right"):
        mk, vk = f"hand_{side}_valid", f"hand_{side}_verts"
        if mk not in opt or vk not in opt:
            continue
        rows = np.where(opt[mk].astype(bool))[0]
        if len(rows) < min_valid:
            continue
        hverts = opt[vk][:, ::6]
        d = np.array([np.linalg.norm(hverts[k][:, None] - ov[k][None], axis=-1).min() for k in rows])
        frac = float((d < thresh_m).mean())
        if frac >= min_frac:
            out[side] = frac
    return out


def robust_contact_signal(opt: dict, thresh_m: float, tie_m: float = 0.001):
    """3-frame-smoothed in-contact gate, the grasping side and its per-frame contact distance.

    The grasping side has the lowest median hand-object distance; a near-tie (within tie_m)
    goes to the hand with more valid frames, then to the right hand, so the choice is stable."""
    total = len(opt["frame_indices"])
    object_vertices = opt["obj_verts"][:, ::10]
    candidates = []
    for side in ("left", "right"):
        valid = opt[f"hand_{side}_valid"].astype(bool)
        if valid.sum() < 3:
            continue
        hand_vertices = opt[f"hand_{side}_verts"][:, ::6]
        distance = np.full(total, np.inf)
        for index in np.where(valid)[0]:
            distance[index] = np.linalg.norm(
                hand_vertices[index][:, None] - object_vertices[index][None],
                axis=-1,
            ).min()
        candidates.append(
            (
                float(np.median(distance[np.isfinite(distance)])),
                int(valid.sum()),
                side,
                distance,
            )
        )
    if not candidates:
        return np.zeros(total), None, np.full(total, np.inf)
    minimum = min(row[0] for row in candidates)
    tied = [row for row in candidates if row[0] <= minimum + tie_m]
    _, _, side, distance = max(tied, key=lambda row: (row[1], row[2] == "right"))
    raw = (distance < thresh_m).astype(np.float32)
    smoothed = np.convolve(raw, np.ones(3) / 3, mode="same")
    return (smoothed >= 0.5).astype(np.float32), side, distance


def refine_one_segment(
    seg_name: str,
    *,
    mano,
    device: torch.device,
    model=None,
    stats=None,
    post_smooth: int = 1,
    post_smooth_obj: int = 0,
    art_smooth: int = 1,
    art_twist_mcp_deg: float = 25.0,
    art_twist_pip_deg: float = 15.0,
    art_swing_deg: float = 135.0,
    vis_lo: float = 0.45,
    vis_hi: float = 0.80,
    contact_mm: float = 30.0,
    stability_mm: float = 15.0,
    cam_gate_deg_s: float = 5.0,
    bridge_min_run: int = 3,
    bridge_ramp: int = 2,
    bias_max_mm: float = 250.0,
    force: bool = False,
) -> dict | None:
    """Run the stage-2 chain on one segment. Returns the report dict, or None if skipped.

    Skip logic: the atomic output and last-published transaction report are both valid,
    bound to this input generation, and not force."""
    seg_name = _safe_child_name(seg_name, label="segment identity")

    seg_dir = RESULTS_DIR / seg_name
    if not seg_dir.exists():
        return None
    if not seg_dir.is_dir() or seg_dir.is_symlink():
        raise ValueError(f"segment directory is absent or unsafe: {seg_dir}")
    # multi-window export children ({seg}__iwN) share the parent's chain via symlinks: refining
    # one would write through the optimized.npz symlink into the parent; the parent's refine
    # already covers them (their export re-crops the shared stage-2 chain)
    meta_p = seg_dir / "metadata.json"
    if meta_p.exists():
        try:
            if json.loads(meta_p.read_text()).get("interaction_window") is not None:
                print(f"  SKIP {seg_name}: multi-window child (refined via its parent)")
                return {"segment": seg_name, "skipped": True}
        except (json.JSONDecodeError, OSError):
            pass
    in_dir = seg_dir / "optimized_mano"
    if not in_dir.is_dir() or in_dir.is_symlink():
        return None
    opt_path = in_dir / "optimized.npz"
    out_dir = seg_dir / "optimized_stage2"
    if out_dir.exists() or out_dir.is_symlink():
        if not out_dir.is_dir() or out_dir.is_symlink():
            raise ValueError(f"refine output directory is unsafe: {out_dir}")
    out_path = out_dir / "optimized.npz"
    report_path = out_dir / "refine_report.json"
    if not opt_path.is_file() or opt_path.is_symlink():
        return None
    if out_path.is_symlink() or report_path.is_symlink():
        raise ValueError(f"refine output contains an unsafe symlink: {out_dir}")
    if not force and refine_outputs_complete(opt_path, out_path, report_path, seg_name):
        return {"segment": seg_name, "skipped": True}

    t0 = time.time()
    input_sha256 = _sha256_file(opt_path)
    opt = dict(np.load(opt_path, allow_pickle=True))
    K = len(opt["frame_indices"])
    if K < 3:
        raise RefinePreconditionReject(K)

    # The report is the transaction marker. Withdraw it only after all immutable
    # input/path preconditions pass, so a rejected or unsafe invocation cannot
    # disturb an existing published transaction.
    report_path.unlink(missing_ok=True)

    fps = video_fps(seg_dir)
    ki = np.asarray(opt["frame_indices"], dtype=int)
    step_ms = float(np.median(np.diff(ki)) / fps * 1000)
    report: dict = {"segment": seg_name, "K": K, "fps": round(fps, 3), "keyframe_step_ms": round(step_ms, 1), "warnings": []}
    if not (60 <= step_ms <= 160):
        report["warnings"].append(f"keyframe step {step_ms:.0f} ms outside the denoiser's trained ~100 ms band")

    states, valid, ctx = recover_states(opt, mano, device)
    report["recover_resid_mm"] = {k: round(v, 2) for k, v in ctx["recover_resid_mm"].items()}
    report["obj_rigid_resid_mm"] = round(ctx["obj_rigid_resid_mm"], 2)
    if ctx["obj_rigid_resid_mm"] > 2.0:
        # informational only: the delta rebuild preserves non-rigid structure
        report["warnings"].append(f"object chain not rigid (resid {ctx['obj_rigid_resid_mm']:.2f} mm; per-keyframe SAM3D scale, preserved by the delta rebuild)")

    # ── 1. hand-rigidity object-rotation bridge (occlusion gated, open-loop) ──
    hr: dict = {"applied": False}
    tracks_path = seg_dir / "object_pose" / "tracks.npz"
    if tracks_path.exists():
        cam_speed = camera_rotation_speed(seg_dir, fps)
        gated = cam_speed is None or cam_speed > cam_gate_deg_s
        w_occ = occlusion_weight(seg_dir, ki, vis_lo, vis_hi)
        contact, side, cdist = robust_contact_signal(opt, contact_mm / 1000.0)
        w_frame = condition_bridge_weight(torch.from_numpy(w_occ * contact).float().to(device), min_run=bridge_min_run, ramp=bridge_ramp)
        hr.update(
            {
                "cam_speed_deg_s": round(cam_speed, 2) if cam_speed is not None else None,
                "cam_gated": bool(gated),
                "grasp_side": side,
                "active_frac": round(float((w_frame > 0).float().mean()), 3),
            }
        )
        if side is not None and float(w_frame.max()) > 0 and not gated:
            hc = ctx[f"hand_{side}"]
            R_obj = rotation_6d_to_matrix(states[:, OBJ][:, OBJ_ROT])
            t_obj = states[:, OBJ][:, OBJ_TRANS]
            R_new, t_new = repair_chain(
                R_obj,
                t_obj,
                hc["R"],
                hc["wrist"],
                hc["rows"],
                w_frame,
                anchor_pos=ctx["anchor_pos"],
                contact_dist=torch.from_numpy(cdist).float().to(device),
                stability_mm=stability_mm,
            )

            # do-no-harm step check: the bridge removes occlusion drift and must not add
            # rotation energy. If the bridged chain's max per-step rotation exceeds the raw
            # chain's by more than a drift-rate margin (2x + 3 deg), the carrier hand is too
            # noisy on this segment: revert.
            def _max_step_deg(R):
                rel = R[1:] @ R[:-1].transpose(1, 2)
                tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
                return float(torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).max())

            step_raw, step_new = _max_step_deg(R_obj), _max_step_deg(R_new)
            hr["max_step_deg_raw_new"] = [round(step_raw, 2), round(step_new, 2)]
            if step_new > 2.0 * step_raw + 3.0:
                hr["step_sanity_reverted"] = True
            else:
                changed = float(torch.rad2deg(torch.arccos((((R_new @ R_obj.transpose(1, 2)).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))).max())
                states[:, OBJ.start + OBJ_ROT.start : OBJ.start + OBJ_ROT.stop] = matrix_to_rotation_6d(R_new)
                states[:, OBJ.start + OBJ_TRANS.start : OBJ.start + OBJ_TRANS.stop] = t_new
                hr["applied"] = True
                hr["max_rot_change_deg"] = round(changed, 2)
    else:
        hr["error"] = "object_pose/tracks.npz missing"
    report["handrigid"] = hr

    # ── 2. learned per-segment translation bias (denoiser; production chain only) ──
    bias: dict = {"applied": False}
    if model is not None:
        frame_stride = max(1, round(100.0 / step_ms)) if step_ms > 0 else 1
        bias["frame_stride"] = frame_stride
        tie_fracs = contact_tie_sides(opt, contact_mm / 1000.0)
        M = torch.tensor([-1.0, 1.0, 1.0], device=states.device)

        def to_cam(name, d):
            return d * M if name == "left" else d

        deltas = bias_deltas(states, valid, model, stats, device, frame_stride=frame_stride)
        for i, (col, sl) in enumerate(TRANS_COLS):
            bias[ENTITY_NAMES[i] + "_raw_mm"] = [round(x, 1) for x in (deltas[ENTITY_NAMES[i]] * 1000).tolist()]
        for i, (col, sl) in enumerate(TRANS_COLS):
            name = ENTITY_NAMES[i]
            if valid[:, col].any():
                p_mean = states[valid[:, col], sl].mean(dim=0)
                deltas[name] = radial_project(deltas[name], p_mean)
        # contact tie: during a grasp the true bias is the shared depth factor; per-entity
        # differentials are model noise that would tear apart the optimizer-enforced contact.
        # One shared, validity-weighted delta for all contacting hands + the object preserves their
        # relative geometry; the consensus is formed in camera space (the left channel
        # lives in mirrored right-model coords, the M-conjugation convention).
        if tie_fracs:
            group = list(tie_fracs) + ["obj"]
            K_total = len(opt["frame_indices"])
            w = torch.tensor([float(opt[f"hand_{s}_valid"].sum()) for s in tie_fracs] + [float(K_total)], device=deltas["obj"].device)
            tied_cam = (torch.stack([to_cam(n, deltas[n]) for n in group]) * (w / w.sum()).unsqueeze(-1)).sum(0)
            for name in group:
                deltas[name] = to_cam(name, tied_cam).clone()
            bias["contact_tied"] = group
            bias["contact_frac"] = {s: round(f, 3) for s, f in tie_fracs.items()}
        for i, (col, sl) in enumerate(TRANS_COLS):
            name = ENTITY_NAMES[i]
            d = deltas[name]
            # trust the corrector only inside the envelope it was trained on: corrections
            # beyond bias_max_mm are extrapolation; scale them down, direction-preserving
            n_mm = float(d.norm()) * 1000
            if n_mm > bias_max_mm:
                d = d * (bias_max_mm / n_mm)
                bias.setdefault("capped", []).append(name)
            rows = valid[:, col]
            states[rows, sl] = states[rows, sl] + d.unsqueeze(0)
            bias[name + "_mm"] = [round(x, 1) for x in (d * 1000).tolist()]
        bias["applied"] = True
    report["bias"] = bias

    # ── 3. temporal smoothing of the rigid channels (hands radius 1, object 0) ──
    if post_smooth > 0 or post_smooth_obj > 0:
        states = apply_temporal_filter(states, valid, radius_hands=post_smooth, radius_obj=post_smooth_obj)
    report["post_smooth"] = {"hands": post_smooth, "obj": post_smooth_obj}

    # delta object rebuild: preserves the input's per-keyframe (non-rigid) SAM3D scale
    out = rebuild_npz(opt, states, valid, ctx, mano, device)

    # ── 4. articulation: swing-twist clamp + Karcher smoothing of the finger joints ──
    # (temporal_filter above only touches the rigid wrist channels)
    report["articulation"] = apply_articulation_filter(
        out,
        mano,
        device,
        smooth_radius=art_smooth,
        twist_mcp_deg=art_twist_mcp_deg,
        twist_pip_deg=art_twist_pip_deg,
        swing_deg=art_swing_deg,
    )

    # ── 5. depth-from-size repair: scale-coupled object depth wander ──
    # Rigidify the per-keyframe object scale + flatten the mask/sil apparent-size ratio
    # trend (the one depth signal the video constrains over time); in-contact hands are
    # carried via per-run anchor vertices. Scope-gated (fires only on chains with the
    # scale artifact, size_fold >= 1.35) and reduce-only, so a no-op on rigid chains.
    # Must run before the export's object refiner (it confounds the ratio<->depth relation).
    from gtf.extraction.stage2.depth_repair import repair_stage2_inplace

    report["depth_repair"] = repair_stage2_inplace(out, seg_dir)

    out_dir.mkdir(exist_ok=True)
    savez_compressed_level(
        out_path,
        compresslevel=6,
        allow_pickle=True,
        **out,
    )
    if _sha256_file(opt_path) != input_sha256:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("stage-1 optimized input changed during refinement; refusing to publish a mixed-generation stage-2 trajectory")
    report["elapsed_s"] = round(time.time() - t0, 1)
    input_stat = opt_path.stat()
    output_stat = out_path.stat()
    output_sha256 = _sha256_file(out_path)
    report["transaction"] = {
        "format": REFINE_TRANSACTION_FORMAT,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "input_size_bytes": input_stat.st_size,
        "input_mtime_ns": input_stat.st_mtime_ns,
        "output_size_bytes": output_stat.st_size,
        "frame_count": K,
    }
    _write_json_atomic(report_path, report)
    return report


# Settings without a flag of their own; change one with --set KEY=VALUE (gtf/utils/cli_settings.py).
SETTINGS = {
    "post_smooth": Setting(1, int, "temporal-filter radius for hand rigid channels (1 matches ground-truth jerk)"),
    "post_smooth_obj": Setting(0, int, "temporal-filter radius for the object (0: the optimized object chain is already smoother than ground truth)"),
    "art_smooth": Setting(1, int, "per-joint Karcher articulation-smoothing radius (1 matches ground-truth fingertip jitter; 0 disables)"),
    "art_twist_mcp_deg": Setting(25.0, float, "twist limit of the MCP joints (deg)"),
    "art_twist_pip_deg": Setting(15.0, float, "twist limit of the PIP / DIP joints (deg)"),
    "art_swing_deg": Setting(135.0, float, "swing limit of every finger joint (deg)"),
    "vis_lo": Setting(0.45, float, "bridge occlusion weight is 1 at or below this normalized track visibility"),
    "vis_hi": Setting(0.8, float, "bridge occlusion weight is 0 at or above this normalized track visibility"),
    "contact_mm": Setting(30.0, float, "hand-object distance counted as contact (bridge gate and bias contact tie)"),
    "stability_mm": Setting(15.0, float, "drop bridge runs whose contact distance varies more than this (std)"),
    "cam_gate_deg_s": Setting(5.0, float, "skip the bridge when the median camera rotation speed exceeds this (deg/s)"),
    "bridge_min_run": Setting(3, int, "zero hand-rigidity bridge runs shorter than this (flicker injects single noisy hand increments)"),
    "bridge_ramp": Setting(2, int, "taper bridge-run boundaries over this many steps (hard 0->1 transitions are velocity kinks)"),
    "bias_max_mm": Setting(250.0, float, "cap per-entity bias correction at the denoiser's trained envelope (direction-preserving)"),
}


def main():
    ap = argparse.ArgumentParser(
        prog="python -m gtf.extraction.stage2.refine_trajectory",
        description="Stage-2 trajectory refinement (hand-rigidity bridge, bias, filters, depth repair) on optimizer outputs",
    )
    ap.add_argument("--segment", action="append", default=None, help="segment name(s) under the results directory (repeatable)")
    ap.add_argument("--all", action="store_true", help="every segment of the results directory with optimized_mano/optimized.npz")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="denoiser checkpoint (default checkpoints/denoiser/v0.pt)")
    ap.add_argument("--no-bias", action="store_true", help="skip the learned bias corrector (data-independent repairs only)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    chains.add_argument(ap)
    add_settings(ap, SETTINGS)
    args = ap.parse_args()
    apply_settings(ap, args, SETTINGS)
    chain = chains.get(args.chain)

    if args.all:
        segs = sorted(d.parent.parent.name for d in RESULTS_DIR.glob("*/optimized_mano/optimized.npz"))
    else:
        segs = args.segment or []
    if not segs:
        ap.error("give --segment or --all")

    from gtf.model.repr.mano import MANOHelper
    from gtf.paths import MANO_DIR

    device = torch.device(args.device)
    mano = MANOHelper(MANO_DIR).to(device)
    model = stats = None
    if chain.refine_bias and not args.no_bias:  # the upgraded chain runs the filters only
        model, stats = load_denoiser(Path(args.ckpt), device)
        print(f"denoiser: {args.ckpt}")

    n_done = n_skip = n_reject = 0
    for seg in segs:
        try:
            chains.claim(RESULTS_DIR / seg, chain)
            r = refine_one_segment(
                seg,
                mano=mano,
                device=device,
                model=model,
                stats=stats,
                post_smooth=args.post_smooth,
                post_smooth_obj=args.post_smooth_obj,
                art_smooth=args.art_smooth,
                art_twist_mcp_deg=args.art_twist_mcp_deg,
                art_twist_pip_deg=args.art_twist_pip_deg,
                art_swing_deg=args.art_swing_deg,
                vis_lo=args.vis_lo,
                vis_hi=args.vis_hi,
                contact_mm=args.contact_mm,
                stability_mm=args.stability_mm,
                cam_gate_deg_s=args.cam_gate_deg_s,
                bridge_min_run=args.bridge_min_run,
                bridge_ramp=args.bridge_ramp,
                bias_max_mm=args.bias_max_mm,
                force=args.force,
            )
        except RefinePreconditionReject as exc:
            from gtf.utils.action100m_quality_precondition import (
                publish_quality_precondition_reject,
            )

            record = publish_quality_precondition_reject(
                RESULTS_DIR / seg,
                expected_segment=seg,
            )
            n_reject += 1
            print(f"  PRECONDITION REJECT {seg}: {record['reason']} ({record['observed_keyframes']}/{record['minimum_required_keyframes']} keyframes); {exc}")
            continue
        if r is None:
            continue
        if r.get("skipped"):
            n_skip += 1
            continue
        n_done += 1
        hr, bi = r["handrigid"], r["bias"]
        hr_s = f"hr={'ON' if hr.get('applied') else 'off'}" + (
            f"(cam{hr['cam_speed_deg_s']}d/s{' GATED' if hr.get('cam_gated') else ''},act{hr.get('active_frac', 0):.0%})" if "cam_speed_deg_s" in hr else ""
        )
        bi_s = "bias=" + (",".join(f"{e}:{np.linalg.norm(bi[e + '_mm']):.0f}mm" for e in ENTITY_NAMES if e + "_mm" in bi) if bi.get("applied") else "off")
        warn = (" WARN: " + "; ".join(r["warnings"])) if r["warnings"] else ""
        print(f"{seg:<46} K={r['K']:<4} step={r['keyframe_step_ms']:.0f}ms {hr_s} {bi_s} ({r['elapsed_s']}s){warn}")
    print(f"\n[refine] {n_done} done, {n_skip} skipped (up to date), {n_reject} precondition rejects")


if __name__ == "__main__":
    main()
