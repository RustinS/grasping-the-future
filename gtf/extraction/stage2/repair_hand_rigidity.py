#!/usr/bin/env python3
"""Hand-rigidity object-chain repair: the stage-2 object-rotation bridge.

Under a rigid grasp T_obj = T_hand * T_grasp, so during occluded in-contact intervals
(where the object pose is visually unobservable and BA rotation drift accumulates)
the object's per-step rotation equals the well-tracked grasping hand's. Extraction-side
signals only, no ground truth:

  occlusion  w_occ(k): per-segment-normalized CoTracker track-visibility fraction
             (object_pose/tracks.npz `visibility`)
  contact    c(k): min hand-object vertex distance < threshold, 3-frame smoothed
  blending   rotation only, open-loop increment composition from the SAM3D anchor (see
             repair_chain for why); hand increments tangent-space smoothed; translation
             never touched
  gates      grasp stability (contact-distance std <= 15 mm/run) + the caller's camera-
             motion gate (refine_trajectory.py)

Called by gtf/extraction/stage2/refine_trajectory.py (stage-2 step 1).
"""

from pathlib import Path

import numpy as np
import torch

from gtf.extraction.stage2.temporal_filter import so3_log
from gtf.model.repr.rotation import axis_angle_to_matrix


def slerp_increment(dR_a: torch.Tensor, dR_b: torch.Tensor, w: float) -> torch.Tensor:
    """Geodesic interpolation from increment dR_a toward dR_b by w ∈ [0,1]."""
    delta = dR_b @ dR_a.transpose(-1, -2)
    return axis_angle_to_matrix(w * so3_log(delta)) @ dR_a


def occlusion_weight(seg_dir: Path, ki: np.ndarray, vis_lo: float, vis_hi: float) -> np.ndarray:
    """Per-keyframe occlusion weight ∈ [0,1] from the CoTracker visibility fraction,
    normalized by the segment's own p90 (tracks also die for non-occlusion reasons)."""
    tr = np.load(seg_dir / "object_pose" / "tracks.npz", allow_pickle=True)
    visfrac = tr["visibility"].mean(axis=1)
    kk = np.minimum(ki, len(visfrac) - 1)
    norm = visfrac / max(np.percentile(visfrac, 90), 1e-6)
    return np.clip((vis_hi - norm[kk]) / max(vis_hi - vis_lo, 1e-6), 0.0, 1.0)


def repair_chain(
    R_obj: torch.Tensor,
    t_obj: torch.Tensor,
    R_hand: torch.Tensor,
    wrist: torch.Tensor,
    hand_rows: torch.Tensor,
    w_frame: torch.Tensor,
    anchor_pos: int = 0,
    contact_dist: torch.Tensor | None = None,
    stability_mm: float = 15.0,
    smooth_inc: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Open-loop rotation bridging composed globally from the anchor keyframe; translation
    untouched.

    Channel asymmetry: the extracted translation is absolutely re-anchored per keyframe
    (SAM3D centroids), so hand transport would only add wrist noise to the centroid. The
    extracted rotation chain is relative (BA-composed, never absolutely re-anchored), so
    (a) occlusion drift accumulates (the channel the bridge fixes) and (b) closing a
    bridged interval back to the extraction's exit pose would drag it onto the very drift it
    just avoided. Hence open-loop for rotation: blended increments composed from the SAM3D
    anchor, with the correction carried downstream.

    Hand increments are tangent-space smoothed over ±smooth_inc steps: the recovered hand
    orient carries white per-frame noise that costs per-step error; the bridge's value is
    bias-freeness, which smoothing preserves.

    Grasp-stability gate: a run whose contact distance wobbles (std > stability_mm) is a
    re-grip or a badly tracked hand; rigidity does not hold, so that run's weights go to 0."""
    K = len(R_obj)
    has_hand = torch.zeros(K, dtype=torch.bool, device=R_obj.device)
    has_hand[hand_rows] = True
    Rh = torch.zeros(K, 3, 3, device=R_obj.device)
    Rh[hand_rows] = R_hand
    R_new = R_obj.clone()

    w_step = torch.zeros(K - 1, device=R_obj.device)
    for k in range(K - 1):
        if has_hand[k] and has_hand[k + 1]:
            w_step[k] = max(float(w_frame[k]), float(w_frame[k + 1]))

    # grasp-stability gate per contiguous run
    if contact_dist is not None:
        k = 0
        while k < K - 1:
            if w_step[k] <= 0:
                k += 1
                continue
            a = k
            while k < K - 1 and w_step[k] > 0:
                k += 1
            cd = contact_dist[a : k + 1]
            cd = cd[torch.isfinite(cd)]
            if len(cd) >= 2 and float(cd.std()) * 1000 > stability_mm:
                w_step[a:k] = 0.0
    if float(w_step.max()) <= 0:
        return R_new, t_obj.clone()

    # smoothed hand increments (tangent-space moving average over valid neighbors)
    inc_ok = torch.zeros(K - 1, dtype=torch.bool, device=R_obj.device)
    inc_aa = torch.zeros(K - 1, 3, device=R_obj.device)
    for k in range(K - 1):
        if has_hand[k] and has_hand[k + 1]:
            inc_ok[k] = True
            inc_aa[k] = so3_log(Rh[k + 1] @ Rh[k].T)
    inc_sm = inc_aa.clone()
    for k in range(K - 1):
        if not inc_ok[k]:
            continue
        lo, hi = max(0, k - smooth_inc), min(K - 1, k + smooth_inc + 1)
        sel = inc_ok[lo:hi]
        inc_sm[k] = inc_aa[lo:hi][sel].mean(dim=0)

    # global composition from the anchor, both directions
    def blended(k):  # step k → k+1
        dR_ext = R_obj[k + 1] @ R_obj[k].T
        w = float(w_step[k])
        if w <= 0:
            return dR_ext
        return slerp_increment(dR_ext, axis_angle_to_matrix(inc_sm[k]), w)

    for k in range(anchor_pos, K - 1):
        R_new[k + 1] = blended(k) @ R_new[k]
    for k in range(anchor_pos - 1, -1, -1):
        R_new[k] = blended(k).T @ R_new[k + 1]
    return R_new, t_obj.clone()
