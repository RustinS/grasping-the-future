#!/usr/bin/env python3
"""Sign-robust hand-object penetration + contact metrics.

The sign comes from the generalized (fast) winding number, which stays correct on the
non-watertight / self-intersecting meshes SAM3D produces, inside cavities and in concave
grasps. Distance magnitude is exact point-to-mesh distance (igl AABB). This is the
penetration/contact measure of quality_gate's `penetration` gate.

Usage (standalone):
    python -m gtf.extraction.gates.hoi_contact --segment "seg_name"            # raw chain
    python -m gtf.extraction.gates.hoi_contact --segment "seg_name" --subdir optimized_stage2
"""

import argparse
import json
from pathlib import Path

import igl
import numpy as np

# Winding number of an open/dirty mesh is fuzzy near holes; 0.5 is the standard
# inside/outside decision boundary for the generalized winding number.
_WINDING_INSIDE = 0.5


def signed_distance_to_mesh(points, verts, faces):
    """Signed distance (m) of (P,3) points to a triangle mesh. Negative = inside,
    sign by fast winding number (sound on non-watertight meshes)."""
    P = np.ascontiguousarray(points, dtype=np.float64)
    V = np.ascontiguousarray(verts, dtype=np.float64)
    F = np.ascontiguousarray(faces, dtype=np.int64)
    d2, _, _ = igl.point_mesh_squared_distance(P, V, F)
    w = igl.fast_winding_number(V, F, P)
    sign = np.where(w > _WINDING_INSIDE, -1.0, 1.0)
    return sign * np.sqrt(np.maximum(d2, 0.0))


def side_contact_metrics(hand_verts, hand_valid, obj_verts, obj_faces, contact_mm=10.0):
    """Per-side penetration/contact stats over valid frames.

    Returns None if < 3 valid frames, else dict with:
      pen_p90_mm / pen_max_mm   per-frame deepest penetration (p90 / max over frames)
      min_dist_med_mm           median over frames of the per-frame closest approach
                                (signed; ~0 during grasp, negative = penetrating)
      contact_frac              fraction of valid frames with closest approach < contact_mm
      n_frames                  number of valid frames measured
    """
    rows = np.flatnonzero(np.asarray(hand_valid).astype(bool))
    if len(rows) < 3:
        return None
    pen, min_dist = [], []
    F = np.ascontiguousarray(obj_faces, dtype=np.int64)
    for t in rows:
        s = signed_distance_to_mesh(hand_verts[t], obj_verts[t], F) * 1000.0  # mm
        pen.append(max(0.0, float(-s.min())))
        min_dist.append(float(s.min()))
    pen = np.array(pen)
    min_dist = np.array(min_dist)
    return {
        "pen_p90_mm": round(float(np.percentile(pen, 90)), 2),
        "pen_max_mm": round(float(pen.max()), 2),
        "min_dist_med_mm": round(float(np.median(min_dist)), 2),
        "contact_frac": round(float((min_dist < contact_mm).mean()), 3),
        "n_frames": int(len(rows)),
    }


def segment_contact_metrics(npz, contact_mm=10.0):
    """All-sides metrics for an optimized.npz dict-like. Returns {side: metrics} plus
    a 'worst' summary (max pen_p90 over sides: the gate signal)."""
    out = {}
    for side in ("left", "right"):
        vk, mk = f"hand_{side}_verts", f"hand_{side}_valid"
        if vk not in npz or mk not in npz:
            continue
        m = side_contact_metrics(npz[vk], npz[mk], npz["obj_verts"], npz["obj_faces"], contact_mm)
        if m is not None:
            out[side] = m
    if out:
        worst_side = max(out, key=lambda s: out[s]["pen_p90_mm"])
        out["worst"] = {"side": worst_side, **out[worst_side]}
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.extraction.gates.hoi_contact")
    ap.add_argument("--segment", type=str, required=True)
    ap.add_argument("--subdir", type=str, default="optimized_mano")
    ap.add_argument("--contact-mm", type=float, default=10.0)
    args = ap.parse_args()

    from gtf.utils import RESULTS_DIR

    npz_path = Path(RESULTS_DIR) / args.segment / args.subdir / "optimized.npz"
    d = dict(np.load(npz_path, allow_pickle=True))
    print(json.dumps(segment_contact_metrics(d, args.contact_mm), indent=2))


if __name__ == "__main__":
    main()
