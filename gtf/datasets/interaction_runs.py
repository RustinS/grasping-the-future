"""Interaction-window detection for splitting ground-truth manipulation recordings.

Long recordings (OakInk2 whole-task sequences, HO-Cap multi-object sessions) contain
stretches where the exported object sits idle; windows sampled there teach the model to
stay put. This module finds the disjoint interaction windows of a canonical segment so
each can be exported as its own child segment:

  per frame (against the exported object):
    contact[i]  = min dist(any valid hand mesh, object surface) <= contact_m
    moving[i]   = world-frame object displacement over +-motion_halfwin_s >= motion_m
                  (world keys where present: camera keys of egocentric sources would
                  add camera motion to the object motion)
    interacting = contact | moving  (the motion clause keeps tool-mediated manipulation,
                  e.g. a brush moving a bowl held by the other hand)

  runs: interacting frames -> merge gaps <= gap_s -> drop cores < min_core_s

  Each run start walks backward while the hand is within approach_radius_m of the object
  (up to approach_max_s, at least min_pad_s) so the reach is included; the end gets
  release_pad_s. Extensions stop at the midpoints between cores, so windows are disjoint
  and stay inside the segment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class InteractionWindow:
    start: int  # inclusive, canonical frame index
    end: int  # inclusive
    core_start: int
    core_end: int
    contact_frac: float
    moving_frac: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def hand_object_distance(data: dict, max_obj_verts: int = 800, hand_stride: int = 12) -> np.ndarray:
    """(T,) min distance between any valid hand mesh and the object surface, meters.

    Computed in the object's canonical frame (the hand transformed by the inverse
    object pose), so one KDTree serves every frame. inf where no hand is valid.
    """
    T = data["obj_rotation"].shape[0]
    R = Rotation.from_rotvec(np.asarray(data["obj_rotation"], dtype=np.float64)).as_matrix()
    tr = np.asarray(data["obj_translation"], dtype=np.float64)
    verts0 = np.asarray(data["obj_canonical_verts"], dtype=np.float64) * np.asarray(data["obj_scale"], dtype=np.float64).reshape(1, 3)
    sub = verts0[np.linspace(0, len(verts0) - 1, min(max_obj_verts, len(verts0))).round().astype(int)]
    kd = cKDTree(sub)

    dist = np.full(T, np.inf, dtype=np.float64)
    for side in ("left", "right"):
        valid = np.asarray(data[f"hand_{side}_valid"]).astype(bool)
        if not valid.any():
            continue
        hv = data[f"hand_{side}_verts"]
        for i in np.flatnonzero(valid):
            h_obj = (np.asarray(hv[i][::hand_stride], dtype=np.float64) - tr[i]) @ R[i]
            d, _ = kd.query(h_obj, k=1)
            dist[i] = min(dist[i], float(d.min()))
    return dist


def object_motion_flags(data: dict, fps: float, motion_m: float = 0.008, motion_halfwin_s: float = 0.17, rot_deg: float = 6.0) -> np.ndarray:
    """(T,) bool: object displaced >= motion_m OR rotated >= rot_deg over +-motion_halfwin_s.

    World frame where available. The rotation clause catches in-place manipulation
    (an object turned in the hand with almost no centroid displacement).
    """
    rot_key = "obj_rotation_world" if "obj_rotation_world" in data else "obj_rotation"
    tr_key = "obj_translation_world" if "obj_translation_world" in data else "obj_translation"
    R = Rotation.from_rotvec(np.asarray(data[rot_key], dtype=np.float64)).as_matrix()
    tr = np.asarray(data[tr_key], dtype=np.float64)
    verts0 = np.asarray(data["obj_canonical_verts"], dtype=np.float64) * np.asarray(data["obj_scale"], dtype=np.float64).reshape(1, 3)
    c0 = verts0.mean(0)
    cent = np.einsum("tij,j->ti", R, c0) + tr
    T = cent.shape[0]
    w = max(1, int(round(motion_halfwin_s * fps)))
    lo = np.clip(np.arange(T) - w, 0, T - 1)
    hi = np.clip(np.arange(T) + w, 0, T - 1)
    moving = np.linalg.norm(cent[hi] - cent[lo], axis=1) >= motion_m
    rel = np.einsum("tij,tkj->tik", R[hi], R[lo])  # R_hi @ R_lo^T
    ang = np.degrees(np.arccos(np.clip((np.einsum("tii->t", rel) - 1.0) / 2.0, -1.0, 1.0)))
    return moving | (ang >= rot_deg)


def interaction_windows(
    data: dict,
    fps: float,
    *,
    contact_m: float = 0.03,
    motion_m: float = 0.008,
    rot_deg: float = 6.0,
    motion_halfwin_s: float = 0.17,
    gap_s: float = 2.0,
    min_core_s: float = 1.0,
    approach_radius_m: float = 0.45,
    approach_max_s: float = 2.0,
    min_pad_s: float = 0.75,
    release_pad_s: float = 0.75,
) -> list[InteractionWindow]:
    """Detect disjoint interaction windows. Empty list = no interaction at all."""
    dist = hand_object_distance(data)
    contact = dist <= contact_m
    moving = object_motion_flags(data, fps, motion_m=motion_m, motion_halfwin_s=motion_halfwin_s, rot_deg=rot_deg)
    inter = contact | moving
    T = len(inter)
    idx = np.flatnonzero(inter)
    if len(idx) == 0:
        return []

    gap_f = gap_s * fps
    cores: list[tuple[int, int]] = []
    s = e = int(idx[0])
    for i in idx[1:]:
        if i - e <= gap_f:
            e = int(i)
        else:
            cores.append((s, e))
            s = e = int(i)
    cores.append((s, e))
    cores = [(s, e) for s, e in cores if (e - s + 1) >= min_core_s * fps]
    if not cores:
        # The core floor drops short blips inside long segments; a short segment that is
        # mostly interaction (SHOW3D emits 0.5-1 s runs) becomes one full-span window.
        if T < 2 * min_core_s * fps and float(inter.mean()) >= 0.35:
            return [
                InteractionWindow(
                    start=0,
                    end=T - 1,
                    core_start=int(idx[0]),
                    core_end=int(idx[-1]),
                    contact_frac=float(contact.mean()),
                    moving_frac=float(moving.mean()),
                )
            ]
        return []

    min_pad_f = int(round(min_pad_s * fps))
    approach_max_f = int(round(approach_max_s * fps))
    release_f = int(round(release_pad_s * fps))
    out: list[InteractionWindow] = []
    for k, (cs, ce) in enumerate(cores):
        lo = 0 if k == 0 else (cores[k - 1][1] + cs) // 2 + 1
        hi = T - 1 if k == len(cores) - 1 else (ce + cores[k + 1][0]) // 2
        # Include the reach: walk back while the hand is within approach_radius of the
        # object (capped), and at least min_pad.
        s2 = cs
        while s2 - 1 >= max(lo, cs - approach_max_f) and dist[s2 - 1] <= approach_radius_m:
            s2 -= 1
        s2 = max(lo, min(s2, cs - min_pad_f))  # at least min_pad, at most the approach walk, never past lo
        e2 = min(hi, ce + release_f)
        w = InteractionWindow(
            start=int(s2),
            end=int(e2),
            core_start=int(cs),
            core_end=int(ce),
            contact_frac=float(contact[s2 : e2 + 1].mean()),
            moving_frac=float(moving[s2 : e2 + 1].mean()),
        )
        out.append(w)
    return out


def is_full_span(windows: list[InteractionWindow], n_frames: int, tol_frames: int = 16) -> bool:
    """True when the detection is a no-op: one window covering ~the whole segment."""
    if len(windows) != 1:
        return False
    return windows[0].start <= tol_frames and (n_frames - 1 - windows[0].end) <= tol_frames
