"""Thresholds of the quality gates and the pass / fail decision from a segment's metrics."""


# ── Thresholds ──────────────────────────────────────────────────────────────

THRESHOLDS = {
    "loose": {
        "vol_cv": 1.5,
        "acc_max": 0.2,
        "ang_acc_max": 50,
        "sil_iou": 0.2,
        "area_lo": 0.4,
        "area_hi": 2.5,
        "rel_pose": 5.0,
        "iou_trend": -0.5,
        "hand_rate": 0.15,
        "hand_scale_lo": 0.5,
        "hand_scale_hi": 2.0,
        "hand_scale_drift": 2.2,
        "interact_dist": 0.10,
        "pen_p90": 30.0,
        "obj_size_max_m": 1.2,
        "mask_in_hand_max": 0.35,
        "reground_refusal": 0.80,
        "reground_dist": 32.0,
        "shape_rgb": 0.10,
        "shape_border": 0.6,
        "rot_jerk": 8.0,
        "contact_cover": 0.05,
        "m2d_phantom_sil": 35,
        "m2d_phantom_ratio": 2.8,
        "m2d_phantom_corr": 0.4,
        "m2d_missing_mask": 80,
        "m2d_missing_ratio": 0.45,
        "m2d_missing_corr": 0.15,
        "m2d_wobble_path": 80,
        "m2d_wobble_corr": 0.12,
        "size_drift": 1.75,
        "size_zx": 1.9,
        "size_corr": 0.82,
        "contact_time": 0.12,
    },
    "moderate": {
        "vol_cv": 1.0,
        "acc_max": 0.1,
        "ang_acc_max": 40,
        "sil_iou": 0.3,
        "area_lo": 0.5,
        "area_hi": 2.0,
        "rel_pose": 3.0,
        "iou_trend": -0.3,
        "hand_rate": 0.25,
        "hand_scale_lo": 0.6,
        "hand_scale_hi": 1.6,
        "hand_scale_drift": 1.8,
        "interact_dist": 0.05,
        "pen_p90": 20.0,
        "obj_size_max_m": 0.9,
        "mask_in_hand_max": 0.25,
        "reground_refusal": 0.65,
        "reground_dist": 25.0,
        "shape_rgb": 0.15,
        "shape_border": 0.5,
        "rot_jerk": 6.0,
        "contact_cover": 0.08,
        "m2d_phantom_sil": 25,
        "m2d_phantom_ratio": 2.2,
        "m2d_phantom_corr": 0.5,
        "m2d_missing_mask": 60,
        "m2d_missing_ratio": 0.55,
        "m2d_missing_corr": 0.25,
        "m2d_wobble_path": 60,
        "m2d_wobble_corr": 0.2,
        "size_drift": 1.6,
        "size_zx": 1.65,
        "size_corr": 0.85,
        "contact_time": 0.25,
    },
    "strict": {
        "vol_cv": 0.7,
        "acc_max": 0.05,
        "ang_acc_max": 20,
        "sil_iou": 0.5,
        "area_lo": 0.6,
        "area_hi": 1.5,
        "rel_pose": 1.5,
        "iou_trend": -0.15,
        "hand_rate": 0.40,
        "hand_scale_lo": 0.75,
        "hand_scale_hi": 1.35,
        "hand_scale_drift": 1.4,
        "interact_dist": 0.03,
        "pen_p90": 12.0,
        "obj_size_max_m": 0.7,
        "mask_in_hand_max": 0.20,
        "reground_refusal": 0.60,
        "reground_dist": 18.0,
        "shape_rgb": 0.25,
        "shape_border": 0.4,
        "rot_jerk": 5.0,
        "contact_cover": 0.12,
        "m2d_phantom_sil": 20,
        "m2d_phantom_ratio": 1.8,
        "m2d_phantom_corr": 0.6,
        "m2d_missing_mask": 45,
        "m2d_missing_ratio": 0.65,
        "m2d_missing_corr": 0.35,
        "m2d_wobble_path": 45,
        "m2d_wobble_corr": 0.3,
        "size_drift": 1.5,
        "size_zx": 1.5,
        "size_corr": 0.9,
        "contact_time": 0.40,
    },
}


def evaluate_gates(metrics, thresholds):
    """Evaluate pass/fail for each gate. Returns dict of gate→bool and list of failed gates."""
    ar = metrics["area_ratio"]
    hs, hd = metrics.get("hand_scale_med"), metrics.get("hand_scale_drift")
    mind = metrics.get("min_hand_obj_dist")
    cuts = metrics.get("n_scene_cuts")
    gates = {
        "vol_cv": metrics["vol_cv"] < thresholds["vol_cv"],
        "acc_max": metrics["acc_max"] < thresholds["acc_max"],
        "ang_acc_max": metrics["ang_acc_max"] < thresholds["ang_acc_max"],
        "sil_iou": metrics["sil_iou"] > thresholds["sil_iou"],
        "area_ratio": ar is not None and thresholds["area_lo"] < ar < thresholds["area_hi"],
        "rel_pose": metrics["rel_pose_max"] < thresholds["rel_pose"],
        "iou_trend": metrics["iou_trend"] > thresholds["iou_trend"],
        "hand_rate": metrics["hand_rate"] > thresholds["hand_rate"],
        # hand metric size: None (no tracked hand) defers to hand_rate
        "hand_scale": hs is None or (thresholds["hand_scale_lo"] < hs < thresholds["hand_scale_hi"] and hd < thresholds["hand_scale_drift"]),
        # some tracked hand must come near the object at some point. Object motion alone
        # does not pass: extraction depth drift can move an idle object by tens of cm, and
        # object motion without a hand nearby is unusable for HOI training even when genuine.
        "interaction": mind is not None and mind < thresholds["interact_dist"],
        # a shot boundary inside the segment is fatal at every level
        "scene_cut": not cuts,
        # worst-side p90 hand-into-object penetration, winding-number sign
        # (hoi_contact.py). None (no tracked hand-side) defers to hand_rate.
        "penetration": metrics.get("pen_p90_mm") is None or metrics["pen_p90_mm"] < thresholds["pen_p90"],
        # wrong 3D model that still matches the mask (passes sil_iou): a reconstruction from a
        # border-truncated observation whose self-baked appearance doesn't re-project
        # consistently (e.g. a tall glass reconstructed as a squat disc). A conjunction on
        # purpose: low-texture or hand-wrapped objects read low rgb alone, full-frame
        # objects touch borders alone; only both together is diagnostic. None
        # (unmeasurable rgb) tolerated.
        "obj_shape": not (
            metrics.get("obj_rgb_corr") is not None
            and metrics["obj_rgb_corr"] < thresholds["shape_rgb"]
            and metrics.get("mask_border_frac") is not None
            and metrics["mask_border_frac"] > thresholds["shape_border"]
        ),
        # hand-manipulable object scale: SAM3 can track a person-scale region as "the object"
        # (e.g. a garment being zipped; mask, recon and contact are then all self-consistently
        # wrong together and pass the other gates). Genuine hand-held objects stay well
        # under these thresholds (0.7-1.2 m diagonal). None defers.
        "obj_size": metrics.get("obj_diag_m") is None or metrics["obj_diag_m"] < thresholds.get("obj_size_max_m", 1e9),
        # the mask is the hand (agnostic pointing bound to the hand itself): median SAM3-mask
        # coverage by the projected MANO hand silhouette. Self-consistent through every prior
        # gate (mask, recon and contact all track the hand). None (no packed hands /
        # no usable frames) defers to hand_rate/interaction.
        "obj_is_hand": metrics.get("mask_in_hand") is None or metrics["mask_in_hand"] < thresholds.get("mask_in_hand_max", 1e9),
        # re-grounding verifier (gate 21): fresh Molmo2 evidence vs the tracked mask;
        # catches a wrong tracked object, which every prior gate is structurally blind to.
        # Two clauses, each deferring on None (no evidence): mostly-refused "object held in
        # the hand" (nothing is held: a fixture, a body part, a talking demo, a machine) or
        # held-points far from the tracked mask (something else is held, e.g. a worn item).
        # The anyin axis is recorded but not gated: a genuine pass can track a manipulated
        # object that is not held (the named noun differs from the held one).
        "obj_reground": not (
            (metrics.get("reground_held_refusal") is not None and metrics["reground_held_refusal"] > thresholds.get("reground_refusal", 1e9))
            or (metrics.get("reground_held_dist") is not None and metrics["reground_held_dist"] > thresholds.get("reground_dist", 1e9))
        ),
        # ghost (off-object) contact: 3D-contact hand verts that don't project onto the
        # observed object mask (interaction_metrics.py). None (no 3D-contact frames)
        # defers to the `interaction` gate.
        "contact_cover": metrics.get("contact_cover") is None or metrics["contact_cover"] > thresholds["contact_cover"],
        # non-rigid/utensil grasp wobble: hand<->object 2D relative-rotation jerk over the
        # grasp (the false positive the physics gates miss, e.g. a stirring spoon tracked as a
        # hand). None (no grasp run) defers to the `interaction` gate.
        "rot_jerk": metrics.get("rot_jerk") is None or metrics["rot_jerk"] < thresholds["rot_jerk"],
    }
    # ── Gates 16-18 (motion consistency) ───────────────────────────────────
    # 16 motion2d: the projected object's 2D path must agree with the mask's path.
    # Three clauses, each requiring enough observed/claimed motion to be meaningful:
    # phantom (extraction moves, video doesn't), missing (video moves, extraction
    # doesn't), wobble (both move, uncorrelated). None fields defer.
    m2r, m2c = metrics.get("motion2d_ratio"), metrics.get("motion2d_corr")
    mp, sp = metrics.get("mask_path_pctW"), metrics.get("sil_path_pctW")
    motion2d_ok = True
    if m2r is not None and mp is not None and sp is not None:
        corr = m2c if m2c is not None else 1.0
        phantom = sp >= thresholds["m2d_phantom_sil"] and m2r >= thresholds["m2d_phantom_ratio"] and corr < thresholds["m2d_phantom_corr"]
        missing = mp >= thresholds["m2d_missing_mask"] and m2r <= thresholds["m2d_missing_ratio"] and corr < thresholds["m2d_missing_corr"]
        wobble = max(mp, sp) >= thresholds["m2d_wobble_path"] and corr < thresholds["m2d_wobble_corr"]
        motion2d_ok = not (phantom or missing or wobble)
    gates["motion2d"] = motion2d_ok
    # 17 size_consistency: a conjunction on purpose (each leg alone over-flags): the
    # mask/sil apparent-size ratio drifts (trend disagreement; constant scale offsets are
    # invisible by design), the extraction claims a large depth excursion (attribution:
    # the drift is extraction-side, not mask noise on a transparent/filling object) and
    # in-plane coherence is mediocre (protects genuine camera-follow carries, whose
    # depth excursion is real and whose in-plane agreement is near-perfect).
    dr, zx = metrics.get("area_ratio_drift"), metrics.get("depth_claim_x")
    gates["size_consistency"] = not (
        dr is not None and zx is not None and dr >= thresholds["size_drift"] and zx >= thresholds["size_zx"] and (m2c is None or m2c < thresholds["size_corr"])
    )
    # 18 interaction_sustain: contact must be sustained, not a single approach frame
    # (a hand gesturing near an idle object clears gate 10 with a small contact fraction).
    ctf = metrics.get("contact_frac_time")
    gates["interaction_sustain"] = ctf is None or ctf >= thresholds["contact_time"]
    failed = [g for g, ok in gates.items() if not ok]
    return gates, failed


# ── Metrics judged on the raw extraction ───────────────────────────────────


# Silhouette-derived metrics measure extraction-vs-image fidelity. The stage-2 refinement
# intentionally deviates from the image in the channel the image cannot constrain (metric
# depth), so these are always evaluated on the raw extraction: their job is detecting
# extraction failures (wrong object, lost tracking), not punishing the depth correction.
# Physics/kinematics metrics are evaluated on the refined trajectory (what the model trains on).
# Interaction evidence is also judged on the raw extraction (the refinement's per-entity
# deltas must not change whether a segment counts as interactive); the refined copy is kept
# as `min_hand_obj_dist_refined`, which shows any residual contact tearing.
IMAGE_METRICS = (
    "sil_iou",
    "area_ratio",
    "iou_trend",
    "min_hand_obj_dist",
    "obj_motion_range",
    "obj_rgb_corr",
    "mask_border_frac",
    "sil_iou_p10",
)
