#!/usr/bin/env python
"""WiLoR batch video inference runner (runs in WiLoR's own virtualenv).

Runs the WiLoR hand detector + 3D MANO reconstruction on every frame of a video and
writes a single .npz with camera-space (OpenCV convention, meters) MANO vertices per
side, WiLoR's native MANO parameters, and the raw detection records.

The hand stage (gtf/extraction/hands/run_wilor_hands.py) launches this script inside the
WiLoR virtualenv, directly or through run_wilor_persistent.py.

Usage (must be run with the WiLoR virtualenv's python):
    third-party/wilor/.venv/bin/python gtf/extraction/hands/run_wilor.py \
        --video results/SEG/video.mp4 \
        --K 613.076,611.999,313.028,245.009 \
        --out /tmp/wilor_SEG.npz \
        [--overlay /tmp/wilor_overlay.mp4] [--max-frames N] [--det-conf 0.3]

GPU selection is via CUDA_VISIBLE_DEVICES (set it in the calling shell).

Camera model / focal handling:
  WiLoR predicts a weak-perspective camera pred_cam = [s, tx, ty] in the crop.
  The crop-to-full conversion (t_z = 2*f/(s*box_size); t_x/t_y from box-center
  offsets relative to the principal point) is done here with the given focal
  length and principal point when --K is given (focal_source="true_K").
  Without --K, the method default focal 5000/256*max(H,W) and image-center
  principal point are used (focal_source="default").

  Residual approximation when fx != fy: the weak-perspective model has a single
  scale, so one t_z must serve both axes. We use f = (fx+fy)/2 for t_z; the
  resulting error in t_x/t_y scales with |fx-fy|/f (about 0.1% for typical
  calibrated cameras, below a millimeter). Exact when fx == fy. The principal
  point itself is handled exactly (no image-center assumption).

Output npz keys:
  det_frames (M,) int32, det_is_right (M,) uint8, det_conf (M,) float32,
  det_boxes (M,4) float32 xyxy pixels                      [all raw detections]
  {left,right}_frames (Ns,) int32                          [best det per side/frame]
  {left,right}_verts  (Ns,778,3) float32  camera-space meters, OpenCV (+x right,
                                          +y down, +z forward); translation applied
  {left,right}_joints (Ns,21,3)  float32  same space; OpenPose hand ordering
  {left,right}_conf   (Ns,) float32       detector confidence
  {left,right}_global_orient (Ns,1,3,3), {left,right}_hand_pose (Ns,15,3,3) float32
                                          WiLoR's native MANO rotation matrices in
                                          the right-hand frame (for the left hand
                                          only the verts are x-mirrored)
  {left,right}_betas  (Ns,10) float32     native MANO shape
  K (3,3) float64 (the K actually used), fps () float64, n_frames () int32,
  method="wilor", focal_source ("true_K"|"default"), joint_order (str)
"""

import argparse
import os
import sys
import time
from pathlib import Path

# --- WiLoR checkout on sys.path (before heavy imports) ------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
WILOR_ROOT = _REPO_ROOT / "third-party" / "wilor" / "WiLoR"
sys.path.insert(0, str(WILOR_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

JOINT_ORDER = "openpose_hand21: 0=wrist, 1-4=thumb(base->tip), 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky (WiLoR mano_to_openpose mapping, tips = MANO vertex_ids)"


def parse_args():
    p = argparse.ArgumentParser(description="WiLoR batch video inference")
    p.add_argument("--video", required=True, help="Input video (mp4)")
    p.add_argument("--K", default=None, help="True intrinsics 'fx,fy,cx,cy' (optional)")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--overlay", default=None, help="Optional overlay .mp4 path")
    p.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    p.add_argument("--det-conf", type=float, default=0.3, help="Detector confidence threshold")
    p.add_argument("--rescale-factor", type=float, default=2.0, help="Bbox padding factor (WiLoR demo default 2.0)")
    p.add_argument("--batch-size", type=int, default=32, help="Recon batch size (crops batched across frames)")
    return p.parse_args()


def cam_crop_to_full_K(pred_cam, box_center, box_size, fx, fy, cx, cy):
    """Weak-perspective crop camera -> full-image translation with true K.

    Generalization of wilor.utils.renderer.cam_crop_to_full: replaces the
    image-center principal point with the true (cx, cy) and the assumed focal
    with the true focal. f = (fx+fy)/2 is used for t_z (see module docstring).

    pred_cam: (B,3) [s, tx, ty] (tx already sign-corrected for left hands)
    box_center: (B,2) pixels; box_size: (B,) pixels.
    Returns (B,3) translation in meters.
    """
    f = 0.5 * (fx + fy)
    s = pred_cam[:, 0]
    bs = box_size * s + 1e-9
    tz = 2.0 * f / bs
    tx = 2.0 * (box_center[:, 0] - cx) / bs + pred_cam[:, 1]
    ty = 2.0 * (box_center[:, 1] - cy) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def main():
    args = parse_args()
    video_path = Path(args.video).resolve()
    out_path = Path(args.out).resolve()
    overlay_path = Path(args.overlay).resolve() if args.overlay else None
    if not video_path.exists():
        sys.exit(f"video not found: {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # WiLoR's load_wilor() hardcodes ./mano_data/ + ./pretrained_models/ paths.
    os.chdir(WILOR_ROOT)

    from ultralytics import YOLO
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.models import load_wilor
    from wilor.utils import recursive_to

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[run_wilor] device={device} video={video_path}")

    model, model_cfg = load_wilor(
        checkpoint_path="./pretrained_models/wilor_final.ckpt",
        cfg_path="./pretrained_models/model_config.yaml",
    )
    model = model.to(device).eval()
    detector = YOLO("./pretrained_models/detector.pt").to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- intrinsics ----------------------------------------------------------
    if args.K:
        fx, fy, cx, cy = [float(v) for v in args.K.split(",")]
        focal_source = "true_K"
    else:
        f_default = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(W, H)
        fx = fy = float(f_default)
        cx, cy = W / 2.0, H / 2.0
        focal_source = "default"
        print(f"[run_wilor] no --K given; using method default focal {fx:.1f}, image-center principal point")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"[run_wilor] K used: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f} ({focal_source})")

    # --- accumulators ---------------------------------------------------------
    det_frames, det_is_right, det_conf_all, det_boxes = [], [], [], []
    results = {
        "left": {"frames": [], "verts": [], "joints": [], "conf": [], "go": [], "hp": [], "betas": []},
        "right": {"frames": [], "verts": [], "joints": [], "conf": [], "go": [], "hp": [], "betas": []},
    }

    pending = []  # buffered crop items awaiting recon
    t_det = 0.0
    t_recon = 0.0
    n_crops = 0

    @torch.no_grad()
    def flush():
        nonlocal t_recon
        if not pending:
            return
        t0 = time.time()
        batch = {
            "img": torch.stack([torch.as_tensor(it["img"]) for it in pending]),
            "right": torch.as_tensor(np.array([it["right"] for it in pending], dtype=np.float32)),
            "box_center": torch.as_tensor(np.stack([it["box_center"] for it in pending]).astype(np.float32)),
            "box_size": torch.as_tensor(np.array([it["box_size"] for it in pending], dtype=np.float32)),
        }
        batch = recursive_to(batch, device)
        out = model(batch)
        multiplier = 2.0 * batch["right"] - 1.0  # +1 right, -1 left
        pred_cam = out["pred_cam"].float().clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]  # un-flip crop tx for left hands
        cam_t = cam_crop_to_full_K(pred_cam, batch["box_center"], batch["box_size"], fx, fy, cx, cy)
        verts = out["pred_vertices"].float()  # (B,778,3) hand-centered, meters
        joints = out["pred_keypoints_3d"].float()  # (B,21,3)
        verts = verts.clone()
        joints = joints.clone()
        verts[:, :, 0] = multiplier[:, None] * verts[:, :, 0]  # mirror x back for left
        joints[:, :, 0] = multiplier[:, None] * joints[:, :, 0]
        verts_cam = (verts + cam_t[:, None, :]).cpu().numpy()
        joints_cam = (joints + cam_t[:, None, :]).cpu().numpy()
        # WiLoR's native MANO params (rotation matrices, pose2rot=False), in the
        # right-hand frame (left is x-mirrored only at the verts); the hand stage
        # ingests them directly.
        mp = out["pred_mano_params"]
        B = len(pending)
        go_b = mp["global_orient"].reshape(B, -1, 3, 3).float().cpu().numpy()  # (B,1,3,3)
        hp_b = mp["hand_pose"].reshape(B, -1, 3, 3).float().cpu().numpy()  # (B,15,3,3)
        be_b = mp["betas"].reshape(B, -1).float().cpu().numpy()  # (B,10)
        for i, it in enumerate(pending):
            side = "right" if it["right"] > 0.5 else "left"
            results[side]["frames"].append(it["frame"])
            results[side]["verts"].append(verts_cam[i])
            results[side]["joints"].append(joints_cam[i])
            results[side]["conf"].append(it["conf"])
            results[side]["go"].append(go_b[i])
            results[side]["hp"].append(hp_b[i])
            results[side]["betas"].append(be_b[i])
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_recon += time.time() - t0
        pending.clear()

    # --- main loop ------------------------------------------------------------
    frame_idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        t0 = time.time()
        det = detector(frame, conf=args.det_conf, verbose=False)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_det += time.time() - t0

        data = det.boxes.data.cpu().numpy()  # (N,6) x1,y1,x2,y2,conf,cls(0=left,1=right)
        best = {}  # side -> (conf, box)
        for row in data:
            box, conf, cls = row[:4], float(row[4]), int(row[5])
            det_frames.append(frame_idx)
            det_is_right.append(cls)
            det_conf_all.append(conf)
            det_boxes.append(box.astype(np.float32))
            side = "right" if cls == 1 else "left"
            if side not in best or conf > best[side][0]:
                best[side] = (conf, box)

        if best:
            boxes = np.stack([b for _, b in best.values()])
            right = np.array([1.0 if s == "right" else 0.0 for s in best.keys()], dtype=np.float32)
            confs = [c for c, _ in best.values()]
            ds = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=args.rescale_factor)
            for i in range(len(ds)):
                item = ds[i]
                pending.append(
                    {
                        "img": item["img"],
                        "right": float(item["right"]),
                        "box_center": np.asarray(item["box_center"], dtype=np.float32),
                        "box_size": float(item["box_size"]),
                        "frame": frame_idx,
                        "conf": float(confs[i]),
                    }
                )
                n_crops += 1
        if len(pending) >= args.batch_size:
            flush()
        frame_idx += 1
    flush()
    cap.release()
    n_frames = frame_idx
    t_total = time.time() - t_start
    print(f"[run_wilor] {n_frames} frames, {len(det_frames)} raw detections, {n_crops} recon crops")
    print(
        f"[run_wilor] timing: det {t_det / max(n_frames, 1) * 1000:.1f} ms/frame, "
        f"recon {t_recon / max(n_frames, 1) * 1000:.1f} ms/frame "
        f"({t_recon / max(n_crops, 1) * 1000:.1f} ms/crop), total {t_total / max(n_frames, 1) * 1000:.1f} ms/frame"
    )

    # --- save -----------------------------------------------------------------
    out_dict = {
        "det_frames": np.array(det_frames, dtype=np.int32),
        "det_is_right": np.array(det_is_right, dtype=np.uint8),
        "det_conf": np.array(det_conf_all, dtype=np.float32),
        "det_boxes": (np.stack(det_boxes) if det_boxes else np.zeros((0, 4))).astype(np.float32),
        "K": K,
        "fps": np.float64(fps),
        "n_frames": np.int32(n_frames),
        "method": "wilor",
        "focal_source": focal_source,
        "joint_order": JOINT_ORDER,
    }
    for side in ("left", "right"):
        r = results[side]
        ns = len(r["frames"])
        out_dict[f"{side}_frames"] = np.array(r["frames"], dtype=np.int32)
        out_dict[f"{side}_verts"] = (np.stack(r["verts"]) if ns else np.zeros((0, 778, 3))).astype(np.float32)
        out_dict[f"{side}_joints"] = (np.stack(r["joints"]) if ns else np.zeros((0, 21, 3))).astype(np.float32)
        out_dict[f"{side}_conf"] = np.array(r["conf"], dtype=np.float32)
        # native MANO params (rotation matrices)
        out_dict[f"{side}_global_orient"] = (np.stack(r["go"]) if ns else np.zeros((0, 1, 3, 3))).astype(np.float32)
        out_dict[f"{side}_hand_pose"] = (np.stack(r["hp"]) if ns else np.zeros((0, 15, 3, 3))).astype(np.float32)
        out_dict[f"{side}_betas"] = (np.stack(r["betas"]) if ns else np.zeros((0, 10))).astype(np.float32)
        print(f"[run_wilor] {side}: {ns} frames with output")
    temporary = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **out_dict)
        with np.load(temporary, allow_pickle=False) as archive:
            if "n_frames" not in archive.files or "K" not in archive.files:
                raise RuntimeError("temporary WiLoR output is incomplete")
        os.replace(temporary, out_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[run_wilor] wrote {out_path}")

    # --- overlay (projects the saved camera-space verts with the same K) ------
    if overlay_path is not None:
        write_overlay(video_path, overlay_path, out_dict, args.max_frames)
        print(f"[run_wilor] wrote overlay {overlay_path}")


def write_overlay(video_path, overlay_path, out_dict, max_frames):
    K = out_dict["K"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    by_frame = {"left": {}, "right": {}}
    for side in ("left", "right"):
        for i, f in enumerate(out_dict[f"{side}_frames"]):
            by_frame[side][int(f)] = i
    boxes_by_frame = {}
    for i, f in enumerate(out_dict["det_frames"]):
        boxes_by_frame.setdefault(int(f), []).append(i)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    colors = {"right": (0, 255, 0), "left": (0, 0, 255)}  # BGR: right=green, left=red
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames is not None and frame_idx >= max_frames):
            break
        for i in boxes_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = out_dict["det_boxes"][i].astype(int)
            side = "right" if out_dict["det_is_right"][i] == 1 else "left"
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[side], 1)
        for side in ("left", "right"):
            i = by_frame[side].get(frame_idx)
            if i is None:
                continue
            v = out_dict[f"{side}_verts"][i]  # (778,3) camera-space meters
            z = np.clip(v[:, 2], 1e-6, None)
            u = (fx * v[:, 0] / z + cx).astype(int)
            w = (fy * v[:, 1] / z + cy).astype(int)
            valid = (u >= 0) & (u < W) & (w >= 0) & (w < H)
            for uu, ww in zip(u[valid][::4], w[valid][::4]):
                cv2.circle(frame, (uu, ww), 1, colors[side], -1)
        vw.write(frame)
        frame_idx += 1
    cap.release()
    vw.release()


if __name__ == "__main__":
    main()
