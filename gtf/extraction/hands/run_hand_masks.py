#!/usr/bin/env python3
"""Hand masks straight from the RGB (SAM 3 image model, text prompt "hand") at a segment's object keyframes.

Independent evidence for where the hands are, used by the upgraded chain's optimizer: it drops WiLoR detections with no
hand under them (WiLoR fires on faces, steam, hand-like objects) and reads the visible depth order of hand and object.
Only the `upgraded` chain uses it (gtf/extraction/chains.py); the orchestrator runs it inside the optimize step when the chain asks for hand masks.

    python -m gtf.extraction.hands.run_hand_masks --segment=NAME [--segment=NAME ...] [--results-dir results]

Writes {segment}/hand_masks/masks.npz: frame_indices (M,), masks = packbits of (M,H,W) bool (union of all "hand" instances)
along the last axis, shape (3,), scores (M,8) instance scores (-1 = none), prompt.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

from gtf.utils import RESULTS_DIR


def keyframes(seg_dir):
    p = os.path.join(seg_dir, "sam3d", "metadata.json")
    if os.path.exists(p):
        return [int(f) for f in json.load(open(p))["keyframe_indices"]]
    p = os.path.join(seg_dir, "optimized_mano", "optimized.npz")
    return [int(f) for f in np.load(p)["frame_indices"]]


def read_frames(video, want):
    want, got, k = set(want), {}, 0
    cap = cv2.VideoCapture(video)
    while len(got) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if k in want:
            got[k] = f
        k += 1
    cap.release()
    return got


def segment_hand_masks(proc, seg_dir, prompt="hand"):
    import torch
    from PIL import Image

    fi = keyframes(seg_dir)
    frames = read_frames(os.path.join(seg_dir, "video.mp4"), fi)
    if not frames:
        raise RuntimeError(f"no frames read from {seg_dir}/video.mp4")
    H, W = next(iter(frames.values())).shape[:2]
    M = np.zeros((len(fi), H, W), bool)
    S = np.full((len(fi), 8), -1.0, np.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for m, f in enumerate(fi):
            if f not in frames:
                continue
            st = proc.set_image(Image.fromarray(cv2.cvtColor(frames[f], cv2.COLOR_BGR2RGB)))
            o = proc.set_text_prompt(prompt, st)
            ms = o.get("masks")
            if ms is not None and len(ms):
                M[m] = ms.squeeze(1).any(0).cpu().numpy().astype(bool)
                sc = o.get("scores")
                if sc is not None:
                    sc = sc.float().cpu().numpy().ravel()[:8]
                    S[m, : len(sc)] = sc
    od = os.path.join(seg_dir, "hand_masks")
    os.makedirs(od, exist_ok=True)
    tmp = os.path.join(od, "masks.tmp.npz")
    np.savez_compressed(tmp, frame_indices=np.array(fi), masks=np.packbits(M, axis=-1), shape=np.array(M.shape), scores=S, prompt=np.array(prompt))
    os.replace(tmp, os.path.join(od, "masks.npz"))
    return len(fi), float(M.any((1, 2)).mean())


def load_hand_masks(seg_dir):
    """{frame_index: (H,W) bool} or None when the segment has no hand masks."""
    p = os.path.join(seg_dir, "hand_masks", "masks.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    shp = tuple(int(x) for x in z["shape"])
    M = np.unpackbits(z["masks"], axis=-1)[..., : shp[2]].astype(bool)
    return {int(f): M[i] for i, f in enumerate(z["frame_indices"])}


def main():
    ap = argparse.ArgumentParser(prog="python -m gtf.extraction.hands.run_hand_masks", description=__doc__.split("\n")[0])
    ap.add_argument("--segment", action="append", required=True)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--prompt", default="hand")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.4)
    for s in a.segment:
        d = os.path.join(a.results_dir, s)
        if os.path.exists(os.path.join(d, "hand_masks", "masks.npz")) and not a.force:
            continue
        try:
            n, frac = segment_hand_masks(proc, d, a.prompt)
            print(f"{s}: {n} keyframes, {100 * frac:.0f}% with a hand", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"{s}: FAILED {ex!r}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
