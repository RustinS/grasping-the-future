#!/usr/bin/env python3
"""Run Molmo2-8B pointing on extracted video segments.

Prompts the model to point at the in-hand object: "point to the {object} in hand"
when a groundable object name is known, else the object-agnostic "point to the
object held in the hand" (gtf.utils.build_pointing_prompt). Works with or without a
fixed --object.

Frame policy: rank candidate frames by hand presence (the hand stage always runs before
pointing) and sharpness (pointing_frames module), collect all points of each generation,
validate the set (cluster-outlier filter + hand proximity), and move on to the next
candidate frame when one fails. A frame is accepted with >= 2 validated points, or 1 that
passed the proximity check.

Usage:
    python -m gtf.extraction.frontend.run_molmo2
    python -m gtf.extraction.frontend.run_molmo2 --max-videos 2
    python -m gtf.extraction.frontend.run_molmo2 --force

Outputs go to results/{segment_name}/molmo2/:
    - frame.jpg        (frame used for pointing)
    - points.json      (collected point coordinates + candidate-frame evidence)
    - .molmo2_done     (marker file)
"""

import argparse
import json
import os
import re
import tempfile
import traceback
from pathlib import Path

import torch
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from gtf.utils import RESULTS_DIR as _RESULTS_DIR, build_pointing_prompt, get_segment_object, is_action_valid, load_manifest, rprint as print
from gtf.utils.video_timing import file_sha256

RESULTS_DIR = Path(_RESULTS_DIR)
MODEL_ID = "allenai/Molmo2-8B"
MODEL_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"

COORD_RE = re.compile(r"<(?:points|tracks)[^>]*?\bcoords=\"([0-9\t:;, .]+)\"[^>]*/?>")
FRAME_RE = re.compile(r"(?:^|[\t:;,])\s*([0-9.]+)\s+([0-9. ]+)")
POINT_RE = re.compile(r"([0-9]+)\s+([0-9]{3,4})\s+([0-9]{3,4})")

COLORS = [(255, 50, 50), (50, 200, 50), (50, 100, 255), (255, 165, 0), (180, 50, 255)]

ATTEMPTS_PER_FRAME = 4  # generations per candidate frame (all points of a generation are taken)
MAX_TOTAL_ATTEMPTS = 16  # generation budget across all candidates (up to 4 frames, the default frame included)
MAX_KEEP_POINTS = 8  # cap on validated points written for SAM3


def extract_points(text, img_w, img_h):
    """Extract pointing coordinates from Molmo2 output text.

    Returns list of (raw_x, raw_y, pixel_x, pixel_y) tuples.
    """

    def _parse(matches):
        return [
            (int(rx), int(ry), rx / 1000 * img_w, ry / 1000 * img_h)
            for m in matches
            for rx, ry in [(float(m.group(2)), float(m.group(3)))]
            if 0 <= rx / 1000 * img_w <= img_w and 0 <= ry / 1000 * img_h <= img_h
        ]

    all_points = []
    for coord_match in COORD_RE.finditer(text):
        coord_str = coord_match.group(1)
        frame_points = []
        for frame_match in FRAME_RE.finditer(coord_str):
            frame_points.extend(_parse(POINT_RE.finditer(frame_match.group(2))))
        all_points.extend(frame_points or _parse(POINT_RE.finditer(coord_str)))
    return all_points or _parse(POINT_RE.finditer(text))


def best_frame_index(name, obj, video_fps, total_frames):
    """Pick middle frame of longest valid time range, or video midpoint."""
    fallback = total_frames // 2
    if not obj:
        return fallback, None
    try:
        with open(RESULTS_DIR / name / "valid_action" / obj / "result.json") as f:
            ranges = json.load(f).get("timestamps", {}).get("ranges", [])
    except (json.JSONDecodeError, OSError):
        return fallback, None

    if not ranges:
        return fallback, None

    longest = max(ranges, key=lambda r: r["end"] - r["start"])
    mid_sec = (longest["start"] + longest["end"]) / 2.0
    return min(int(mid_sec * video_fps), total_frames - 1), longest


def outputs_complete(molmo_dir, video_path=None):
    """Validate the published pointing transaction before a resume skips it."""

    molmo_dir = Path(molmo_dir)
    try:
        if (molmo_dir / ".molmo2_done").read_text() != "done\n":
            return False
        payload = json.loads((molmo_dir / "points.json").read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("points"), list) or not payload["points"] or not isinstance(payload.get("frame_index"), int):
            return False
        if video_path is not None:
            video_path = Path(video_path)
            if (
                not video_path.is_file()
                or payload.get("video_sha256") != file_sha256(video_path)
                or not isinstance(payload.get("video_frames"), int)
                or isinstance(payload.get("video_frames"), bool)
                or payload["video_frames"] <= 0
                or not 0 <= payload["frame_index"] < payload["video_frames"]
            ):
                return False
        with Image.open(molmo_dir / "frame.jpg") as image:
            if image.size != (
                payload.get("image_width"),
                payload.get("image_height"),
            ):
                return False
            image.verify()
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _stat_fingerprint(path):
    stat = Path(path).stat()
    return int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns)


def _stable_video_binding(video_path, initial_stat, total_frames):
    """Hash exactly the media generation used to select the prompt frame."""

    if _stat_fingerprint(video_path) != initial_stat:
        raise RuntimeError(f"video changed during Molmo2 pointing: {video_path}")
    digest = file_sha256(video_path)
    if _stat_fingerprint(video_path) != initial_stat:
        raise RuntimeError(f"video changed while Molmo2 bound its output: {video_path}")
    return {"video_sha256": digest, "video_frames": int(total_frames)}


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path, payload):
    _atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _atomic_jpeg(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_text = tempfile.mkstemp(prefix=f".{path.name}.tmp-", suffix=".jpg", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_text)
    try:
        image.save(temporary, format="JPEG", quality=95)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model {MODEL_ID}@{MODEL_REVISION}...")
    kwargs = dict(
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        revision=MODEL_REVISION,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    print("Model loaded successfully")
    return processor, model


def run_pointing(processor, model, image, obj, temperature=0.7):
    """Run a single pointing inference. Returns generated text.

    Uses the named prompt when *obj* is groundable, else an agnostic fallback
    (build_pointing_prompt), so pointing works without a fixed object name.
    """
    prompt = build_pointing_prompt(obj)
    messages = [{"role": "user", "content": [dict(type="image", image=image), dict(type="text", text=prompt)]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=temperature)

    new_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def point_segment(processor, model, name, obj, *, num_points=5, temperature=0.7):
    """Point one segment; writes frame.jpg / points.json / .molmo2_done under results/{name}/molmo2/.

    Returns "done" | "failed" | "no_video". Shared by the run_molmo2 command line and
    the Action100M pipeline.
    """
    from gtf.extraction.frontend.pointing_frames import accept_points, filter_points, hand_joints_2d, select_candidate_frames

    seg_dir = RESULTS_DIR / name
    molmo_dir = seg_dir / "molmo2"
    video_path = seg_dir / "video.mp4"
    if not video_path.exists():
        return "no_video"
    initial_video_stat = _stat_fingerprint(video_path)
    molmo_dir.mkdir(parents=True, exist_ok=True)
    (molmo_dir / ".molmo2_done").unlink(missing_ok=True)

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    default_idx, valid_range = best_frame_index(name, obj, video_fps, total_frames)

    def grab(reader, fi):
        f = reader[fi]
        return Image.fromarray(f.numpy() if hasattr(f, "numpy") else f.asnumpy())

    candidates = select_candidate_frames(seg_dir, video_path, valid_range=valid_range, fps=video_fps, default_idx=default_idx)
    hand2d = hand_joints_2d(seg_dir, [c["frame"] for c in candidates])

    chosen = None
    weak_accept = None  # 1-point accepted candidate: provisional, a >=2-point accept on a later frame wins
    best_fallback = None  # (n_valid, cand, kept, img)
    cand_reports = []
    attempts_total = 0
    for cand in candidates:
        if attempts_total >= MAX_TOTAL_ATTEMPTS:
            break
        img = grab(vr, cand["frame"])
        img_w, img_h = img.size
        collected, seen_raw = [], set()
        for _ in range(ATTEMPTS_PER_FRAME):
            if attempts_total >= MAX_TOTAL_ATTEMPTS or len(collected) >= num_points:
                break
            attempts_total += 1
            raw_text = run_pointing(processor, model, img, obj=obj, temperature=temperature)
            for raw_x, raw_y, px, py in extract_points(raw_text, img_w, img_h):
                if (raw_x, raw_y) in seen_raw:
                    continue
                seen_raw.add((raw_x, raw_y))
                collected.append({"raw_x": raw_x, "raw_y": raw_y, "pixel_x": round(px, 2), "pixel_y": round(py, 2), "raw_text": raw_text})
        keep_idx, info = filter_points([(p["pixel_x"], p["pixel_y"]) for p in collected], (img_w, img_h), hand_pts=hand2d.get(cand["frame"]))
        kept = [collected[i] for i in keep_idx][:MAX_KEEP_POINTS]
        accepted = accept_points(info)
        strength = ("strong" if info["n_valid"] >= 2 else "weak") if accepted else None
        rep = {**cand, **info, "accepted": accepted, "accept_strength": strength}
        cand_reports.append(rep)
        print(
            f"    frame {cand['frame']}: hand={cand['hand_present']} sharp={cand['sharpness']:.0f}"
            f" pts {info['n_raw']}->{info['n_valid']} hand_dist={info['hand_dist_frac']}"
            f" -> {strength.upper() + ' ACCEPT' if accepted else 'reject'}"
        )
        if accepted and strength == "strong":
            chosen = (cand, kept, img)
            break
        if accepted:  # weak (single-point) accept: hold as provisional, keep hunting
            if weak_accept is None:
                weak_accept = (cand, kept, img)
            continue
        if best_fallback is None or info["n_valid"] > best_fallback[0]:
            best_fallback = (info["n_valid"], cand, kept, img)
    del vr

    if chosen is not None or weak_accept is not None:
        cand, kept, img = chosen if chosen is not None else weak_accept
        status = "done"
    elif best_fallback is not None:
        _, cand, kept, img = best_fallback
        status = "failed"
        print(f"    WARNING: no candidate accepted ({len(cand_reports)} tried) — best had {len(kept)} validated point(s)")
    else:
        return "failed"

    _atomic_jpeg(molmo_dir / "frame.jpg", img)
    result = {
        "frame_index": cand["frame"],
        "frame_timestamp": round(cand["frame"] / video_fps, 2),
        "valid_range": valid_range,
        "image_width": img.size[0],
        "image_height": img.size[1],
        "object": obj,
        "prompt": build_pointing_prompt(obj),
        "points": kept,
        "pointing_mode": "smart",
        "n_attempts": attempts_total,
        "candidates": cand_reports,
        **_stable_video_binding(video_path, initial_video_stat, total_frames),
    }
    _atomic_json(molmo_dir / "points.json", result)
    if status == "done":
        _atomic_text(molmo_dir / ".molmo2_done", "done\n")
    return status


def render_vis(molmo_dir):
    """Draw collected points on the saved frame, write vis.jpg."""
    frame_path = molmo_dir / "frame.jpg"
    points_path = molmo_dir / "points.json"
    if not frame_path.exists() or not points_path.exists():
        return False

    with open(points_path) as f:
        data = json.load(f)

    img = Image.open(frame_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    radius = max(6, min(w, h) // 80)

    for i, pt in enumerate(data["points"]):
        px, py = pt["pixel_x"], pt["pixel_y"]
        color = COLORS[i % len(COLORS)]
        draw.ellipse([(px - radius, py - radius), (px + radius, py + radius)], fill=color, outline=(255, 255, 255), width=2)
        draw.text((px + radius + 4, py - radius), f"#{i + 1}", fill=color)

    img.save(molmo_dir / "vis.jpg", quality=95)
    return True


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.frontend.run_molmo2", description="Run Molmo2-8B pointing on video segments")
    parser.add_argument("--max-videos", type=int, default=0, help="Process only first N segments (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-points", type=int, default=5, help="Number of unique points to collect")
    parser.add_argument("--visualize", action="store_true", help="Generate point visualization images")
    parser.add_argument("--visualize-only", action="store_true", help="Only generate visualizations, skip inference")
    parser.add_argument("--object", type=str, default=None, help="Only process segments validated for this object (auto-detected if omitted)")
    args = parser.parse_args()
    if args.visualize_only:
        args.visualize = True

    entries = load_manifest()
    if args.max_videos > 0:
        entries = entries[: args.max_videos]
        print(f"Limited to first {args.max_videos} segments")

    total = len(entries)
    names = [Path(e["file"]).stem for e in entries]
    names = [n for n in names if is_action_valid(n, args.object)]
    _label = f" ({args.object})" if args.object else ""
    print(f"Action validation filter{_label}: {len(names)}/{total} valid")

    processor = model = None
    if not args.visualize_only:
        processor, model = load_model()

    N = len(names)
    done = skipped = failed = 0
    vis_done = vis_skipped = 0

    no_object = 0
    for vi, name in enumerate(names):
        molmo_dir = RESULTS_DIR / name / "molmo2"
        video_path = RESULTS_DIR / name / "video.mp4"

        obj = args.object or get_segment_object(name)
        if not obj:
            print(f"[{vi + 1}/{N}] SKIP (no object name found): {name}")
            no_object += 1
            continue

        if not video_path.exists():
            print(f"[{vi + 1}/{N}] SKIP (no video): {name}")
            failed += 1
            continue

        if not args.visualize_only:
            if not args.force and outputs_complete(molmo_dir, video_path):
                skipped += 1
            else:
                print(f"\n[{vi + 1}/{N}] Processing: {name} (object={obj})")
                try:
                    status = point_segment(processor, model, name, obj, num_points=args.num_points, temperature=args.temperature)
                    if status == "done":
                        done += 1
                        print(f"  Done: {name}")
                    else:
                        failed += 1
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  ERROR processing {name}: {e}")
                    traceback.print_exc()
                    failed += 1
                    continue

        if args.visualize:
            vis_path = molmo_dir / "vis.jpg"
            if not args.force and vis_path.exists() and vis_path.stat().st_size > 0:
                vis_skipped += 1
            elif (molmo_dir / "points.json").exists():
                if render_vis(molmo_dir):
                    vis_done += 1
                    print(f"  Saved visualization: {vis_path}")
                else:
                    print(f"  SKIP vis (missing files): {name}")

    if not args.visualize_only:
        print(f"\n[Molmo2] Inference: {done} processed, {skipped} skipped, {failed} failed")
        if no_object:
            print(f"[Molmo2] {no_object} skipped (no object name)")
    if args.visualize:
        print(f"[Molmo2] Visualization: {vis_done} rendered, {vis_skipped} skipped")


if __name__ == "__main__":
    main()
