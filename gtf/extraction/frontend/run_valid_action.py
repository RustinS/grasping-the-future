#!/usr/bin/env python3
"""Validate action segments using Molmo2-8B video mode.

The validate step of the pipeline (gtf.extraction.action100m imports its functions; this file
also runs standalone). Copies each manifest video from segments/ into
results/{segment_name}/, runs the validation and saves the verdict. Later stages
only process segments with valid == true.

Validation is object-agnostic: it asks "is an object being manipulated by hand?", so a
clip is not dropped for mismatching a specific noun. The discovered object name is only
a soft label: it keys valid_action/{object}/ and is used for pointing and metadata;
result.json records validation_mode="agnostic".
--object is optional (it filters / labels one object); omit it to run on all segments.

Usage:
    python -m gtf.extraction.frontend.run_valid_action              # object-agnostic (default)
    python -m gtf.extraction.frontend.run_valid_action --max-videos 2
    python -m gtf.extraction.frontend.run_valid_action --object cup  # optional: only this object's segments
    python -m gtf.extraction.frontend.run_valid_action --force

Outputs per segment in results/{segment_name}/:
    video.mp4, metadata.json, valid_action/{object}/result.json, valid_action/{object}/.done
    ({object} = the discovered noun, or "object" when none was discovered)
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch

from gtf.utils import FRAME_PROMPT_AGNOSTIC, RESULTS_DIR as _RESULTS_DIR, SEGMENTS_DIR as _SEGMENTS_DIR, VIDEO_PROMPT_AGNOSTIC, get_segment_object, load_manifest, rprint as print

SEGMENTS_DIR = Path(_SEGMENTS_DIR)
RESULTS_DIR = Path(_RESULTS_DIR)
MODEL_ID = "allenai/Molmo2-8B"


def prepare_segment(entry, obj=None):
    """Create results dir, copy video, write metadata. Returns (name, video_path_str or None).

    If *obj* is given it is stored in metadata.json as ``"object"``.
    The entry's own ``"object"`` field (if present) is used as fallback.
    """
    name = Path(entry["file"]).stem
    seg_dir = RESULTS_DIR / name
    seg_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(entry)
    effective_obj = obj or meta.get("object")
    if effective_obj:
        meta["object"] = effective_obj
    (seg_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    src = SEGMENTS_DIR / entry["file"]
    dst = seg_dir / "video.mp4"
    if not src.exists():
        return name, None
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return name, str(dst)


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading model {MODEL_ID}...")
    kwargs = dict(trust_remote_code=True, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    print("Model loaded successfully")
    return processor, model


def run_video_query(processor, model, video_path, prompt):
    """Run single video QA inference. Returns generated text."""
    messages = [{"role": "user", "content": [dict(type="video", video=video_path), dict(type="text", text=prompt)]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_video_query_batch(processor, model, video_paths, prompts):
    """Run greedy video QA for several clips in one model call.

    Decoder-only generation must be left padded: right padding makes shorter
    examples start generation from a padding token and can change the answer.
    With left padding, each row has the same prompt width and greedy decoding is
    equivalent to the singleton path while amortizing the model launch.
    """
    if len(video_paths) != len(prompts):
        raise ValueError("video_paths and prompts must have the same length")
    if not video_paths:
        return []
    if len(video_paths) == 1:
        return [run_video_query(processor, model, video_paths[0], prompts[0])]

    batch_messages = [[{"role": "user", "content": [dict(type="video", video=path), dict(type="text", text=prompt)]}] for path, prompt in zip(video_paths, prompts)]
    tokenizer = processor.tokenizer
    previous_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    responses = [tokenizer.decode(generated_ids[row, input_len:], skip_special_tokens=True) for row in range(generated_ids.shape[0])]
    return responses


def extract_frames_1fps(video_path):
    """Extract one frame per second. Returns (timestamps, pil_images, fps)."""
    from decord import VideoReader, cpu
    from PIL import Image

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    fps = vr.get_avg_fps()
    n_frames = len(vr)
    duration = n_frames / fps

    timestamps, indices = [], []
    t = 0.0
    while t < duration:
        idx = int(t * fps)
        if idx >= n_frames:
            break
        indices.append(idx)
        timestamps.append(round(t, 2))
        t += 1.0

    frames = vr.get_batch(indices).asnumpy()
    del vr
    return timestamps, [Image.fromarray(f) for f in frames], fps


def run_frame_batch(processor, model, images, prompt, batch_size=8):
    """Run yes/no query on images in batches. Returns list of response strings."""
    responses = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        batch_messages = [[{"role": "user", "content": [dict(type="image", image=img), dict(type="text", text=prompt)]}] for img in batch]
        inputs = processor.apply_chat_template(batch_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        input_len = inputs["input_ids"].shape[1]
        for j in range(generated_ids.shape[0]):
            responses.append(processor.tokenizer.decode(generated_ids[j, input_len:], skip_special_tokens=True))

        del inputs, generated_ids
        torch.cuda.empty_cache()
    return responses


def detect_timestamps(processor, model, video_path, batch_size=8):
    """Detect the seconds of the video where an object is manipulated (one frame per second).

    Returns a dict with the per-frame verdicts, the contiguous ranges and the valid / total counts.
    """
    timestamps, images, _ = extract_frames_1fps(video_path)
    print(f"  Timestamp detection: {len(timestamps)} frames at 1fps (batch_size={batch_size})")

    responses = run_frame_batch(processor, model, images, FRAME_PROMPT_AGNOSTIC, batch_size)
    del images
    verdicts = [parse_valid(r) for r in responses]

    frame_results = [{"timestamp": ts, "valid": v, "raw_response": r.strip()} for ts, v, r in zip(timestamps, verdicts, responses)]

    # Merge consecutive valid timestamps into ranges
    ranges = []
    range_start = None
    for ts, v in zip(timestamps, verdicts):
        if v:
            if range_start is None:
                range_start = ts
        elif range_start is not None:
            ranges.append({"start": range_start, "end": ts})
            range_start = None
    if range_start is not None:
        ranges.append({"start": range_start, "end": timestamps[-1] + 1.0})

    valid_seconds = sum(verdicts)
    print(f"  Timestamps: {valid_seconds}/{len(timestamps)} frames valid, {len(ranges)} contiguous range(s)")
    return {"frames": frame_results, "ranges": ranges, "valid_frame_count": valid_seconds, "total_frame_count": len(timestamps)}


def parse_valid(response_text):
    """Parse yes/no response into bool. Conservative: ambiguous defaults to False."""
    text = response_text.strip().lower()
    first_word = re.split(r"[^a-z]", text)[0]
    if first_word == "yes":
        return True
    if first_word == "no":
        return False
    if re.search(r"\byes\b", text) and not re.search(r"\bno\b", text):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.frontend.run_valid_action", description="Validate action segments using Molmo2-8B video mode")
    parser.add_argument("--object", default=None, help="Object to check for (auto-detected from manifest/metadata if omitted)")
    parser.add_argument("--max-videos", type=int, default=0, help="Process only first N segments (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for frame-level timestamp detection")
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=int(os.environ.get("VALID_ACTION_VIDEO_BATCH_SIZE", "4")),
        help="Number of clips per greedy validation call (default: 4)",
    )
    args = parser.parse_args()
    if args.video_batch_size < 1:
        parser.error("--video-batch-size must be at least 1")

    if args.object:
        print(f"Object override: {args.object}")

    entries = load_manifest()
    if args.max_videos > 0:
        entries = entries[: args.max_videos]
        print(f"Limited to first {args.max_videos} segments")

    print("Preparing segment directories and copying videos...")
    prepared = [(prepare_segment(e, obj=args.object), e) for e in entries]
    print(f"Prepared {len(prepared)} segment dirs in {RESULTS_DIR}")

    N = len(prepared)
    done = skipped = failed = valid_count = invalid_count = no_object = 0
    work = []

    for vi, ((name, video_path), entry) in enumerate(prepared):
        obj = args.object or entry.get("object") or get_segment_object(name)
        if not obj:
            print(f"[{vi + 1}/{N}] SKIP (no object name found): {name}")
            no_object += 1
            continue

        out_dir = RESULTS_DIR / name / "valid_action" / obj
        prompt = VIDEO_PROMPT_AGNOSTIC

        if video_path is None:
            print(f"[{vi + 1}/{N}] SKIP (no source video): {name}")
            failed += 1
            continue

        if not args.force and (out_dir / ".done").exists() and (out_dir / "result.json").exists():
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        work.append(
            {
                "index": vi,
                "name": name,
                "video_path": video_path,
                "object": obj,
                "out_dir": out_dir,
                "prompt": prompt,
            }
        )

    if not work:
        print(f"\n[ValidAction] 0 processed, {skipped} skipped, {failed} failed")
        if no_object:
            print(f"[ValidAction] {no_object} skipped (no object name)")
        return

    processor, model = load_model()

    for batch_start in range(0, len(work), args.video_batch_size):
        batch = work[batch_start : batch_start + args.video_batch_size]
        names = ", ".join(item["name"] for item in batch)
        print(f"\n[ValidAction] Video batch {batch_start // args.video_batch_size + 1}: {names}")

        batch_responses = {}
        try:
            responses = run_video_query_batch(
                processor,
                model,
                [item["video_path"] for item in batch],
                [item["prompt"] for item in batch],
            )
            batch_responses = {item["name"]: response for item, response in zip(batch, responses)}
        except Exception as batch_error:
            # A malformed clip or an unexpectedly long batch must not fail its
            # neighbors: retry the clips of the batch one at a time.
            print(f"  Batch failed ({batch_error}); retrying its {len(batch)} clips individually")
            torch.cuda.empty_cache()
            for item in batch:
                try:
                    batch_responses[item["name"]] = run_video_query(processor, model, item["video_path"], item["prompt"])
                except Exception as item_error:
                    print(f"  ERROR validating {item['name']}: {item_error}")

        for item in batch:
            vi = item["index"]
            name = item["name"]
            video_path = item["video_path"]
            obj = item["object"]
            out_dir = item["out_dir"]
            prompt = item["prompt"]
            raw_response = batch_responses.get(name)
            print(f"\n[{vi + 1}/{N}] Processing: {name} (object={obj})")
            if raw_response is None:
                failed += 1
                continue

            try:
                valid = parse_valid(raw_response)
                result = {
                    "valid": valid,
                    "object": obj,
                    "validation_mode": "agnostic",
                    "prompt": prompt,
                    "raw_response": raw_response,
                }

                if valid:
                    valid_count += 1
                    print(f"  [bold green]VALID[/bold green]: {raw_response[:120]}")
                    result["timestamps"] = detect_timestamps(processor, model, video_path, args.batch_size)
                else:
                    invalid_count += 1
                    print(f"  [bold red]INVALID[/bold red]: {raw_response[:120]}")

                (out_dir / "result.json").write_text(json.dumps(result, indent=2))
                (out_dir / ".done").write_text("done\n")
                done += 1
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"  OOM processing {name}, skipping")
                torch.cuda.empty_cache()
                failed += 1

            except Exception as e:
                print(f"  ERROR processing {name}: {e}")
                failed += 1

    print(f"\n[ValidAction] {done} processed, {skipped} skipped, {failed} failed")
    if no_object:
        print(f"[ValidAction] {no_object} skipped (no object name)")
    print(f"[ValidAction] Results: {valid_count} valid, {invalid_count} invalid")


if __name__ == "__main__":
    main()
