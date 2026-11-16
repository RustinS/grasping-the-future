"""Pipeline steps find, fetch, validate and cutcheck (with the shot children of multi-shot segments)."""

import json
import os
import re
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import torch
from gtf.utils import VIDEO_PROMPT_AGNOSTIC, get_segment_object, reject_action, rprint as print
from gtf.utils.action_validation import resume_action_validation_rejection
from gtf.extraction.action100m.stage_support import RESULTS_DIR, SEGMENTS_DIR, _write_json_atomic, _write_text_atomic


# ---------------------------------------------------------------------------
# find: search the Action100M annotations
# ---------------------------------------------------------------------------


def run_find_step(args):
    """Search Action100M parquet for segments — keyword or LLM discovery."""
    parquet_dir = Path(args.parquet_dir)
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {parquet_dir}")
        return None

    all_results = []

    if args.object:
        from gtf.extraction.action100m.find_segments import process_file

        pattern = re.compile(rf"\b{re.escape(args.object)}\b", re.IGNORECASE)
        print(f"Searching {len(parquet_files)} parquet files for '{args.object}'...")
        for pf in parquet_files:
            results = process_file(pf, pattern, all_fields=args.all_fields)
            all_results.extend(results)
        for r in all_results:
            r["object"] = args.object
        output_path = RESULTS_DIR / f"results_{args.object}_action.json"
    else:
        from gtf.extraction.action100m.find_segments import _load_vllm, process_file_discover
        from vllm import SamplingParams

        llm = _load_vllm()
        sampling_params = SamplingParams(max_tokens=16, temperature=0.0)
        min_level = getattr(args, "min_level", 3)
        print(f"Discovery mode: processing {len(parquet_files)} parquet files (min_level={min_level})...")
        for pf in parquet_files:
            results = process_file_discover(
                pf,
                llm,
                sampling_params,
                min_level=min_level,
            )
            all_results.extend(results)
        del llm
        output_path = RESULTS_DIR / "results_discovered.json"

    all_results.sort(key=lambda x: (x.get("video_uid", ""), x.get("start", 0)))
    print(f"Found {len(all_results)} segments")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {output_path}")

    return str(output_path)


# ---------------------------------------------------------------------------
# fetch: cut the segments out of the local source videos
# ---------------------------------------------------------------------------


def run_fetch_step(args, results_path=None):
    """Cut the found segments out of the source videos in --video-dir into SEGMENTS_DIR (+ manifest.json)."""
    from gtf.extraction.action100m.fetch_segments import cut_segments, filter_segments

    if args.video_dir is None or not Path(args.video_dir).is_dir():
        raise SystemExit(f"the fetch step needs --video-dir: a directory of source videos named by YouTube id (got {args.video_dir})")
    if results_path is None:
        fname = f"results_{args.object}_action.json" if args.object else "results_discovered.json"
        results_path = str(RESULTS_DIR / fname)
    if not Path(results_path).exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} segments from {results_path}")

    filter_args = SimpleNamespace(
        action=args.action_filter,
        top_actions=args.top_actions,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        shuffle=args.shuffle,
        seed=args.seed,
        unique_videos=args.unique_videos,
        limit_per_action=args.limit_per_action,
        limit=args.limit,
    )
    segments = filter_segments(data, filter_args)
    if not segments:
        print("No segments after filtering.")
        return

    succeeded, failed = cut_segments(segments, Path(args.video_dir), SEGMENTS_DIR)
    print(f"Fetch complete: {len(succeeded)} succeeded, {len(failed)} failed")


# ---------------------------------------------------------------------------
# validate: Molmo2 action validation
# ---------------------------------------------------------------------------


def run_validate_step(segment_names, args):
    """Run Molmo2-8B video-level action validation (object-agnostic prompt: is an object manipulated?).

    Writes valid_action/{object}/result.json and .done per segment. Rejections left half-written by an
    interrupted run are completed first.
    """
    resumed_rejections = 0
    for name in segment_names:
        targets = resume_action_validation_rejection(RESULTS_DIR / name)
        if targets is not None:
            resumed_rejections += 1
            print(f"  [Validate] {name}: completed interrupted action rejection for {', '.join(targets)}")
    if resumed_rejections:
        print(f"  [Validate] resumed {resumed_rejections} interrupted rejection transaction(s) before validity filtering")

    from gtf.extraction.frontend.run_valid_action import (
        detect_timestamps,
        load_model,
        parse_valid,
        run_video_query,
        run_video_query_batch,
    )

    def _validation_complete(name, obj):
        va_dir = RESULTS_DIR / name / "valid_action" / obj
        try:
            result = json.loads((va_dir / "result.json").read_text())
            marker = (va_dir / ".done").read_text()
        except (OSError, json.JSONDecodeError):
            return False
        return marker == "done\n" and isinstance(result, dict) and isinstance(result.get("valid"), bool) and result.get("object") == obj

    def _validate_pending(name):
        obj = args.object or get_segment_object(name)
        return bool(obj) and not _validation_complete(name, obj)

    if not args.force and not any(_validate_pending(n) for n in segment_names):
        print(f"  [Validate] all {len(segment_names)} segments already validated — model load skipped")
        return
    done = skipped = failed = invalid = no_object = 0
    work = []
    for vi, name in enumerate(segment_names):
        obj = args.object or get_segment_object(name)
        if not obj:
            # Fall back to metadata.json (written when the orchestrator stages the segment)
            meta_path = RESULTS_DIR / name / "metadata.json"
            if meta_path.exists():
                try:
                    obj = json.loads(meta_path.read_text()).get("object")
                except (json.JSONDecodeError, OSError):
                    pass
        if not obj:
            no_object += 1
            continue

        seg_dir = RESULTS_DIR / name
        va_dir = seg_dir / "valid_action" / obj
        done_marker = va_dir / ".done"

        if not args.force and _validation_complete(name, obj):
            skipped += 1
            continue

        video_path = seg_dir / "video.mp4"
        if not video_path.exists():
            failed += 1
            continue

        work.append(
            {
                "index": vi,
                "name": name,
                "object": obj,
                "video_path": video_path,
                "va_dir": va_dir,
                "done_marker": done_marker,
            }
        )

    if not work:
        print(f"  [Validate] 0 valid, 0 invalid, {skipped} skipped, {failed} failed")
        if no_object:
            print(f"  [Validate] {no_object} skipped (no object name)")
        return

    processor, model = load_model()
    video_batch_size = max(1, int(os.environ.get("VALID_ACTION_VIDEO_BATCH_SIZE", "4")))
    for batch_start in range(0, len(work), video_batch_size):
        batch = work[batch_start : batch_start + video_batch_size]
        for item in batch:
            item["done_marker"].unlink(missing_ok=True)

        responses_by_name = {}
        try:
            responses = run_video_query_batch(
                processor,
                model,
                [str(item["video_path"]) for item in batch],
                [VIDEO_PROMPT_AGNOSTIC] * len(batch),
            )
            if len(responses) != len(batch):
                raise RuntimeError("batched validation returned a different response count")
            responses_by_name = {item["name"]: response for item, response in zip(batch, responses)}
        except Exception as batch_error:
            # One malformed or long clip must not fail the others: retry one clip at a time.
            print(f"  [Validate] batch failed ({batch_error}); retrying {len(batch)} clip(s) individually")
            torch.cuda.empty_cache()
            for item in batch:
                try:
                    responses_by_name[item["name"]] = run_video_query(
                        processor,
                        model,
                        str(item["video_path"]),
                        VIDEO_PROMPT_AGNOSTIC,
                    )
                except Exception as item_error:
                    print(f"    ERROR {item['name']}: {item_error}")

        for item in batch:
            vi = item["index"]
            name = item["name"]
            obj = item["object"]
            video_path = item["video_path"]
            va_dir = item["va_dir"]
            done_marker = item["done_marker"]
            raw_response = responses_by_name.get(name)
            print(f"  [{vi + 1}/{len(segment_names)}] Validate: {name} (object={obj})")
            if raw_response is None:
                failed += 1
                continue
            try:
                valid = parse_valid(raw_response)

                timestamps = None
                if valid:
                    timestamps = detect_timestamps(processor, model, str(video_path))

                va_dir.mkdir(parents=True, exist_ok=True)
                result = {
                    "valid": valid,
                    "object": obj,
                    "validation_mode": "agnostic",
                    "prompt": VIDEO_PROMPT_AGNOSTIC,
                    "raw_response": raw_response,
                }
                if timestamps:
                    result["timestamps"] = timestamps
                _write_json_atomic(va_dir / "result.json", result)
                _write_text_atomic(done_marker, "done\n")

                status = "VALID" if valid else "INVALID"
                print(f"    {status}: {raw_response[:80]}")
                if valid:
                    done += 1
                else:
                    invalid += 1

            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                failed += 1

    del processor, model
    torch.cuda.empty_cache()
    print(f"  [Validate] {done} valid, {invalid} invalid, {skipped} skipped, {failed} failed")
    if no_object:
        print(f"  [Validate] {no_object} skipped (no object name)")


# ---------------------------------------------------------------------------
# cutcheck: truncate at the first shot boundary, or reject
# ---------------------------------------------------------------------------


def _completed_cutcheck_record(seg_dir, marker):
    """The cut_check.json record, or None when it is malformed or the videos it describes (the truncated
    video.mp4, video_full.mp4, the shot children) do not match it."""

    from gtf.extraction.action100m.fetch_segments import probe_duration

    try:
        record = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("action") not in {
        "clean",
        "truncate",
        "reject",
    }:
        return None
    if record["action"] != "reject":
        video = seg_dir / "video.mp4"
        if not video.is_file() or video.stat().st_size <= 0:
            return None
    if record["action"] == "truncate":
        full = seg_dir / "video_full.mp4"
        if not full.is_file() or full.stat().st_size <= 0:
            return None
        try:
            kept_frames = int(record["kept_frames"])
            total_frames = int(record["n_frames"])
            fps = float(record["fps"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if kept_frames < 1 or total_frames <= kept_frames or fps <= 0:
            return None
        expected_cut = kept_frames / fps
        expected_full = total_frames / fps
        actual_cut = probe_duration(video)
        actual_full = probe_duration(full)
        cut_ok = actual_cut is not None and abs(actual_cut - expected_cut) <= max(0.5, 0.2 * expected_cut)
        full_ok = actual_full is not None and abs(actual_full - expected_full) <= max(0.5, 0.2 * expected_full)
        if not cut_ok or not full_ok:
            return None
    shot_children = record.get("shot_children", [])
    if not isinstance(shot_children, list):
        return None
    for child_name in shot_children:
        if not isinstance(child_name, str) or not child_name.startswith(f"{seg_dir.name}__shot"):
            return None
        child_dir = RESULTS_DIR / child_name
        try:
            child_meta = json.loads((child_dir / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None
        child_video = child_dir / "video.mp4"
        multishot = child_meta.get("multishot")
        if not child_video.is_file() or child_video.stat().st_size <= 0 or not isinstance(multishot, dict) or multishot.get("parent") != seg_dir.name:
            return None
    return record


def run_cutcheck_step(segment_names, args):
    """Detect shot boundaries before the expensive extraction stages.

    In-the-wild clips are edited videos with frequent cuts; after a mid-segment cut the
    per-frame stages keep succeeding on unrelated content. Policy: truncate video.mp4 to
    the pre-cut span when it keeps >= cut_min_keep_s of footage (the original is kept as
    video_full.mp4), otherwise reject the segment through the valid_action mechanism so
    later steps skip it. quality_gate's scene_cut gate is a second check.

    Multi-shot harvest (unless no_multishot): every later shot of at least the same length
    becomes a child segment `{seg}__shotN` (video re-encoded from the shot span, parent
    metadata inherited), also for rejected segments whose first shot is too short. The
    children are returned so main() can validate them and add them to the run.
    """
    from gtf.extraction.frontend.detect_cuts import detect_scene_cuts

    def process_one(item):
        vi, name = item
        done = skipped = failed = truncated = rejected = 0
        children = []
        seg_dir = RESULTS_DIR / name
        video = seg_dir / "video.mp4"
        full = seg_dir / "video_full.mp4"
        marker = seg_dir / "cut_check.json"
        if not args.force and (record := _completed_cutcheck_record(seg_dir, marker)) is not None:
            children.extend(record.get("shot_children", []))
            skipped += 1
            return done, skipped, failed, truncated, rejected, children
        source_video = full if full.exists() else video
        if not source_video.exists():
            failed += 1
            return done, skipped, failed, truncated, rejected, children
        try:
            marker.unlink(missing_ok=True)
            cuts, n_frames, fps = detect_scene_cuts(source_video)
            rec = {"cuts": cuts, "n_frames": n_frames, "fps": round(fps, 3)}
            if not cuts:
                rec["action"] = "clean"
                done += 1
            elif cuts[0] / fps >= args.cut_min_keep_s:
                if not full.exists():
                    video.rename(full)
                _reencode_video_prefix(full, video, cuts[0], fps)
                rec["action"] = "truncate"
                rec["kept_frames"] = cuts[0]
                print(f"  [{vi + 1}/{len(segment_names)}] {name}: cut at frame {cuts[0]} ({cuts[0] / fps:.1f}s) — truncated")
                truncated += 1
            else:
                rec["action"] = "reject"
                reject_action(
                    name,
                    f"scene_cut at frame {cuts[0]} ({cuts[0] / fps:.1f}s < {args.cut_min_keep_s}s usable)",
                )
                print(f"  [{vi + 1}/{len(segment_names)}] {name}: cut at {cuts[0] / fps:.1f}s — REJECTED (too little pre-cut footage)")
                rejected += 1
            if cuts and not getattr(args, "no_multishot", False):
                shot_children = _create_shot_children(
                    name,
                    seg_dir,
                    cuts,
                    n_frames,
                    fps,
                    args.cut_min_keep_s,
                )
                if shot_children:
                    rec["shot_children"] = shot_children
                    children += shot_children
                    print(f"      harvested {len(shot_children)} extra shot(s): {', '.join(shot_children)}")
            _write_json_atomic(marker, rec)
        except Exception as e:
            print(f"  [{vi + 1}/{len(segment_names)}] {name}: ERROR {e}")
            traceback.print_exc()
            failed += 1
        return done, skipped, failed, truncated, rejected, children

    workers = min(max(1, int(getattr(args, "prepass_workers", 5))), len(segment_names))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(process_one, enumerate(segment_names)))
    else:
        rows = [process_one(item) for item in enumerate(segment_names)]
    done = sum(row[0] for row in rows)
    skipped = sum(row[1] for row in rows)
    failed = sum(row[2] for row in rows)
    truncated = sum(row[3] for row in rows)
    rejected = sum(row[4] for row in rows)
    # executor.map keeps the input order, so the children come out in the serial order.
    children = [child for row in rows for child in row[5]]
    print(f"  [Cut check] {done} clean, {truncated} truncated, {rejected} rejected, {skipped} skipped, {failed} failed, {len(children)} shot children")
    return children


def _create_shot_children(
    name,
    seg_dir,
    cuts,
    n_frames,
    fps,
    min_keep_s,
    max_children=4,
):
    """Harvest post-first-cut shots as child segments `{name}__shotN` (idempotent; existing valid children are reused).

    Each cut index is the first frame of a new shot; spans run cut_j..next boundary.
    The re-encode starts at cut_j + 1: the boundary frame itself can carry transition
    content that re-detects as "cut at 0.0s" and would reject the child.
    """
    src = seg_dir / "video_full.mp4"
    if not src.exists():
        src = seg_dir / "video.mp4"
    bounds = list(cuts) + [n_frames]
    created = []
    for j in range(len(cuts)):
        a, b = bounds[j], bounds[j + 1]
        if (b - a) / fps < min_keep_s:
            continue
        child = f"{name}__shot{j + 1}"
        child_dir = RESULTS_DIR / child
        child_video = child_dir / "video.mp4"
        child_metadata = child_dir / "metadata.json"
        expected_multishot = {
            "parent": name,
            "shot_index": j + 1,
            "frames": [int(a), int(b)],
        }
        try:
            existing_meta = json.loads(child_metadata.read_text())
        except (OSError, json.JSONDecodeError):
            existing_meta = None
        if child_video.is_file() and child_video.stat().st_size > 0 and isinstance(existing_meta, dict) and existing_meta.get("multishot") == expected_multishot:
            from gtf.extraction.action100m.fetch_segments import (
                video_decord_batch_is_safe,
            )

            if video_decord_batch_is_safe(child_video):
                created.append(child)
                if len(created) >= max_children:
                    break
                continue
        child_dir.mkdir(parents=True, exist_ok=True)
        _reencode_video_span(src, child_video, a + 1, b, fps)
        meta_path = seg_dir / "metadata.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta["file"] = f"{child}.mp4"
        meta["multishot"] = expected_multishot
        _write_json_atomic(child_metadata, meta)
        created.append(child)
        if len(created) >= max_children:
            break
    return created


def _reencode_video_span(src, dst, start_frame, end_frame, fps):
    """Re-encode frames [start_frame, end_frame) of src into dst (sequential decode —
    frame-exact, no keyframe-seek boundary bleed)."""
    import av
    import cv2
    from gtf.extraction.action100m.fetch_segments import video_decord_batch_is_safe

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_text = tempfile.mkstemp(prefix=f".{dst.name}.tmp-", suffix=".mp4", dir=dst.parent)
    os.close(fd)
    temporary = Path(temporary_text)
    cap = None
    container = None
    try:
        normalize_for_decord = not video_decord_batch_is_safe(Path(src))
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"cannot decode source video: {src}")
        container = av.open(str(temporary), mode="w")
        stream = None
        for fi in range(end_frame):
            ret, bgr = cap.read()
            if not ret:
                break
            if fi < start_frame:
                continue
            if normalize_for_decord:
                # The same repair as fetch_segments.extract_segment: a source Decord cannot batch-decode
                # is widened by two pixels (even, aspect-preserving height).
                source_h, source_w = bgr.shape[:2]
                target_w = source_w + 2
                target_h = max(
                    2,
                    2 * round((source_h * target_w / source_w) / 2),
                )
                bgr = cv2.resize(
                    bgr,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            if stream is None:
                stream = container.add_stream("h264", rate=round(fps))
                stream.width, stream.height = bgr.shape[1], bgr.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18"}
            frame = av.VideoFrame.from_ndarray(bgr[:, :, ::-1], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        if stream is None:
            raise RuntimeError(f"no frames decoded for requested span {start_frame}:{end_frame}")
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        cap.release()
        cap = None
        if temporary.stat().st_size <= 0:
            raise RuntimeError("video encoder produced an empty output")
        if not video_decord_batch_is_safe(temporary):
            raise RuntimeError(f"shot-child encode is not Decord batch-safe: {src}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, dst)
    finally:
        if container is not None:
            container.close()
        if cap is not None:
            cap.release()
        temporary.unlink(missing_ok=True)


def _reencode_video_prefix(src, dst, n_frames, fps):
    """Re-encode the first n_frames of src into dst (sequential decode — frame-exact)."""
    import av
    import cv2

    cap = cv2.VideoCapture(str(src))
    container = av.open(str(dst), mode="w")
    stream = None
    for fi in range(n_frames):
        ret, bgr = cap.read()
        if not ret:
            break
        if stream is None:
            stream = container.add_stream("h264", rate=round(fps))
            stream.width, stream.height = bgr.shape[1], bgr.shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18"}
        frame = av.VideoFrame.from_ndarray(bgr[:, :, ::-1], format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    cap.release()
    if stream is not None:
        for packet in stream.encode():
            container.mux(packet)
    container.close()
