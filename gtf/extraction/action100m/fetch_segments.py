"""Cut Action100M segments out of locally stored source videos.

Reads a results JSON (from find_segments.py, the find step), looks up each referenced YouTube video in a local
directory (one file per video, named by its YouTube id: {video_uid}.mp4, .mkv or .webm), and cuts
the segment's time range with ffmpeg, normalizing the codec to h264 (the decoder every later stage
uses only reads h264 reliably). Obtaining the source videos is up to the user.

Writes the clips and manifest.json (succeeded / failed records) to the output directory, which the
extraction orchestrator reads.

Usage:
    python -m gtf.extraction.action100m.fetch_segments --input results/results_cup_action.json --video-dir /data/yt --limit 10
    python -m gtf.extraction.action100m.fetch_segments --input results/results_cup_action.json --video-dir /data/yt --action "Lift cup" --limit-per-action 10
    python -m gtf.extraction.action100m.fetch_segments --input results/results_cup_action.json --video-dir /data/yt --shuffle --unique-videos --limit 20
"""

from __future__ import annotations

import argparse
import builtins
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


from gtf.utils import SEGMENTS_DIR, rprint as print

SUPPORTED_EXTENSIONS = (".mp4", ".mkv", ".webm")


class MediaInfrastructureError(RuntimeError):
    """A required local media tool could not complete reliably."""


class FrameRosterProbeError(MediaInfrastructureError):
    """The decoded-frame roster could not be measured reliably."""


class DecordProbeError(MediaInfrastructureError):
    """The Decord batch decode could not be probed."""


def resolve_local_video(video_dir: Path, uid: str) -> Path | None:
    """The source video of one YouTube id in video_dir ({uid}.mp4 preferred over .mkv / .webm), or None."""
    for ext in SUPPORTED_EXTENSIONS:
        path = Path(video_dir) / f"{uid}{ext}"
        if path.is_file():
            return path
    return None


def probe_duration(
    path: Path, *, strict_infrastructure: bool = False
) -> float | None:
    """Media duration in seconds via ffprobe, or None; with strict_infrastructure, raises MediaInfrastructureError
    when ffprobe itself cannot run."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return float(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        if strict_infrastructure:
            raise MediaInfrastructureError(
                f"ffprobe could not run to measure duration for {path}: "
                f"{type(exc).__name__}"
            ) from exc
        return None
    except ValueError:
        return None


# Decord (Molmo2 video mode and every later stage) reliably decodes only h264: VP9 / AV1 / HEVC sources
# stream-copied into .mp4 fail its threaded decoder ("Error sending packet", -11), so clips are normalized to h264.
DECORD_SAFE_CODECS = {"h264"}


def video_codec(path: Path) -> str | None:
    """Video stream codec_name via ffprobe (e.g. 'h264', 'vp9', 'av1'), or None."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip().lower()
    except subprocess.TimeoutExpired:
        pass
    return None


def probe_video_frame_roster(path: Path) -> dict[str, int | None] | None:
    """Frame counts of a clip's video stream: header (nb_frames), packets and decoded frames.

    ``decord.VideoReader.__len__`` follows the container's frame index for these MP4 clips. A stream copy that
    starts inside an open GOP can leave undecodable leading pictures: the index (and Decord) then report more
    frames than a decoder produces, and Decord fills the gap by repeating its last good frame, which shifts the
    timestamps of the frames it returns.

    ``-count_frames`` decodes the stream without producing RGB frames. The packet count stands in for Decord's
    indexed frame count; ``nb_frames`` is compared too when the container stores it. Returns None when the probe
    fails or its output is malformed, so the caller treats the clip as unchecked.
    """

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-count_packets",
                "-show_entries",
                "stream=nb_frames,nb_read_frames,nb_read_packets",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout)
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not isinstance(streams, list):
            return None
        if not streams:
            # ffprobe ran, but the file has no video stream: a property of the
            # content, not a tool failure. Return counts that fail the
            # consistency check, so the caller rejects the clip instead of
            # retrying it forever.
            return {
                "header_frames": None,
                "packet_frames": None,
                "decoded_frames": None,
            }
        if len(streams) != 1:
            return None
        stream = streams[0]
        if not isinstance(stream, dict):
            return None

        def positive_int(field: str, *, required: bool) -> int | None:
            raw = stream.get(field)
            if raw in (None, "", "N/A") and not required:
                return None
            if isinstance(raw, bool):
                raise ValueError(field)
            value = int(raw)
            if value < 1:
                raise ValueError(field)
            return value

        return {
            "header_frames": positive_int("nb_frames", required=False),
            "packet_frames": positive_int("nb_read_packets", required=True),
            "decoded_frames": positive_int("nb_read_frames", required=True),
        }
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ):
        return None


def frame_roster_is_consistent(roster: object) -> bool:
    """Whether the packet count and the header frame count (when present) equal the decoded frame count."""

    if not isinstance(roster, dict):
        return False
    header = roster.get("header_frames")
    packets = roster.get("packet_frames")
    decoded = roster.get("decoded_frames")
    if (
        isinstance(header, bool)
        or isinstance(packets, bool)
        or isinstance(decoded, bool)
        or not isinstance(packets, int)
        or not isinstance(decoded, int)
        or packets < 1
        or decoded < 1
        or (header is not None and (not isinstance(header, int) or header < 1))
    ):
        return False
    return packets == decoded and (header is None or header == decoded)


def video_frame_roster_is_consistent(path: Path) -> bool:
    """Whether one clip's frame counts agree; raises FrameRosterProbeError when ffprobe cannot measure them."""

    roster = probe_video_frame_roster(path)
    if roster is None:
        raise FrameRosterProbeError(
            f"ffprobe could not measure the decoded-frame roster for {path}"
        )
    return frame_roster_is_consistent(roster)


def video_decord_batch_is_safe(path: Path) -> bool:
    """Whether Decord can batch-decode RGB frames (first, middle and last) of the clip.

    Some otherwise valid H.264 clips (for example with a 144 px coded width) decode as four-channel frames in
    Decord while ``get_batch`` has allocated an RGB buffer. ffprobe and sequential ffmpeg decoding report those
    files as healthy, but Molmo2 and later stages fail with ``DECORDArrayCopyFromTo``, so a clip is probed the way
    those stages read it before it is written or reused.

    Raises DecordProbeError when Decord cannot be imported or cannot open the file for a system reason; returns
    False for a clip it opens but cannot batch-decode as RGB.
    """

    try:
        from decord import VideoReader, cpu
    except (ImportError, OSError) as exc:
        raise DecordProbeError(
            "decord is unavailable while validating the extraction media "
            f"contract for {path}"
        ) from exc

    try:
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    except (ImportError, OSError) as exc:
        raise DecordProbeError(
            f"decord could not open {path}: {type(exc).__name__}"
        ) from exc
    except Exception:
        return False

    try:
        total_frames = len(reader)
        if total_frames < 1:
            return False
        indices = sorted({0, total_frames // 2, total_frames - 1})
        frames = reader.get_batch(indices)
        shape = tuple(frames.shape)
        return (
            len(shape) == 4
            and shape[0] == len(indices)
            and shape[1] > 0
            and shape[2] > 0
            and shape[3] == 3
        )
    except Exception:
        # A decoder / content error: extract_segment repairs it with the
        # width-changing re-encode. A missing Decord was raised above and is
        # never reported as bad content.
        return False
    finally:
        del reader


def extract_segment(input_path: Path, output_path: Path, start: float, end: float) -> tuple[bool, str]:
    """Extract a time segment with ffmpeg, normalizing the codec to h264.

    Stream-copies only when the source video is already h264 (decodable by decord);
    otherwise re-encodes to h264, since VP9 / AV1 / HEVC streams copied into .mp4 fail
    decord's threaded decoder downstream. Returns (success, extraction_mode).
    """
    base_args = ["-y", "-ss", str(start), "-to", str(end), "-i", str(input_path)]
    suffix_args = ["-avoid_negative_ts", "make_zero", str(output_path)]

    expected_duration = end - start

    def output_is_structurally_valid() -> bool:
        actual_duration = probe_duration(
            output_path, strict_infrastructure=True
        )
        if actual_duration is None or abs(
            actual_duration - expected_duration
        ) > max(0.5, 0.2 * expected_duration):
            return False
        try:
            return video_frame_roster_is_consistent(output_path)
        except FrameRosterProbeError:
            output_path.unlink(missing_ok=True)
            raise

    def output_contract() -> tuple[bool, bool]:
        """Return (fully_valid, needs_decord_geometry_normalization)."""

        if not output_path.exists() or not output_is_structurally_valid():
            return False, False
        decord_safe = video_decord_batch_is_safe(output_path)
        return decord_safe, not decord_safe

    def run_ffmpeg(arguments: list[str], timeout: float):
        try:
            return subprocess.run(
                arguments,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output_path.unlink(missing_ok=True)
            raise MediaInfrastructureError(
                f"ffmpeg could not complete extraction for {output_path}: "
                f"{type(exc).__name__}"
            ) from exc

    # Stream copy (fast) only for h264 sources. A zero exit code is not sufficient:
    # input-side seeking can begin on an earlier GOP boundary and produce a clip with
    # the wrong duration, which then falls through to the frame-accurate re-encode.
    needs_decord_geometry_normalization = False
    if video_codec(input_path) in DECORD_SAFE_CODECS:
        result = run_ffmpeg(
            ["ffmpeg"] + base_args + ["-c", "copy"] + suffix_args,
            timeout=300,
        )
        if result is not None and result.returncode == 0 and output_path.exists():
            valid, needs_decord_geometry_normalization = output_contract()
            if valid:
                return True, "copy"
        output_path.unlink(missing_ok=True)

    # Re-encode to h264 (non-h264 source, failed copy, or an imprecise copy), checked
    # like the copy.
    if not needs_decord_geometry_normalization:
        result = run_ffmpeg(
            ["ffmpeg"]
            + base_args
            + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"]
            + suffix_args,
            timeout=600,
        )
        if result is not None and result.returncode == 0 and output_path.exists():
            valid, needs_decord_geometry_normalization = output_contract()
            if valid:
                return True, "reencode"
        output_path.unlink(missing_ok=True)

    if needs_decord_geometry_normalization:
        # A same-size transcode does not repair Decord's RGB/RGBA allocation
        # mismatch. Widen by two pixels with an even, aspect-preserving height;
        # Lanczos + CRF 18 keeps the image quality. This slower path runs only
        # after the Decord probe has failed.
        result = run_ffmpeg(
            ["ffmpeg"]
            + base_args
            + [
                "-vf",
                "scale=iw+2:-2:flags=lanczos",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
            + suffix_args,
            timeout=600,
        )
        if result is not None and result.returncode == 0 and output_path.exists():
            valid, _still_unsafe = output_contract()
            if valid:
                return True, "reencode_decord_safe_scale"

    output_path.unlink(missing_ok=True)
    return False, ""


def filter_segments(data: list[dict], args) -> list[dict]:
    """Apply the filters of args in this order: action, top actions, duration, (shuffle), unique videos, per-action
    limit, global limit; print how many segments each removed."""
    total = len(data)
    excluded = Counter()

    if args.action:
        action_lower = args.action.lower()
        before = len(data)
        data = [seg for seg in data if action_lower in (seg.get("gpt_action_brief") or "").lower()]
        excluded["action filter"] = before - len(data)

    if args.top_actions is not None:
        action_counts = Counter(seg.get("gpt_action_brief") or "unknown" for seg in data)
        top = {action for action, _ in action_counts.most_common(args.top_actions)}
        before = len(data)
        data = [seg for seg in data if (seg.get("gpt_action_brief") or "unknown") in top]
        excluded["top-actions"] = before - len(data)

    before = len(data)
    data = [seg for seg in data if args.min_duration <= seg["end"] - seg["start"] and (args.max_duration is None or seg["end"] - seg["start"] <= args.max_duration)]
    if before - len(data):
        excluded["duration"] = before - len(data)

    if args.shuffle:
        random.Random(args.seed).shuffle(data)

    if args.unique_videos:
        seen = set()
        deduped = []
        for seg in data:
            if seg["video_uid"] not in seen:
                seen.add(seg["video_uid"])
                deduped.append(seg)
        excluded["unique-videos dedup"] = len(data) - len(deduped)
        data = deduped

    if args.limit_per_action is not None:
        counts = defaultdict(int)
        capped = []
        for seg in data:
            action = seg.get("gpt_action_brief") or "unknown"
            if counts[action] < args.limit_per_action:
                counts[action] += 1
                capped.append(seg)
        excluded["limit-per-action"] = len(data) - len(capped)
        data = capped

    if args.limit is not None and len(data) > args.limit:
        excluded["global limit"] = len(data) - args.limit
        data = data[: args.limit]

    detail = ", ".join(f"{v} by {k}" for k, v in excluded.items() if v) or "none"
    print(f"{len(data)} segments after filtering ({total - len(data)} excluded: {detail})")
    return data


def _make_success_record(seg: dict, fname: str, source_path: Path, start: float, end: float, mode: str) -> dict:
    rec = {
        "file": fname,
        "video_uid": seg["video_uid"],
        "video_title": seg.get("video_title", ""),
        "start": start,
        "end": end,
        "duration_sec": round(end - start, 1),
        "action": seg.get("gpt_action_brief", ""),
        "action_detailed": seg.get("gpt_action_detailed", ""),
        "source_uri": str(source_path),
        "source_format": Path(source_path).suffix.lstrip(".").lower(),
        "extraction_mode": mode,
    }
    if seg.get("object"):
        rec["object"] = seg["object"]
    return rec


def _fail_group(group_segs: list[dict], reason: str, detail: str) -> list[dict]:
    return [{"video_uid": seg["video_uid"], "start": seg["start"], "end": seg["end"], "reason": reason, "detail": detail} for seg in group_segs]


def segment_filenames(segments: list[dict]) -> dict[int, str]:
    """Clip file name of every segment ({uid}_{start}-{end}s.mp4; the node id is appended when two segments collide)."""
    counts = Counter(f"{seg['video_uid']}_{seg['start']:.0f}-{seg['end']:.0f}s.mp4" for seg in segments)
    names = {}
    for seg in segments:
        base = f"{seg['video_uid']}_{seg['start']:.0f}-{seg['end']:.0f}s.mp4"
        names[id(seg)] = f"{seg['video_uid']}_{seg['start']:.0f}-{seg['end']:.0f}s_{seg.get('node_id', 'unknown')[:8]}.mp4" if counts[base] > 1 else base
    return names


def cut_segments(segments: list[dict], video_dir: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Cut every segment from its local source video into output_dir and write output_dir/manifest.json.

    An existing clip of the right duration is kept. Returns (succeeded, failed) manifest records.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for seg in segments:
        groups[seg["video_uid"]].append(seg)
    print(f"Processing {len(segments)} segments from {len(groups)} unique videos\n")
    filenames = segment_filenames(segments)
    succeeded, failed = [], []

    for group_idx, (video_uid, group_segs) in enumerate(groups.items(), 1):
        n = len(group_segs)
        print(f"[{group_idx}/{len(groups)}] Video {video_uid} ({n} segment{'s' if n > 1 else ''})")

        source = resolve_local_video(video_dir, video_uid)
        if source is None:
            print(f"  NOT FOUND in {video_dir}")
            failed.extend(_fail_group(group_segs, "not_found", f"no {video_uid}{{{','.join(SUPPORTED_EXTENSIONS)}}} in {video_dir}"))
            continue

        media_duration = probe_duration(source)
        if media_duration is None:
            print("  PROBE FAILED: cannot read the media duration")
            failed.extend(_fail_group(group_segs, "probe_error", "ffprobe could not determine media duration"))
            continue
        print(f"  Source: {source.name}, {media_duration:.1f}s")

        for seg in group_segs:
            start, end = seg["start"], seg["end"]
            fname = filenames[id(seg)]
            output_path = output_dir / fname
            clamped_end = min(end, media_duration)

            if start >= media_duration or start >= clamped_end:
                print(f"  {fname}: SKIP (timestamp invalid: start={start:.1f}, end={clamped_end:.1f}, media={media_duration:.1f})")
                failed.append({"video_uid": video_uid, "start": start, "end": end, "reason": "timestamp_invalid", "detail": f"start={start:.1f} >= media_duration={media_duration:.1f}"})
                continue

            expected_dur = clamped_end - start
            if output_path.exists():
                actual_dur = probe_duration(output_path)
                if actual_dur is not None and abs(actual_dur - expected_dur) <= max(0.5, 0.2 * expected_dur):
                    print(f"  {fname}: skipping, valid ({expected_dur:.1f}s)")
                    succeeded.append(_make_success_record(seg, fname, source, start, clamped_end, "resumed"))
                    continue
                print(f"  {fname}: existing file corrupt, re-extracting")
                output_path.unlink()

            ok, mode = extract_segment(source, output_path, start, clamped_end)
            if ok:
                print(f"  {fname}: OK ({expected_dur:.1f}s, {mode})")
                succeeded.append(_make_success_record(seg, fname, source, start, clamped_end, mode))
            else:
                print(f"  {fname}: FFMPEG FAILED")
                output_path.unlink(missing_ok=True)
                failed.append({"video_uid": video_uid, "start": start, "end": end, "reason": "ffmpeg_error", "detail": "Both copy and re-encode extraction failed"})
        print()

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"succeeded": succeeded, "failed": failed}, f, indent=2)
    return succeeded, failed


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.action100m.fetch_segments", description="Cut Action100M segments out of local source videos")
    parser.add_argument("--input", default="results/results_cup_action.json", help="Path to results JSON")
    parser.add_argument("--video-dir", type=Path, required=True, help="Directory of source videos named by YouTube id ({video_uid}.mp4/.mkv/.webm)")
    parser.add_argument("--output-dir", default=SEGMENTS_DIR, help=f"Output directory (default: {SEGMENTS_DIR})")
    parser.add_argument("--limit", type=int, default=None, help="Global max segments to process")
    parser.add_argument("--action", default=None, help="Filter by gpt_action_brief (case-insensitive substring)")
    parser.add_argument("--top-actions", type=int, default=None, help="Keep only segments from the N most frequent actions")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Min segment duration in seconds (default: 1.0)")
    parser.add_argument("--max-duration", type=float, default=None, help="Max segment duration in seconds")
    parser.add_argument("--unique-videos", action="store_true", help="Keep only first segment per video_uid")
    parser.add_argument("--limit-per-action", type=int, default=None, help="Max segments per unique gpt_action_brief value")
    parser.add_argument("--shuffle", action="store_true", help="Randomize segment order before dedup/limit")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible --shuffle")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        builtins.print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.video_dir.is_dir():
        builtins.print(f"ERROR: video directory not found: {args.video_dir}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} segments from {args.input}")

    segments = filter_segments(data, args)
    if not segments:
        print("No segments to process after filtering.")
        sys.exit(0)

    output_dir = Path(args.output_dir)
    succeeded, failed = cut_segments(segments, args.video_dir, output_dir)
    total = len(succeeded) + len(failed)
    print(f"Done! {len(succeeded)} succeeded, {len(failed)} failed out of {total} segments")
    print(f"Output: {output_dir}/")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
