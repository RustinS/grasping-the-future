"""Search the Action100M annotations for video segments involving hand-object manipulation.

Two modes:
  Keyword mode (--object):      search for a specific object word in the gpt.action fields.
  Discovery mode (no --object): collect all GPT-annotated segments and ask Molmo2-8B (text only,
                                through vllm) for the manipulated object's name.

Usage:
    # Keyword search
    python -m gtf.extraction.action100m.find_segments --object cup
    python -m gtf.extraction.action100m.find_segments --object cup --all-fields

    # Discovery (needs a GPU and vllm)
    python -m gtf.extraction.action100m.find_segments --max-files 1
    python -m gtf.extraction.action100m.find_segments
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from gtf.utils import rprint as print

_TEXT_FIELDS = (
    "gpt_action_brief",
    "gpt_action_detailed",
    "plm_action",
    "gpt_summary_brief",
)

# -- Discovery mode constants --------------------------------------------------

MODEL_ID = "allenai/Molmo2-8B"

DISCOVERY_PROMPT = (
    "Action: {brief}\n"
    "Details: {detailed}\n\n"
    "What is the primary physical object held or manipulated by hand?\n"
    'Reply with just the object name (one to three words, e.g. "cup", "knife", "spray can") '
    'or "none" if no object is held.'
)

_NONE_RESPONSES = frozenset(
    {
        "none",
        "no object",
        "no manipulation",
        "n/a",
        "na",
        "nothing",
        "no tool",
        "no item",
        "not applicable",
        "not specified",
        "not clear",
        "unclear",
        "unknown",
    }
)


def search_node(node: dict, pattern: re.Pattern, all_fields: bool) -> list[str]:
    """Check if a node's text fields mention the target word. Return matching fields."""
    fields = []
    gpt = node.get("gpt")
    if gpt:
        action = gpt.get("action", {}) or {}
        fields += [
            ("gpt.action.brief", action.get("brief")),
            ("gpt.action.detailed", action.get("detailed")),
        ]
        if all_fields:
            summary = gpt.get("summary", {}) or {}
            fields += [
                ("gpt.summary.brief", summary.get("brief")),
                ("gpt.summary.detailed", summary.get("detailed")),
            ]
    if all_fields:
        fields += [
            ("plm_action", node.get("plm_action")),
            ("plm_caption", node.get("plm_caption")),
            ("llama3_caption", node.get("llama3_caption")),
        ]
    return [name for name, val in fields if val and pattern.search(val)]


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _dedup_and_merge(matches, video_uid, metadata):
    """Deduplicate matches to finest-grained non-overlapping segments, then merge consecutive ones.

    *matches* is a list of ``(node_dict, matched_fields_list)`` tuples.
    Returns a list of segment record dicts.
    """
    matches.sort(key=lambda x: (-x[0]["level"], x[0]["start"]))

    kept = []
    covered = []

    for node, matched_fields in matches:
        start, end = float(node["start"]), float(node["end"])
        if any((cs <= start and ce >= end) or (start <= cs and end >= ce) for cs, ce in covered):
            continue
        covered.append((start, end))

        gpt = node.get("gpt") or {}
        action = gpt.get("action", {}) or {}
        summary = gpt.get("summary", {}) or {}

        kept.append(
            {
                "video_uid": video_uid,
                "video_title": metadata.get("title", ""),
                "video_duration": metadata.get("duration"),
                "node_id": node["node_id"],
                "level": int(node["level"]),
                "start": start,
                "end": end,
                "duration_sec": round(end - start, 1),
                "time_range": f"{format_time(start)} - {format_time(end)}",
                "plm_action": node.get("plm_action"),
                "gpt_action_brief": action.get("brief"),
                "gpt_action_detailed": action.get("detailed"),
                "gpt_summary_brief": summary.get("brief"),
                "matched_fields": matched_fields,
            }
        )

    kept.sort(key=lambda x: x["start"])
    merged = []
    for seg in kept:
        if merged and seg["start"] <= merged[-1]["end"]:
            prev = merged[-1]
            prev["end"] = max(prev["end"], seg["end"])
            prev["duration_sec"] = round(prev["end"] - prev["start"], 1)
            prev["time_range"] = f"{format_time(prev['start'])} - {format_time(prev['end'])}"
            prev["level"] = max(prev["level"], seg["level"])
            prev["node_id"] = f"{prev['node_id']}+{seg['node_id']}"
            for field in _TEXT_FIELDS:
                if seg.get(field) and seg[field] != prev.get(field):
                    prev[field] = " | ".join(filter(None, [prev.get(field), seg[field]]))
            prev["matched_fields"] = list(dict.fromkeys(prev["matched_fields"] + seg["matched_fields"]))
        else:
            merged.append(seg)

    return merged


def process_file(parquet_path: Path, pattern: re.Pattern, all_fields: bool) -> list[dict]:
    """Search one parquet file. For each video, keep only finest-grained matching segments."""
    results = []
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    for _, row in df.iterrows():
        video_uid = row["video_uid"]
        metadata = row["metadata"]
        nodes = row["nodes"]

        matches = [(node, fields) for node in nodes if (fields := search_node(node, pattern, all_fields))]
        if not matches:
            continue

        results.extend(_dedup_and_merge(matches, video_uid, metadata))

    return results


def normalize_object_name(raw: str) -> str | None:
    """Clean up LLM response into a canonical object name, or None if not valid."""
    text = raw.strip().split("\n")[0].strip()
    text = text.strip("\"'`.,;:!?()[]{}").strip().lower()
    # Truncate at explanatory clauses
    for sep in (" because", " since", " as ", " (", " -"):
        idx = text.find(sep)
        if idx > 0:
            text = text[:idx].strip()
    # Strip leading articles
    for article in ("a ", "an ", "the "):
        if text.startswith(article):
            text = text[len(article) :]
    text = text.strip()
    if not text or text in _NONE_RESPONSES or text.startswith(("no ", "not ", "none")):
        return None
    words = text.split()
    if len(words) > 4:
        text = " ".join(words[:4])
    return text or None


def _load_vllm():
    """Load Molmo2-8B via vllm for text-only inference."""
    from vllm import LLM

    print(f"Loading {MODEL_ID} via vllm...")
    llm = LLM(model=MODEL_ID, trust_remote_code=True, limit_mm_per_prompt={"image": 0}, max_num_batched_tokens=32768)
    print("Model loaded")
    return llm


def process_file_discover(parquet_path, llm, sampling_params, min_level=3):
    """Discover manipulated objects in one parquet file using LLM extraction.

    For each video, collects all GPT-annotated nodes at level >= *min_level*,
    deduplicates, then runs the LLM on each segment to extract the object name.
    Returns only segments where a physical object was identified.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Phase 1: collect all GPT-annotated segments across videos
    all_segments = []
    for _, row in df.iterrows():
        video_uid = row["video_uid"]
        metadata = row["metadata"]
        nodes = row["nodes"]

        gpt_nodes = []
        for node in nodes:
            gpt = node.get("gpt")
            if not gpt:
                continue
            action = gpt.get("action", {}) or {}
            if not (action.get("brief") or action.get("detailed")):
                continue
            if int(node.get("level", 0)) < min_level:
                continue
            gpt_nodes.append((node, ["discovery"]))

        if not gpt_nodes:
            continue

        segments = _dedup_and_merge(gpt_nodes, video_uid, metadata)
        all_segments.extend(segments)

    if not all_segments:
        return []

    # Phase 2: build prompts and run LLM
    prompts = []
    for seg in all_segments:
        brief = seg.get("gpt_action_brief") or ""
        detailed = seg.get("gpt_action_detailed") or ""
        prompts.append(DISCOVERY_PROMPT.format(brief=brief, detailed=detailed))

    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(messages_list, sampling_params, use_tqdm=False)

    # Phase 3: normalize responses and filter
    results = []
    for seg, output in zip(all_segments, outputs):
        raw = output.outputs[0].text
        obj = normalize_object_name(raw)
        if obj is not None:
            seg["object"] = obj
            results.append(seg)

    return results


def main():
    parser = argparse.ArgumentParser(prog="python -m gtf.extraction.action100m.find_segments", description="Search Action100M for segments with hand-object manipulation")
    parser.add_argument("--object", default=None, help="Object word to search for (if omitted, uses LLM auto-discovery)")
    parser.add_argument("--data-dir", default="datasets/action100m", help="Path to parquet files directory")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--all-fields", action="store_true", help="Keyword mode: search all text fields, not just gpt.action")
    parser.add_argument("--max-files", type=int, default=0, help="Limit parquet files processed (0 = all, for testing)")
    parser.add_argument("--min-level", type=int, default=3, help="Minimum node level for discovery mode (default: 3)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    parquet_files = sorted(data_dir.glob("part-*.parquet"))
    if not parquet_files:
        sys.stderr.write(f"No parquet files found in {data_dir}\n")
        sys.exit(1)
    if args.max_files > 0:
        parquet_files = parquet_files[: args.max_files]

    all_results = []

    if args.object:
        # -- Keyword mode --
        word = args.object
        output_path = args.output or f"results/results_{word}.json"
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)

        print(f"Searching {len(parquet_files)} parquet files for '{word}' in gpt.action fields...")
        if args.all_fields:
            print("  (also searching plm_action, plm_caption, llama3_caption, gpt.summary)")

        for i, pf in enumerate(parquet_files):
            results = process_file(pf, pattern, args.all_fields)
            all_results.extend(results)
            if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
                print(f"  Processed {i + 1}/{len(parquet_files)} files, {len(all_results)} segments so far")

        for r in all_results:
            r["object"] = word
    else:
        # -- Discovery mode --
        from vllm import SamplingParams

        output_path = args.output or "results/results_discovered.json"
        llm = _load_vllm()
        sampling_params = SamplingParams(max_tokens=16, temperature=0.0)

        print(f"Discovery mode: processing {len(parquet_files)} parquet files (min_level={args.min_level})...")

        for i, pf in enumerate(parquet_files):
            results = process_file_discover(pf, llm, sampling_params, min_level=args.min_level)
            all_results.extend(results)
            if (i + 1) % 10 == 0 or i == len(parquet_files) - 1:
                n_objects = len(set(r["object"] for r in all_results))
                print(f"  Processed {i + 1}/{len(parquet_files)} files, {len(all_results)} segments ({n_objects} unique objects)")

        del llm

    all_results.sort(key=lambda r: (r["video_uid"], r["start"]))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    unique_videos = len(set(r["video_uid"] for r in all_results))
    unique_objects = len(set(r.get("object", "") for r in all_results))
    print(f"\nDone! {len(all_results)} segments across {unique_videos} videos, {unique_objects} unique objects.")
    print(f"Results saved to: {output_path}")

    print("\n--- First 15 examples ---")
    for r in all_results[:15]:
        print(f"\n  Video: {r['video_title'][:70]}")
        obj = r.get("object", "?")
        print(f"  Object: {obj}")
        print(f"  Segment: {r['time_range']} ({r['duration_sec']}s, level {r['level']})")
        print(f"  Action: {r['gpt_action_brief'] or r['plm_action'] or 'N/A'}")


if __name__ == "__main__":
    main()
