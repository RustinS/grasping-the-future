#!/usr/bin/env python3
"""Re-grounding verifier (gate 21): fresh VLM evidence against the tracked object.

The other quality gates catch extraction corruption, but a segment can track the wrong object while every stage is
self-consistent about it (a receptacle instead of the manipulated object, a body part, a fixed structure, worn clothing,
a hand carrying nothing). This module re-grounds the tracked object with fresh Molmo2 evidence instead of trusting the
propagated mask and chain.

At K spread frames inside the exported interaction window, Molmo2 is asked
    held   "point to the object held in the hand"   (no noun bias)
    named  "point to the {noun}"                    (no "in hand" wording, so it does not presuppose a grasp)
    all    "point to every {noun} in the image"     (covers every instance)
    vqa    "Is a person's hand touching or holding the {noun}?"  (greedy yes/no)
and every point is scored against the segment's own SAM 3 mask at that frame (distance transform: inside the
DILATE_FRAC-dilated mask, and the distance in % of the image diagonal).

Gate axes:
    reground_held_refusal  fraction of "held" generations the VLM refused (no point): nothing is actually held
                           (fixture, body part, demonstration without contact, machine-driven motion)
    reground_held_dist     median %diag distance of the "held" points from the tracked mask: something is held, but it
                           is not the tracked object (worn objects, a grasp on something else)
    reground_held_anyin    fraction of frames where any "held" point lands inside the tracked mask; recorded only, not
                           gated (a genuine segment can track a non-held manipulandum, e.g. poured contents)
Gated axes are None when unmeasurable, and the gate then defers (the quality_gate convention). The named / all / vqa
channels are recorded as evidence but are not gate axes: instance ambiguity (two cups) and noun / tracked-object mismatch
(noun = contents, mask = container) put genuine segments at zero containment. Not caught: a receptacle tracked as the
primary object (the hands work inside its silhouette), fused multi-instance masks, deformable objects tracked as rigid.

Flow: the orchestrator's `reground` step (after quality_gate, GPU) collects evidence only for gate-passing segments
into {seg}/reground/reground.json and re-evaluates the quality gate, so gate 21 becomes active. Without evidence every
axis is None and the gate defers, so CPU-only gating works without it.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from gtf.utils import POINTING_PROMPT_FALLBACK, RESULTS_DIR, is_groundable_object_name
from gtf.utils.action_validation import resolve_action_validation
from gtf.utils.video_timing import file_sha256

K_FRAMES = 5
N_GENS = 6  # generations per pointing prompt; enough for a stable refusal fraction near the 0.65 threshold
TEMPERATURE = 0.7
REGROUND_SEED = 42
REGROUND_SEED_SCHEME = "sha256_segment_frame_prompt_draw_v1"
REGROUND_DRAW_BATCH_SIZE = 6
DILATE_FRAC = 0.015  # "inside" tolerance as a fraction of the image diagonal
MIN_MASK_PX = 64
MIN_FRAMES = 2  # fewer scored frames than this -> axes None (gate defers)


# ---------------------------------------------------------------------------
# Evidence collection (GPU: needs a loaded Molmo2)
# ---------------------------------------------------------------------------


def resolve_noun(seg_dir: Path) -> str | None:
    """The segment's validated object noun, or None when action validation did not resolve a valid one."""

    resolution = resolve_action_validation(seg_dir)
    return resolution.object_name if resolution.valid else None


def resolve_window(seg_dir: Path, n_frames: int) -> tuple[int, int]:
    """[first, last] video-frame span of the exported interaction window."""
    meta_p = seg_dir / "metadata.json"
    if meta_p.exists():
        iw = json.loads(meta_p.read_text()).get("interaction_window")
        if iw and "first_frame" in iw:
            return int(iw["first_frame"]), int(iw["last_frame"])
    sc = seg_dir / "optimized_stage2" / "objref_chain.npz"
    if sc.exists():
        try:
            fi = np.load(sc)["frame_indices"]
            if len(fi):
                return int(fi.min()), int(fi.max())
        except Exception:
            pass
    return 0, n_frames - 1


def pick_frames(masks: np.ndarray, lo: int, hi: int, k: int) -> list[int]:
    """Pick k spread, substantially-masked frames from the interval [lo, hi]."""

    hi = min(hi, len(masks) - 1)
    candidate_indices = np.arange(lo, hi + 1, dtype=np.int64)
    if candidate_indices.size == 0:
        return []
    areas = masks[candidate_indices].reshape(len(candidate_indices), -1).sum(axis=1)
    nz = areas[areas > 0]
    floor = max(MIN_MASK_PX, 0.15 * (np.median(nz) if len(nz) else 0))
    cand = candidate_indices[np.where(areas >= floor)[0]]
    if len(cand) == 0:
        cand = candidate_indices[np.where(areas > 0)[0]]
    if len(cand) == 0:
        return []
    if len(cand) <= k:
        return [int(c) for c in cand]
    sel = np.linspace(0, len(cand) - 1, k).round().astype(int)
    return [int(cand[i]) for i in sorted(set(sel))]


def score_points(points, mask: np.ndarray, diag: float):
    """Per point: distance to the mask (px, 0 inside) via distance transform.

    points: run_molmo2.extract_points tuples (raw_x, raw_y, pixel_x, pixel_y)."""
    inv = (mask == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    out = []
    for _rx, _ry, x, y in points:
        xi = int(np.clip(round(x), 0, mask.shape[1] - 1))
        yi = int(np.clip(round(y), 0, mask.shape[0] - 1))
        d = float(dist[yi, xi])
        out.append({"x": float(x), "y": float(y), "dist_px": d, "dist_pct": d / diag * 100.0, "inside": bool(d <= DILATE_FRAC * diag)})
    return out


def reground_query_seed(
    base_seed: int,
    segment: str,
    frame_index: int,
    prompt_name: str,
    draw_index: int,
) -> int:
    """Stable independent seed for one stochastic Molmo2 generation."""

    identity = (f"{REGROUND_SEED_SCHEME}\0{int(base_seed)}\0{segment}\0{int(frame_index)}\0{prompt_name}\0{int(draw_index)}").encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little") % (2**63 - 1)


def _prepare_molmo_query(processor, model, image, prompt):
    """Build the prompt tensors of one (singleton) generation."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {key: value.to(model.device) for key, value in inputs.items()}


def _decode_molmo_completion(processor, sequence, prompt_length: int) -> str:
    return processor.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)


def _query_molmo_singleton(
    processor,
    model,
    image,
    prompt: str,
    *,
    query_seed: int,
    temperature: float,
    greedy: bool = False,
) -> str:
    """One generation seeded with its own query seed (the global RNG state is restored afterwards); the reference
    path that the batched draws reproduce."""

    import torch

    inputs = _prepare_molmo_query(processor, model, image, prompt)
    gen_kwargs = (
        {"max_new_tokens": 16, "do_sample": False}
        if greedy
        else {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": temperature,
        }
    )
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        torch.manual_seed(int(query_seed))
        generated = model.generate(**inputs, **gen_kwargs)
    return _decode_molmo_completion(processor, generated[0], inputs["input_ids"].size(1))


class _ExactSeededRows:
    """Select the same sampled token as one independently-seeded query per row.

    Transformers exposes one RNG stream for an entire generation batch, while gate 21
    binds every draw to its own stable seed. This logits processor keeps those streams
    independent, applies the temperature / top-k sampling, and forces greedy generation
    to emit the sampled token, so the decoder can share a batch without changing the
    seed scheme.
    """

    def __init__(self, generators, *, temperature: float, top_k: int):
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
        )

        self.generators = list(generators)
        self.temperature = TemperatureLogitsWarper(float(temperature))
        self.top_k = TopKLogitsWarper(top_k=int(top_k))

    @classmethod
    def from_seeds(cls, seeds, *, device, temperature: float, top_k: int):
        import torch

        return cls([torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds], temperature=temperature, top_k=top_k)

    def __call__(self, input_ids, scores):
        import torch

        if scores.ndim != 2 or scores.shape[0] != len(self.generators):
            raise RuntimeError("seeded re-ground batch differs from its RNG roster")
        warped = self.temperature(input_ids, scores)
        warped = self.top_k(input_ids, warped)
        probabilities = torch.nn.functional.softmax(warped, dim=-1)
        selected = torch.cat(
            [torch.multinomial(row[None], 1, generator=generator) for row, generator in zip(probabilities, self.generators)],
            dim=0,
        )
        forced = torch.full_like(scores, -torch.inf)
        return forced.scatter(1, selected, 0.0)


def _exact_sampling_top_k(model) -> int:
    """Validate the pinned model's sampling contract before sharing a prefill."""

    import transformers

    from gtf.extraction.frontend.run_molmo2 import MODEL_ID, MODEL_REVISION

    config = model.generation_config
    model_config = model.config
    runtime_identity = {
        "transformers": transformers.__version__,
        "model_name": getattr(model_config, "_name_or_path", None),
        "model_revision": getattr(model_config, "_commit_hash", None),
        "model_class": type(model).__name__,
    }
    expected_identity = {
        "transformers": "4.57.1",
        "model_name": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_class": "Molmo2ForConditionalGeneration",
    }
    if runtime_identity != expected_identity:
        raise RuntimeError(f"unsupported Molmo runtime for exact batched draws: {runtime_identity!r} != {expected_identity!r}")
    expected_defaults = {
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "bad_words_ids": None,
        "sequence_bias": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "watermarking_config": None,
    }
    unexpected = {name: getattr(config, name, None) for name, expected in expected_defaults.items() if getattr(config, name, None) != expected}
    top_k = getattr(config, "top_k", None)
    if unexpected or not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise RuntimeError(f"unsupported Molmo sampling config for exact batched draws: top_k={top_k!r}, unexpected={unexpected}")
    return top_k


def _repeat_molmo_cache(cache, model, count: int):
    """Clone a singleton prompt KV cache into independent decoder rows."""

    from transformers.cache_utils import DynamicCache

    layers = cache if isinstance(cache, tuple) else cache.to_legacy_cache()
    return DynamicCache(
        ddp_cache_data=(
            (
                keys.repeat_interleave(count, dim=0),
                values.repeat_interleave(count, dim=0),
            )
            for keys, values in layers
        ),
        config=model.config.get_text_config(decoder=True),
    )


def _query_molmo_seeded_batch(
    processor,
    model,
    image,
    prompt: str,
    *,
    query_seeds: list[int],
    temperature: float,
) -> list[str]:
    """Branch exact seeded draws from one visual/text prefill and decode together."""

    import torch

    if len(query_seeds) < 2:
        return [
            _query_molmo_singleton(
                processor,
                model,
                image,
                prompt,
                query_seed=query_seeds[0],
                temperature=temperature,
            )
        ]
    top_k = _exact_sampling_top_k(model)
    inputs = _prepare_molmo_query(processor, model, image, prompt)
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True, logits_to_keep=1)
    count = len(query_seeds)
    prompt_length = inputs["input_ids"].size(1)
    input_ids = inputs["input_ids"].repeat(count, 1)
    attention_mask = inputs["attention_mask"].repeat(count, 1)
    sampler = _ExactSeededRows.from_seeds(
        query_seeds,
        device=model.device,
        temperature=temperature,
        top_k=top_k,
    )
    first_scores = sampler(input_ids, prefill.logits[:, -1, :].repeat(count, 1))
    first_tokens = first_scores.argmax(dim=-1, keepdim=True)
    eos_raw = model.generation_config.eos_token_id
    eos_ids = {int(value) for value in (eos_raw if isinstance(eos_raw, list) else [eos_raw]) if value is not None}
    active = [index for index, token in enumerate(first_tokens[:, 0].tolist()) if int(token) not in eos_ids]
    completions = [_decode_molmo_completion(processor, first_tokens[index], 0) for index in range(count)]
    if not active:
        return completions

    active_index = torch.tensor(active, dtype=torch.long, device=input_ids.device)
    active_first = first_tokens.index_select(0, active_index)
    active_sequence = torch.cat((input_ids.index_select(0, active_index), active_first), dim=1)
    active_attention = torch.cat(
        (
            attention_mask.index_select(0, active_index),
            torch.ones_like(active_first, dtype=attention_mask.dtype),
        ),
        dim=1,
    )
    active_sampler = _ExactSeededRows(
        [sampler.generators[index] for index in active],
        temperature=temperature,
        top_k=top_k,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_ids=active_sequence,
            attention_mask=active_attention,
            past_key_values=_repeat_molmo_cache(prefill.past_key_values, model, len(active)),
            max_new_tokens=255,
            do_sample=False,
            logits_processor=[active_sampler],
        )
    for row, output_index in enumerate(active):
        completions[output_index] = _decode_molmo_completion(processor, generated[row], prompt_length)
    return completions


def _query_molmo_draws(
    processor,
    model,
    image,
    prompt: str,
    *,
    query_seeds: list[int],
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Run ordered draws in bounded batches, falling back to exact singleton draws when a batch fails."""

    import torch

    responses = []
    for offset in range(0, len(query_seeds), batch_size):
        chunk = query_seeds[offset : offset + batch_size]
        if len(chunk) == 1:
            responses.append(
                _query_molmo_singleton(
                    processor,
                    model,
                    image,
                    prompt,
                    query_seed=chunk[0],
                    temperature=temperature,
                )
            )
            continue
        try:
            responses.extend(
                _query_molmo_seeded_batch(
                    processor,
                    model,
                    image,
                    prompt,
                    query_seeds=chunk,
                    temperature=temperature,
                )
            )
        except Exception as error:  # noqa: BLE001
            # Nothing is written yet: a model / config / memory incompatibility falls back to the
            # independently-seeded singleton path, which gives the same output.
            print(f"    Reground batch unavailable; retrying exact singleton draws ({type(error).__name__}: {error})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            responses.extend(
                _query_molmo_singleton(
                    processor,
                    model,
                    image,
                    prompt,
                    query_seed=query_seed,
                    temperature=temperature,
                )
                for query_seed in chunk
            )
    return responses


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reground_support_frames(rec: dict) -> list[int]:
    """Unique frames carrying an actually-issued held-object prompt."""

    frames = []
    seen = set()
    for frame in rec.get("frames", []):
        if not isinstance(frame, dict):
            continue
        fi = frame.get("fi")
        held = frame.get("prompts", {}).get("held")
        n_gens = held.get("n_gens") if isinstance(held, dict) else None
        if not isinstance(fi, int) or isinstance(fi, bool) or fi in seen or not isinstance(n_gens, int) or isinstance(n_gens, bool) or n_gens <= 0:
            continue
        seen.add(fi)
        frames.append(fi)
    return sorted(frames)


def has_minimum_reground_support(rec: dict, min_frames: int = MIN_FRAMES) -> bool:
    return len(reground_support_frames(rec)) >= int(min_frames)


def materialized_quality_binds_current_reground(seg_dir: Path) -> bool:
    """Whether quality.json was computed from the current re-ground evidence (its recorded SHA-256 matches)."""

    seg_dir = Path(seg_dir)
    quality_path = seg_dir / "quality" / "quality.json"
    reground_path = seg_dir / "reground" / "reground.json"
    try:
        quality_payload = json.loads(quality_path.read_text())
        live_reground_sha = file_sha256(reground_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False
    return isinstance(quality_payload, dict) and quality_payload.get("reground_sha256") == live_reground_sha


def reground_evidence_needs_refresh(seg_dir: Path, *, force: bool = False) -> bool:
    """Whether the evidence is missing, older than the SAM 3 masks, or has too few supported frames."""

    if force:
        return True
    seg_dir = Path(seg_dir)
    output = seg_dir / "reground" / "reground.json"
    masks = seg_dir / "SAM3" / "masks.npz"
    if not output.is_file() or not masks.is_file():
        return True
    try:
        record = json.loads(output.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return True
    return not has_minimum_reground_support(record) or output.stat().st_mtime < masks.stat().st_mtime


def collect_evidence(
    processor,
    model,
    seg_name: str,
    *,
    results_dir=None,
    k_frames=K_FRAMES,
    n_gens=N_GENS,
    temperature=TEMPERATURE,
    seed=REGROUND_SEED,
    draw_batch_size=None,
) -> dict:
    """Query Molmo2 at k spread frames; score points vs the SAM3 mask; write reground/reground.json."""
    from decord import VideoReader, cpu
    from PIL import Image

    from gtf.extraction.frontend.run_molmo2 import MODEL_ID, MODEL_REVISION, extract_points

    results_dir = Path(results_dir) if results_dir else Path(RESULTS_DIR)
    seg_dir = results_dir / seg_name
    masks = np.load(seg_dir / "SAM3" / "masks.npz")["masks"]
    vr = VideoReader(str(seg_dir / "video.mp4"), ctx=cpu(0), num_threads=2)
    noun = resolve_noun(seg_dir)
    groundable = bool(noun) and is_groundable_object_name(noun)
    lo, hi = resolve_window(seg_dir, len(masks))
    frames = pick_frames(masks, lo, hi, k_frames)
    prompts = {"held": POINTING_PROMPT_FALLBACK}
    if groundable:
        prompts["named"] = f"point to the {noun}"
        prompts["all"] = f"point to every {noun} in the image"
    vqa_prompt = (
        f"Is a person's hand touching or holding the {noun}? Answer yes or no." if groundable else "Is a person's hand touching or holding an object? Answer yes or no."
    )

    if draw_batch_size is None:
        raw_draw_batch_size = os.environ.get("REGROUND_DRAW_BATCH_SIZE", str(REGROUND_DRAW_BATCH_SIZE))
        try:
            draw_batch_size = int(raw_draw_batch_size)
        except ValueError as error:
            raise RuntimeError("REGROUND_DRAW_BATCH_SIZE must be an integer from 1 through 6") from error
    if not isinstance(draw_batch_size, int) or isinstance(draw_batch_size, bool) or not 1 <= draw_batch_size <= REGROUND_DRAW_BATCH_SIZE:
        raise RuntimeError("re-ground draw batch size must be from 1 through 6")

    def query(image_pil, prompt, *, query_seed, greedy=False):
        return _query_molmo_singleton(
            processor,
            model,
            image_pil,
            prompt,
            query_seed=query_seed,
            temperature=temperature,
            greedy=greedy,
        )

    rec = {
        "segment": seg_name,
        "noun": noun,
        "groundable": groundable,
        "window": [lo, hi],
        "n_gens": n_gens,
        "generator": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "seed": int(seed),
            "seed_scheme": REGROUND_SEED_SCHEME,
            "temperature": float(temperature),
        },
        "frames": [],
    }
    for fi in frames:
        f = vr[fi]
        arr = f.numpy() if hasattr(f, "numpy") else f.asnumpy()
        img = Image.fromarray(arr)
        mask = masks[fi]
        diag = float(np.hypot(*mask.shape))
        frec = {"fi": int(fi), "mask_area": int((mask > 0).sum()), "prompts": {}}
        for pname, prompt in prompts.items():
            pts_all, n_empty = [], 0
            points_seeds = [reground_query_seed(seed, seg_name, fi, pname, draw_index) for draw_index in range(n_gens)]
            for response in _query_molmo_draws(
                processor,
                model,
                img,
                prompt,
                query_seeds=points_seeds,
                temperature=temperature,
                batch_size=draw_batch_size,
            ):
                pts = extract_points(
                    response,
                    img.size[0],
                    img.size[1],
                )
                if not pts:
                    n_empty += 1
                    continue
                pts_all.extend(pts)
            frec["prompts"][pname] = {"n_gens": n_gens, "n_empty": n_empty, "points": score_points(pts_all, mask, diag)}
        vqa_seed = reground_query_seed(seed, seg_name, fi, "contact_vqa", 0)
        vqa_text = query(img, vqa_prompt, query_seed=vqa_seed, greedy=True).strip().lower()
        frec["contact_vqa"] = {"text": vqa_text[:60], "yes": vqa_text.startswith("yes") or " yes" in vqa_text[:12]}
        rec["frames"].append(frec)
    _atomic_json(seg_dir / "reground" / "reground.json", rec)
    return rec


# ---------------------------------------------------------------------------
# Aggregation -> the three re-ground axes (CPU, no model)
# ---------------------------------------------------------------------------


def reground_scores(rec: dict, min_frames: int = MIN_FRAMES) -> dict:
    """Evidence dict -> {reground_held_refusal, reground_held_dist, reground_held_anyin}.

    Every axis is None when unmeasurable (missing prompt / too few scored frames), so the
    gate clause defers (the None convention of gates 12-20)."""
    out = {"reground_held_refusal": None, "reground_held_dist": None, "reground_held_anyin": None}
    per_frame_any, per_frame_dist, n_empty, n_gens = [], [], 0, 0
    support_frames = 0
    for f in rec.get("frames", []):
        p = f.get("prompts", {}).get("held")
        if p is None:
            continue
        if int(p.get("n_gens", 0)) <= 0:
            continue
        support_frames += 1
        n_empty += p["n_empty"]
        n_gens += p["n_gens"]
        pts = p["points"]
        if not pts:
            continue
        per_frame_any.append(float(any(q["inside"] for q in pts)))
        per_frame_dist.append(float(np.median([q["dist_pct"] for q in pts])))
    if support_frames >= min_frames and n_gens > 0:
        out["reground_held_refusal"] = round(n_empty / n_gens, 4)
    if len(per_frame_any) >= min_frames:
        out["reground_held_anyin"] = round(float(np.mean(per_frame_any)), 4)
        out["reground_held_dist"] = round(float(np.median(per_frame_dist)), 4)
    return out


def load_reground_scores(seg_dir) -> dict:
    """quality_gate hook: read {seg}/reground/reground.json -> gate axes (all None if absent)."""
    p = Path(seg_dir) / "reground" / "reground.json"
    if not p.exists():
        return {"reground_held_refusal": None, "reground_held_dist": None, "reground_held_anyin": None}
    try:
        return reground_scores(json.loads(p.read_text()))
    except Exception:
        return {"reground_held_refusal": None, "reground_held_dist": None, "reground_held_anyin": None}
