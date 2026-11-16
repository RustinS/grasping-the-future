#!/usr/bin/env python3
"""Fit the global PCA basis of the DINOv3 scene features.

Samples a source-stratified subset of manifest segments, extracts DINOv3 ViT-L/16
patch features at each segment's middle frame, and fits a 32-D PCA basis. Every scene
cache is projected with this one basis, so the semantic channels mean the same thing
across segments. A trained model expects the basis it was trained with: keep the
basis file next to the checkpoint and do not refit it for an existing model.

Usage:
    python -m gtf.datasets.scene.fit_pca --n-segments 500 --features-per-seg 200
    python -m gtf.datasets.scene.fit_pca --n-components 32 --output datasets/scene_dinov3_pca.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from gtf.paths import REPO_ROOT
from gtf.datasets.scene.depth_sources import MOGE_MODEL_ID, MOGE_SOURCES, load_anchor_depth, load_canonical_for_scene
from gtf.datasets.scene.dinov3_semantic import DinoV3Extractor

ROOT = REPO_ROOT


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m gtf.datasets.scene.fit_pca", description=__doc__)
    p.add_argument("--manifest", default=str(ROOT / "datasets" / "canonical_manifest.json"))
    p.add_argument("--output", default=str(ROOT / "datasets" / "scene_dinov3_pca.pt"))
    p.add_argument("--n-segments", type=int, default=500, help="Segments to sample (stratified across sources).")
    p.add_argument("--features-per-seg", type=int, default=200, help="Random patches per segment.")
    p.add_argument("--n-components", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite existing basis file.")
    p.add_argument("--source", default=None, help="Restrict to one source dataset (otherwise stratified).")
    return p.parse_args()


def _stratified_sample(entries: list[dict], n_total: int, rng: np.random.RandomState) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_source[e["source_dataset"]].append(e)
    # Proportional to counts, minimum 1 per source
    total = sum(len(v) for v in by_source.values())
    picked: list[dict] = []
    for src, items in by_source.items():
        n_src = max(1, int(round(n_total * len(items) / total)))
        n_src = min(n_src, len(items))
        idx = rng.choice(len(items), n_src, replace=False)
        picked.extend([items[i] for i in idx])
    if len(picked) > n_total:
        rng.shuffle(picked)
        picked = picked[:n_total]
    return picked


def main() -> None:
    args = _parse_args()
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)")
        sys.exit(1)

    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = [e for e in manifest["entries"] if e.get("validated", True)]
    if args.source:
        entries = [e for e in entries if e["source_dataset"] == args.source]
    print(f"Manifest: {len(entries)} validated entries")

    rng = np.random.RandomState(int(args.seed))
    sampled = _stratified_sample(entries, int(args.n_segments), rng)
    print(f"Sampled {len(sampled)} segments for PCA fit")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    moge_model = None
    if any(s["source_dataset"] in MOGE_SOURCES for s in sampled):
        print("Loading MoGe-2 for the RGB-only sources...")
        from moge.model.v2 import MoGeModel

        moge_model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(device).eval()

    extractor = DinoV3Extractor(device=device)
    extractor._load()  # patch_size is recorded in the basis
    patch_size = int(extractor.patch_size)

    collected: list[np.ndarray] = []
    n_features_target = args.features_per_seg
    n_used_segments = 0

    for entry in tqdm(sampled):
        try:
            canonical_data = load_canonical_for_scene(entry["canonical_path"])
            anchor = load_anchor_depth(entry, canonical_data, moge_model)
            if anchor is None:
                continue
            patch_feats = extractor.extract_patch_features(anchor.rgb)  # (H_p, W_p, C)
            flat = patch_feats.reshape(-1, patch_feats.shape[-1])
            n_take = min(n_features_target, flat.shape[0])
            pick = rng.choice(flat.shape[0], n_take, replace=False)
            collected.append(flat[pick])
            n_used_segments += 1
        except Exception as e:
            print(f"  skip {entry['segment_id']}: {type(e).__name__}: {e}")
            continue

    if not collected:
        sys.exit("No features collected — check manifest / sources")

    features = np.concatenate(collected, axis=0).astype(np.float32)
    print(f"Collected {features.shape[0]} features from {n_used_segments} segments (dim={features.shape[1]})")

    n_components = min(int(args.n_components), features.shape[0], features.shape[1])
    pca = PCA(n_components=n_components, whiten=False, random_state=int(args.seed))
    pca.fit(features)
    print("Explained variance ratio:")
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for i, (v, c) in enumerate(zip(pca.explained_variance_ratio_.tolist(), cumulative.tolist())):
        if i < 8 or i == n_components - 1:
            print(f"  dim {i}: {v:.4f}  (cum={c:.4f})")
    print(f"Total explained by {n_components} components: {cumulative[-1]:.4f}")

    basis = {
        "components": pca.components_.astype(np.float32),  # (k, C)
        "mean": pca.mean_.astype(np.float32),  # (C,)
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "n_segments_used": int(n_used_segments),
        "features_per_seg": int(args.features_per_seg),
        "n_components": int(n_components),
        "manifest_version": manifest.get("version", "unknown"),
        "encoder_model_id": extractor.model_id,
        "encoder_patch_size": patch_size,
        "seed": int(args.seed),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, str(out))
    print(f"Saved PCA basis to {out}")


if __name__ == "__main__":
    main()
