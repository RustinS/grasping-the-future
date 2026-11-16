"""Versioned cache of frozen Uni3D object embeddings.

The object point cloud is deterministic per segment (area-weighted surface
samples seeded by the segment id) and Uni3D is frozen, so each segment's
1024-D embedding is computed once and stored in one artifact.

The artifact stores both native outputs: bf16-autocast for training and fp32 for
AMP-off evaluation. Uni3D's two numerical paths differ substantially, so one
cannot be cast to reconstruct the other. Each is generated at the batch shape it
stands in for (B32 training, B16 evaluation); the trainable object projection
consumes the cached tensor directly.

Artifact layout (``torch.save`` dict, loadable with ``weights_only=True``):
``segment_ids`` (list[str]), ``embeddings_bf16`` (N, 1024) bf16,
``embeddings_fp32`` (N, 1024) fp32, ``metadata`` (built-in types only; see
``_require_metadata``). ``generate_padded_embeddings`` and
``write_uni3d_cache_atomic`` build it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import torch
from torch import Tensor

UNI3D_CACHE_FORMAT_VERSION = 1
UNI3D_EMBEDDING_DIM = 1024
UNI3D_GENERATION_BATCH_SIZE = 32
UNI3D_EVAL_BATCH_SIZE = 16


def sha256_file(path: str | os.PathLike, *, short: bool = False) -> str:
    """Streaming SHA-256 for large assets/artifacts."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:16] if short else digest


def _hash_string_sequence(values: Sequence[str], *, ordered: bool) -> str:
    seq = list(values) if ordered else sorted(values)
    h = hashlib.sha256()
    for value in seq:
        b = str(value).encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def ordered_segment_ids_digest(segment_ids: Sequence[str]) -> str:
    """Digest that binds cache row i to segment_ids[i]."""
    return _hash_string_sequence(segment_ids, ordered=True)


def population_digest(segment_ids: Iterable[str]) -> str:
    """Order-independent digest of a set of segment ids."""
    return _hash_string_sequence(list(segment_ids), ordered=False)


def tensor_digest(tensor: Tensor) -> str:
    """SHA-256 over dtype, shape, and exact contiguous tensor bytes."""
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    # Viewing as bytes works for bf16, which NumPy cannot represent natively.
    h.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return h.hexdigest()


def _cfg_get(obj, key: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def uni3d_cache_data_contract(cfg, *, uni3d_args: Mapping | None = None) -> dict:
    """Description of how the cached embeddings were produced (stored in the artifact).

    The population covers every split of the scene-complete manifest, so one
    artifact serves training and evaluation.
    """
    data = _cfg_get(cfg, "data")
    args = dict(uni3d_args or {})
    return {
        "population_scope": ["train", "val", "test"],
        "manifest_path": str(_cfg_get(data, "manifest_path", "datasets/canonical_manifest.json")),
        "exclude_sources": [],
        "require_scene_features": True,
        "subsample_fraction": 1.0,
        "subsample_seed": 0,
        "cache_obj_points": True,
        "obj_points_area_weighted": True,
        "point_count": int(args.get("npoints", 10000)),
        "point_seed": "crc32(segment_id)&0x7fffffff",
        "point_sampler": "deterministic_triangle_area_barycentric_v1",
        # Identifier of the split policy; a fixed string, part of the cache digest.
        "split_policy": "gtf.data.compute_stats.get_split",
        "uni3d_args": {str(k): args[k] for k in sorted(args)},
    }


def uni3d_cache_config_digest(contract: Mapping) -> str:
    """Full SHA-256 over a data contract."""
    return hashlib.sha256(json.dumps(dict(contract), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_uni3d_dataset_kwargs(data_cfg) -> dict:
    """GTFDataset kwargs that attach the configured Uni3D embedding cache."""
    path = str(_cfg_get(data_cfg, "uni3d_cache_path", "") or "")
    if not path:
        raise ValueError("data.uni3d_cache_path must name the Uni3D embedding cache")
    return {"uni3d_cache_path": path}


def update_points_digest(hasher, segment_id: str, points: Tensor) -> None:
    """Fold one exact Uni3D input cloud into a streaming population digest."""
    value = points.detach().cpu().to(torch.float32).contiguous()
    sid = str(segment_id).encode("utf-8")
    hasher.update(len(sid).to_bytes(8, "little"))
    hasher.update(sid)
    hasher.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    hasher.update(value.view(torch.uint8).numpy().tobytes(order="C"))


@dataclass
class Uni3DCacheLookup:
    """Compact segment-id lookup over the two contiguous embedding tensors."""

    segment_ids: tuple[str, ...]
    embeddings_bf16: Tensor
    embeddings_fp32: Tensor

    def __post_init__(self) -> None:
        self._index = {sid: i for i, sid in enumerate(self.segment_ids)}

    def __len__(self) -> int:
        return len(self.segment_ids)

    def __contains__(self, segment_id: object) -> bool:
        return segment_id in self._index

    def embedding(self, segment_id: str, *, precision: str) -> Tensor:
        try:
            idx = self._index[str(segment_id)]
        except KeyError as exc:
            raise KeyError(f"Uni3D cache has no embedding for segment {segment_id!r}") from exc

        if precision == "bf16":
            return self.embeddings_bf16[idx]
        if precision == "fp32":
            return self.embeddings_fp32[idx]
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")

    def __getitem__(self, segment_id: str) -> tuple[Tensor, Tensor]:
        """Return the ``(bf16, fp32)`` rows of one segment."""
        return self.embedding(segment_id, precision="bf16"), self.embedding(segment_id, precision="fp32")


@dataclass(frozen=True)
class _CachedUni3DArtifact:
    lookup: Uni3DCacheLookup
    metadata: dict
    file_digest: str


# One process may construct the LitModule, the train Dataset and the val Dataset
# from the same artifact. Reusing the validated CPU tensors avoids repeated
# deserialization (and lets forked workers share their pages copy-on-write).
# File identity, rather than pathname alone, makes an atomic os.replace visible.
_UNI3D_CACHE_LOCK = threading.Lock()
_UNI3D_CACHE_BY_FILE_IDENTITY: dict[tuple[str, int, int, int, int], _CachedUni3DArtifact] = {}


def _uni3d_cache_file_identity(
    path: str | os.PathLike,
) -> tuple[Path, tuple[str, int, int, int, int]]:
    requested = Path(path).expanduser()
    if not requested.is_file():
        raise FileNotFoundError(f"Uni3D cache artifact not found: {requested}")
    resolved = requested.resolve()
    stat = resolved.stat()
    return resolved, (
        str(resolved),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _check_required_segments(segment_ids: Sequence[str], required_segment_ids: Iterable[str] | None) -> None:
    if required_segment_ids is None:
        return
    available = set(segment_ids)
    missing = sorted({str(segment_id) for segment_id in required_segment_ids} - available)
    if missing:
        raise ValueError(f"Uni3D cache misses {len(missing)} required segment(s): {missing[:8]}")


def _require_metadata(metadata: Mapping) -> None:
    required = {
        "format_version",
        "population_scope",
        "segment_count",
        "split_counts",
        "segment_ids_digest",
        "population_digest",
        "config_digest",
        "data_contract",
        "generation",
        "embedding_bf16_digest",
        "embedding_fp32_digest",
        "embedding_dim",
        "embedding_bf16_dtype",
        "embedding_fp32_dtype",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Uni3D cache metadata misses required keys: {missing}")


def _require_weights_only_safe_metadata(value, *, path: str = "metadata") -> None:
    """Reject pickle-only metadata before an artifact is written.

    The loader uses ``weights_only=True``. Some library values (for example
    PyTorch's ``TorchVersion`` string subclass) serialize with a GLOBAL pickle
    opcode even though they look like plain strings. Restricting metadata to
    exact built-in containers/scalars guarantees the safe loader can read it.
    """
    if value is None or type(value) in (bool, int, float, str):
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _require_weights_only_safe_metadata(item, path=f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"Uni3D cache {path} has non-string key {key!r} ({type(key).__name__})")
            _require_weights_only_safe_metadata(item, path=f"{path}.{key}")
        return
    raise ValueError(f"Uni3D cache {path} must use weights-only-safe built-in metadata, got {type(value).__module__}.{type(value).__qualname__}")


def validate_uni3d_cache_payload(
    payload: Mapping,
    *,
    required_segment_ids: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], Tensor, Tensor, dict]:
    """Validate an in-memory artifact and return normalized components.

    Checks structure, dtypes, shapes, finiteness and the stored tensor/segment
    digests. ``required_segment_ids`` is a subset contract: an artifact holds all
    splits, while one Dataset only requires its selected segments.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(f"Uni3D cache payload must be a mapping, got {type(payload).__name__}")
    expected_payload_keys = {
        "segment_ids",
        "embeddings_bf16",
        "embeddings_fp32",
        "metadata",
    }
    if set(payload) != expected_payload_keys:
        raise ValueError(f"Uni3D cache payload keys must be {sorted(expected_payload_keys)}, got {sorted(payload)}")

    raw_ids = payload["segment_ids"]
    embeddings_bf16 = payload["embeddings_bf16"]
    embeddings_fp32 = payload["embeddings_fp32"]
    metadata = payload["metadata"]
    if not isinstance(raw_ids, (list, tuple)) or not all(isinstance(v, str) and v for v in raw_ids):
        raise ValueError("Uni3D cache segment_ids must be non-empty strings")
    segment_ids = tuple(raw_ids)
    if len(segment_ids) != len(set(segment_ids)):
        raise ValueError("Uni3D cache segment_ids contain duplicates (row lookup would be ambiguous)")
    if not isinstance(embeddings_bf16, Tensor) or not isinstance(embeddings_fp32, Tensor):
        raise ValueError("Uni3D cache embeddings_bf16/embeddings_fp32 must both be Tensors")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Uni3D cache metadata must be a mapping, got {type(metadata).__name__}")
    metadata = dict(metadata)
    _require_metadata(metadata)
    _require_weights_only_safe_metadata(metadata)

    if int(metadata["format_version"]) != UNI3D_CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported Uni3D cache format {metadata['format_version']!r}; expected {UNI3D_CACHE_FORMAT_VERSION}")
    for label, embeddings, dtype in (
        ("bf16", embeddings_bf16, torch.bfloat16),
        ("fp32", embeddings_fp32, torch.float32),
    ):
        if embeddings.device.type != "cpu":
            raise ValueError(f"persisted Uni3D {label} embeddings must load on CPU, got {embeddings.device}")
        if embeddings.ndim != 2 or embeddings.shape[1] != UNI3D_EMBEDDING_DIM:
            raise ValueError(f"Uni3D {label} embeddings must have shape (N,{UNI3D_EMBEDDING_DIM}), got {tuple(embeddings.shape)}")
        if embeddings.dtype != dtype:
            raise ValueError(f"Uni3D {label} embeddings must retain native {dtype}, got {embeddings.dtype}")
        if len(segment_ids) != embeddings.shape[0]:
            raise ValueError(f"segment-id rows ({len(segment_ids)}) != {label} embedding rows ({embeddings.shape[0]})")
        if not bool(torch.isfinite(embeddings.float()).all()):
            raise ValueError(f"Uni3D cache contains non-finite {label} embeddings")
    if int(metadata["segment_count"]) != len(segment_ids):
        raise ValueError(f"metadata segment_count={metadata['segment_count']} != payload rows={len(segment_ids)}")
    if metadata["population_scope"] != "runtime_filtered_train+val+test":
        raise ValueError(f"unsupported Uni3D cache population_scope={metadata['population_scope']!r}")
    split_counts = metadata["split_counts"]
    expected_splits = {"train", "val", "test"}
    if not isinstance(split_counts, Mapping) or set(split_counts) != expected_splits:
        raise ValueError(f"Uni3D cache split_counts must contain exactly train/val/test, got {split_counts!r}")
    if any(int(split_counts[k]) <= 0 for k in expected_splits) or sum(int(v) for v in split_counts.values()) != len(segment_ids):
        raise ValueError(f"Uni3D cache split_counts do not partition all rows: {split_counts!r} vs {len(segment_ids)}")
    if int(metadata["embedding_dim"]) != UNI3D_EMBEDDING_DIM:
        raise ValueError(f"metadata embedding_dim={metadata['embedding_dim']} != {UNI3D_EMBEDDING_DIM}")
    if str(metadata["embedding_bf16_dtype"]) != "torch.bfloat16":
        raise ValueError(f"metadata embedding_bf16_dtype={metadata['embedding_bf16_dtype']!r} != 'torch.bfloat16'")
    if str(metadata["embedding_fp32_dtype"]) != "torch.float32":
        raise ValueError(f"metadata embedding_fp32_dtype={metadata['embedding_fp32_dtype']!r} != 'torch.float32'")

    if str(metadata["segment_ids_digest"]) != ordered_segment_ids_digest(segment_ids):
        raise ValueError("Uni3D cache ordered segment_ids_digest does not match its row order")
    if str(metadata["population_digest"]) != population_digest(segment_ids):
        raise ValueError("Uni3D cache population_digest does not match its segment IDs")
    contract = metadata["data_contract"]
    if not isinstance(contract, Mapping) or str(metadata["config_digest"]) != uni3d_cache_config_digest(contract):
        raise ValueError("Uni3D cache config_digest does not match its data_contract")
    if str(metadata["embedding_bf16_digest"]) != tensor_digest(embeddings_bf16):
        raise ValueError("Uni3D cache embedding_bf16_digest does not match its tensor payload")
    if str(metadata["embedding_fp32_digest"]) != tensor_digest(embeddings_fp32):
        raise ValueError("Uni3D cache embedding_fp32_digest does not match its tensor payload")

    generation = metadata["generation"]
    if not isinstance(generation, Mapping):
        raise ValueError("Uni3D cache generation metadata must be a mapping")
    exact_generation = {"matmul_precision": "high", "encoder_mode": "eval"}
    for key, expected in exact_generation.items():
        if generation.get(key) != expected:
            raise ValueError(f"Uni3D cache generation.{key}={generation.get(key)!r} != exact contract {expected!r}")
    variants = generation.get("variants")
    expected_variants = {
        "bf16": {
            "batch_size": UNI3D_GENERATION_BATCH_SIZE,
            "autocast": "torch.bfloat16",
            "output_dtype": "torch.bfloat16",
            "padding": "repeat_last_to_full_batch_then_slice",
            "padding_rows": (-len(segment_ids)) % UNI3D_GENERATION_BATCH_SIZE,
        },
        "fp32": {
            "batch_size": UNI3D_EVAL_BATCH_SIZE,
            "autocast": "disabled",
            "output_dtype": "torch.float32",
            "padding": "repeat_last_to_full_batch_then_slice",
            "padding_rows": (-len(segment_ids)) % UNI3D_EVAL_BATCH_SIZE,
        },
    }
    if variants != expected_variants:
        raise ValueError(f"Uni3D cache generation.variants={variants!r} != exact dual-precision contract {expected_variants!r}")

    _check_required_segments(segment_ids, required_segment_ids)
    return (
        segment_ids,
        embeddings_bf16.contiguous(),
        embeddings_fp32.contiguous(),
        metadata,
    )


def load_uni3d_cache(
    path: str | os.PathLike,
    *,
    required_segment_ids: Iterable[str] | None = None,
) -> tuple[Uni3DCacheLookup, dict, str]:
    """Load and validate an artifact, reusing it within this process.

    Returns ``(lookup, metadata, file_sha256[:16])``. The process cache is keyed
    by resolved path, inode, size, mtime_ns and ctime_ns, so an atomic
    replacement causes a reload. ``required_segment_ids`` is checked on every
    call, including cache hits.
    """
    resolved, identity = _uni3d_cache_file_identity(path)
    with _UNI3D_CACHE_LOCK:
        cached = _UNI3D_CACHE_BY_FILE_IDENTITY.get(identity)
        if cached is None:
            # If an atomic replacement races deserialization or hashing, retry
            # once rather than pairing one generation's tensors with another
            # generation's file digest.
            for attempt in range(2):
                try:
                    payload = torch.load(resolved, map_location="cpu", weights_only=True)
                except Exception as exc:
                    raise RuntimeError(f"failed to load Uni3D cache artifact {resolved}: {exc}") from exc
                segment_ids, embeddings_bf16, embeddings_fp32, metadata = validate_uni3d_cache_payload(payload)
                file_digest = sha256_file(resolved, short=True)
                _post_resolved, post_identity = _uni3d_cache_file_identity(resolved)
                if post_identity == identity:
                    break
                if attempt == 1:
                    raise RuntimeError(f"Uni3D cache artifact changed repeatedly while loading: {resolved}")
                identity = post_identity
            cached = _CachedUni3DArtifact(
                lookup=Uni3DCacheLookup(segment_ids, embeddings_bf16, embeddings_fp32),
                metadata=metadata,
                file_digest=file_digest,
            )
            # Keep at most one generation per resolved pathname.
            for stale in tuple(_UNI3D_CACHE_BY_FILE_IDENTITY):
                if stale[0] == identity[0] and stale != identity:
                    del _UNI3D_CACHE_BY_FILE_IDENTITY[stale]
            _UNI3D_CACHE_BY_FILE_IDENTITY[identity] = cached

    _check_required_segments(cached.lookup.segment_ids, required_segment_ids)
    # Copy the metadata so a caller cannot mutate the shared cached record.
    return cached.lookup, copy.deepcopy(cached.metadata), cached.file_digest


def write_uni3d_cache_atomic(
    path: str | os.PathLike,
    *,
    segment_ids: Sequence[str],
    embeddings_bf16: Tensor,
    embeddings_fp32: Tensor,
    metadata: Mapping,
) -> str:
    """Atomically persist, reload, and validate a cache artifact; return its short file digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "segment_ids": list(segment_ids),
        "embeddings_bf16": embeddings_bf16.detach().cpu().contiguous(),
        "embeddings_fp32": embeddings_fp32.detach().cpu().contiguous(),
        "metadata": dict(metadata),
    }
    # Validate before touching disk, then again from serialized bytes.
    validate_uni3d_cache_payload(payload)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        loaded = torch.load(tmp, map_location="cpu", weights_only=True)
        validate_uni3d_cache_payload(loaded)
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Some shared filesystems do not support directory fsync; the file
            # itself was fsynced and os.replace remains atomic.
            pass
    finally:
        if tmp.exists():
            tmp.unlink()
    # Load through the public path, not the temporary payload, before success.
    _lookup, _metadata, file_digest = load_uni3d_cache(path)
    return file_digest


def generate_padded_embeddings(
    segment_ids: Sequence[str],
    point_loader: Callable[[str], Tensor],
    encode_batch_bf16: Callable[[Tensor], Tensor],
    encode_batch_fp32: Callable[[Tensor], Tensor],
    *,
    device: torch.device | str,
    batch_size: int = UNI3D_GENERATION_BATCH_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Tensor, Tensor, str, dict[str, int]]:
    """Generate native embeddings at their exact batch shapes.

    bf16 uses padded B32 (training batches); fp32 uses padded B16 (evaluation
    batches). Fixed-size padding matters: fp32 Uni3D on GPU is invariant to
    batch peers/order at a fixed B, but B32 and B16 select measurably different
    numerical kernels. Final partial rows repeat their last input and are sliced.

    Returns ``(embeddings_bf16, embeddings_fp32, points_sha256, padding_rows)``.
    """
    if batch_size != UNI3D_GENERATION_BATCH_SIZE:
        raise ValueError(f"Uni3D cache generation is fixed at B{UNI3D_GENERATION_BATCH_SIZE}; got B{batch_size}")
    if not segment_ids:
        raise ValueError("cannot build an empty Uni3D cache")
    if len(segment_ids) != len(set(str(v) for v in segment_ids)):
        raise ValueError("cannot build Uni3D cache from duplicate segment IDs")

    device = torch.device(device)
    point_hasher = hashlib.sha256()
    chunks_bf16: list[Tensor] = []
    chunks_fp32: list[Tensor] = []
    padding_rows = {
        "bf16": (-len(segment_ids)) % UNI3D_GENERATION_BATCH_SIZE,
        "fp32": (-len(segment_ids)) % UNI3D_EVAL_BATCH_SIZE,
    }
    total = len(segment_ids)
    for start in range(0, total, batch_size):
        ids = [str(v) for v in segment_ids[start : start + batch_size]]
        points = []
        shape = None
        for sid in ids:
            value = point_loader(sid)
            if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[-1] != 6:
                raise ValueError(f"Uni3D input for {sid!r} must have shape (N,6), got {getattr(value, 'shape', None)}")
            value = value.detach().cpu().to(torch.float32).contiguous()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Uni3D input for {sid!r} contains non-finite values")
            if shape is None:
                shape = tuple(value.shape)
            elif tuple(value.shape) != shape:
                raise ValueError(f"Uni3D input shape changed within batch: {shape} vs {tuple(value.shape)} for {sid!r}")
            update_points_digest(point_hasher, sid, value)
            points.append(value)

        real = len(points)
        if real < batch_size:
            points.extend([points[-1]] * (batch_size - real))
        batch = torch.stack(points, dim=0).to(device=device, non_blocking=False)
        with torch.inference_mode():
            encoded_bf16 = encode_batch_bf16(batch)
        if not isinstance(encoded_bf16, Tensor) or tuple(encoded_bf16.shape) != (UNI3D_GENERATION_BATCH_SIZE, UNI3D_EMBEDDING_DIM):
            raise ValueError(
                f"Uni3D bf16 encoder must return ({UNI3D_GENERATION_BATCH_SIZE},{UNI3D_EMBEDDING_DIM}), got {getattr(encoded_bf16, 'shape', None)}"
            )
        if encoded_bf16.dtype != torch.bfloat16:
            raise ValueError(f"Uni3D bf16 encoder output must be native {torch.bfloat16}, got {encoded_bf16.dtype}")
        if not bool(torch.isfinite(encoded_bf16.float()).all()):
            raise ValueError(f"Uni3D bf16 encoder returned non-finite values for batch starting at row {start}")
        chunks_bf16.append(encoded_bf16[:real].detach().cpu().contiguous())

        # Keep the global population order while subdividing each B32 load into
        # two B16 fp32 forwards. The terminal subgroup is padded independently to
        # B16, then sliced back to its real rows.
        fp32_rows: list[Tensor] = []
        for sub_start in range(0, real, UNI3D_EVAL_BATCH_SIZE):
            sub_real = min(UNI3D_EVAL_BATCH_SIZE, real - sub_start)
            sub = batch[sub_start : sub_start + sub_real]
            if sub_real < UNI3D_EVAL_BATCH_SIZE:
                sub = torch.cat([sub, sub[-1:].expand(UNI3D_EVAL_BATCH_SIZE - sub_real, *sub.shape[1:])], dim=0)
            with torch.inference_mode():
                encoded_fp32 = encode_batch_fp32(sub)
            if not isinstance(encoded_fp32, Tensor) or tuple(encoded_fp32.shape) != (UNI3D_EVAL_BATCH_SIZE, UNI3D_EMBEDDING_DIM):
                raise ValueError(
                    f"Uni3D fp32 encoder must return ({UNI3D_EVAL_BATCH_SIZE},{UNI3D_EMBEDDING_DIM}), got {getattr(encoded_fp32, 'shape', None)}"
                )
            if encoded_fp32.dtype != torch.float32:
                raise ValueError(f"Uni3D fp32 encoder output must be native {torch.float32}, got {encoded_fp32.dtype}")
            if not bool(torch.isfinite(encoded_fp32).all()):
                raise ValueError(f"Uni3D fp32 encoder returned non-finite values for batch starting at row {start + sub_start}")
            fp32_rows.append(encoded_fp32[:sub_real].detach().cpu().contiguous())
        chunks_fp32.append(torch.cat(fp32_rows, dim=0))
        if progress is not None:
            progress(min(start + real, total), total)
    return (
        torch.cat(chunks_bf16, dim=0),
        torch.cat(chunks_fp32, dim=0),
        point_hasher.hexdigest(),
        padding_rows,
    )
