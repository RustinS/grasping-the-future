"""Exact, process-local caching for the per-frame HDF5 scene reader.

A bounded LRU of read-only HDF5 handles plus a bounded LRU of decoded rows. The
injected ``reader(h5, i)`` is authoritative for row semantics. There are no
parallel reads, disk writes, prefetches, coordinate transforms, RNG calls, or
negative-cache entries.

Every lookup stats the source, including RAM hits. Changing or disappearing
files invalidate their handles and rows. Metadata identity is not a content
hash; concurrently mutating a training corpus is unsupported. Cached tensors
are private CPU snapshots and callers receive independent clones. ``max_bytes``
counts tensor payload, not Python or HDF5-library overhead.
"""

from collections import Counter, OrderedDict
import os
import threading
import weakref

import torch


class SourceChangedError(RuntimeError):
    """A source changed during its read; the result was not cached."""


def _lexical_path(path):
    path = os.fsdecode(os.fspath(path))
    # Do not normalize link/..: kernel traversal can select a different file.
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def _identity(path):
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _clone_row(row):
    if not isinstance(row, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in row.items()
    ):
        raise TypeError("Scene reader must return dict[str, CPU Tensor] or None")
    if any(value.device.type != "cpu" or value.requires_grad for value in row.values()):
        raise ValueError("Scene rows must contain non-grad CPU tensors")
    return {key: value.clone() for key, value in row.items()}


def _fork_before(reference):
    cache = reference()
    if cache is not None:
        cache._lock.acquire()


def _fork_parent(reference):
    cache = reference()
    if cache is not None:
        cache._lock.release()


def _fork_child(reference):
    cache = reference()
    if cache is not None:
        cache._reset_process()


class SceneRowCache:
    """Callable ``cache(path, frame)`` with independent FD and RAM budgets.

    ``reader`` accepts an open read-only ``h5py.File`` plus the integer frame.
    Setting either budget to zero disables that layer. ``close()`` is final.
    A fork/PID change discards inherited HDF5 handles and cached tensors before
    use. The parent keeps its own cache. Calls within a process are serialized.
    """

    def __init__(self, reader, *, max_open_files=32, max_bytes=128 * 1024**2):
        if not callable(reader):
            raise TypeError("reader must be callable")
        for name, value in (("max_open_files", max_open_files), ("max_bytes", max_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.reader = reader
        self.max_open_files = max_open_files
        self.max_bytes = max_bytes
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._handles = OrderedDict()
        self._rows = OrderedDict()
        self._identities = {}
        self._row_counts = Counter()
        self._bytes = 0
        self._closed = False
        if hasattr(os, "register_at_fork"):
            # Weak references avoid extending the cache/handle lifetime merely
            # because Python retains registered callbacks for process lifetime.
            reference = weakref.ref(self)
            os.register_at_fork(
                before=lambda: _fork_before(reference),
                after_in_parent=lambda: _fork_parent(reference),
                after_in_child=lambda: _fork_child(reference),
            )

    def _clear(self):
        for _, handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._rows.clear()
        self._identities.clear()
        self._row_counts.clear()
        self._bytes = 0

    def _reset_process(self):
        # A lock inherited from a vanished parent thread must not be acquired.
        self._lock = threading.RLock()
        self._clear()
        self._pid = os.getpid()

    def _ensure_process(self):
        if self._pid != os.getpid():
            self._reset_process()

    def _invalidate(self, path):
        old = self._handles.pop(path, None)
        if old is not None:
            old[1].close()
        for key in list(self._rows):
            if key[0] == path:
                _, size = self._rows.pop(key)
                self._bytes -= size
        self._identities.pop(path, None)
        self._row_counts.pop(path, None)

    def _open(self, path, identity):
        import h5py

        existing = self._handles.get(path)
        if existing is not None:
            if existing[0] == identity and existing[1].id.valid:
                self._handles.move_to_end(path)
                return existing[1]
            self._invalidate(path)
        # Evict before opening so the configured retained-FD limit is not
        # transiently exceeded. Budget zero uses one temporary handle instead.
        while self.max_open_files and len(self._handles) >= self.max_open_files:
            old_path, (_, old) = self._handles.popitem(last=False)
            old.close()
            if not self._row_counts[old_path]:
                self._identities.pop(old_path, None)
        handle = h5py.File(path, "r")
        if self.max_open_files:
            self._handles[path] = identity, handle
            self._identities[path] = identity
        return handle

    def _put(self, key, raw):
        size = sum(value.numel() * value.element_size() for value in raw.values())
        if size == 0 or size > self.max_bytes:
            return
        while self._rows and self._bytes + size > self.max_bytes:
            old_key, (_, old_size) = self._rows.popitem(last=False)
            self._bytes -= old_size
            old_path = old_key[0]
            self._row_counts[old_path] -= 1
            if not self._row_counts[old_path]:
                del self._row_counts[old_path]
                if old_path not in self._handles:
                    self._identities.pop(old_path, None)
        self._rows[key] = raw, size
        self._identities[key[0]] = key[1]
        self._row_counts[key[0]] += 1
        self._bytes += size

    def load(self, path, canonical_idx):
        self._ensure_process()
        path = _lexical_path(path)
        idx = int(canonical_idx)
        with self._lock:
            if self._closed:
                raise RuntimeError("SceneRowCache is closed")
            try:
                identity = _identity(path)
            except OSError:
                self._invalidate(path)
                raise
            key = path, identity, idx
            # Rows may outlive their handles. Remove any old generation on a
            # source change, not just when that path still has an open handle.
            existing_identity = self._identities.get(path)
            if existing_identity is not None and existing_identity != identity:
                self._invalidate(path)
            cached = self._rows.get(key)
            if cached is not None:
                self._rows.move_to_end(key)
                return _clone_row(cached[0])
            handle = None
            try:
                handle = self._open(path, identity)
                row = self.reader(handle, idx)
                if _identity(path) != identity:
                    raise SourceChangedError(f"Scene source changed during row read: {path}")
                if row is None:
                    return None
                raw = _clone_row(row)
                self._put(key, raw)
                return _clone_row(raw)
            except BaseException:
                self._invalidate(path)
                raise
            finally:
                if handle is not None and not self.max_open_files:
                    handle.close()

    __call__ = load

    def close(self):
        self._ensure_process()
        with self._lock:
            self._clear()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
