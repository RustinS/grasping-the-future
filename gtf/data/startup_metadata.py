"""Process-local cache for the parsed training manifest.

The training and validation datasets (and the run-name helper) each read the
same large manifest JSON. ``load_manifest`` parses it once per process and
returns an independent deep copy on every call. A change to the file's
identity (path, device/inode, size, mtime_ns, ctime_ns) triggers a re-read.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading

_LOCK = threading.RLock()
_MEMO: tuple[tuple, object] | None = None  # (identity, parsed manifest); one manifest per process


def _after_fork():
    # A DataLoader child must not inherit an RLock held by a vanished thread.
    global _LOCK, _MEMO
    _LOCK = threading.RLock()
    _MEMO = None


os.register_at_fork(after_in_child=_after_fork)


def _identity(path: Path) -> tuple:
    st = os.stat(path)
    return (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def load_manifest(path):
    """Return a fresh parsed manifest; no fields are removed or reinterpreted."""
    global _MEMO
    path = Path(path).expanduser().resolve()
    identity = _identity(path)
    with _LOCK:
        if _MEMO is None or _MEMO[0] != identity:
            with path.open("rb") as handle:
                parsed = json.loads(handle.read())
            _MEMO = (identity, parsed)
        return copy.deepcopy(_MEMO[1])
