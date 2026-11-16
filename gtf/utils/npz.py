"""NPZ helpers used by high-volume extraction stages."""

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat


def savez_compressed_level(
    path,
    *,
    compresslevel=1,
    allow_pickle=True,
    return_sha256=False,
    return_payload=False,
    **arrays,
):
    """Atomically write an NPZ with an explicit deflate level.

    NumPy's ``savez_compressed`` fixes zlib at level 6. Level 1 loads to the
    same arrays while substantially reducing depth-cache write time.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        # With return_sha256 / return_payload the (typically one-frame) NPZ is
        # built in memory, so the digest covers the exact bytes written without
        # reading the file back. Otherwise the archive streams straight to the
        # temporary file (large NPZs).
        target = io.BytesIO() if return_sha256 or return_payload else temporary
        with zipfile.ZipFile(
            target,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=compresslevel,
            allowZip64=True,
        ) as archive:
            for key, value in arrays.items():
                with archive.open(f"{key}.npy", mode="w", force_zip64=True) as handle:
                    npformat.write_array(
                        handle,
                        np.asanyarray(value),
                        allow_pickle=allow_pickle,
                    )
        binding = None
        if return_sha256 or return_payload:
            payload = target.getvalue()
            binding = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            if return_payload:
                binding["payload"] = payload
            with temporary.open("wb") as stream:
                stream.write(payload)
        os.replace(temporary, path)
        return binding
    finally:
        if temporary.exists():
            temporary.unlink()
