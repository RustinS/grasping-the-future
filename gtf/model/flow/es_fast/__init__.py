"""Nearest-support search used by the energy score's support-penetration term.

``pen_below_support`` is called from ``gtf.model.flow.losses``; the optional Triton kernels live in
``_kernels`` and are loaded lazily, so CPU-only installations never import Triton.
"""

from .search import nearest_support, pen_below_support, reference_search

__all__ = ["nearest_support", "pen_below_support", "reference_search"]
