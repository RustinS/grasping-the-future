"""The two extraction chains.

Everything up to SAM 3D (validation, WiLoR, pointing, SAM 3, SAM 3D, point tracking) is shared. A chain is the set of
options the later stages run with; this file is the only place they are defined.

    stage               production                               upgraded
    object pose (BA)    reprojection + rotation smoothness       + depth-relief term, no rotation smoothness
    hand masks          not used                                 SAM 3 "hand" at the keyframes (RGB evidence for the optimizer)
    optimizer           hand depth from MoGe, object scale       hand depth from its apparent size (scene depth per frame for
                        from MoGe, contact from proximity        small hands), object follows the hand where it is held, hand
                                                                 slid along its viewing ray onto the object where the video
                                                                 shows contact, contact / hold terms gated by the image,
                                                                 WiLoR detections without an RGB hand dropped, BA rotation kept
    stage 2             denoiser bias + filters                  filters only (no bias)
    export refiner      6DoF silhouette fit, hard accept         image-plane slide only, soft accept, per-run do-no-harm
    export smoother     off                                      on (trajectory_smoother.py)

`production` is the default (every option at its default value) and the chain the training data and the benchmark
references were extracted with. `upgraded` places the hands closer to the video's hands, with less jitter, steadier object
rotation and tighter grasps. The quality-gate thresholds are calibrated on `production` outputs.

Select a chain with `--chain NAME` on the orchestrator or any stage script, or `EXTRACTION_CHAIN=NAME` for a whole run.
A segment directory belongs to one chain (`claim`): skip-if-exists would otherwise mix two chains' outputs.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

ENV = "EXTRACTION_CHAIN"
MARKER = ".extraction_chain"


@dataclass(frozen=True)
class ObjectBA:
    """Object bundle adjustment (run_object_pose.py; the relief term lives in object_pose_relief.py)."""

    relief_weight: float = 0.0  # weight of the depth-relief term; 0 = off
    relief_delta: float = 0.15  # its Huber scale, as a fraction of the object radius
    rotation_smooth: bool = True  # first-order rotation smoothness

    def spec(self):
        """Record bound into the BA cache digest; None for the default BA (nothing is added to the digest)."""
        return None if self == ObjectBA() else {"version": 1, **asdict(self)}


@dataclass(frozen=True)
class ObjectRefine:
    """Export object-pose refiner (object_pose_refine.py)."""

    fit: str = "6dof"  # what the per-keyframe silhouette fit may change: "6dof", or "lateral" = slide in the image plane
    rotation: bool = True  # apply the fitted rotation correction (False: translation only)
    trans_radius: int | None = None  # moving-average radius of the translation correction; None = the refiner's smooth_radius
    soft_accept: int = 0  # > 0: the accept / grasp-force decision is a weight smoothed over this radius (no on/off steps)
    run_guard: bool = False  # soft path: a grasp run is forced only if that does not lower its mean mask IoU
    front_silhouette: bool = False  # rasterize camera-facing triangles only


@dataclass(frozen=True, eq=False)
class Chain:
    name: str
    object_ba: ObjectBA = ObjectBA()
    hand_masks: bool = False  # run_hand_masks.py before the optimizer
    optimizer: Mapping = field(default_factory=lambda: MappingProxyType({}))  # overrides of optimize_mano config keys
    refine_bias: bool = True  # stage-2 denoiser bias
    object_refine: ObjectRefine = ObjectRefine()
    smooth: bool = False  # export smoother fitter


PRODUCTION = Chain("production")

UPGRADED = Chain(
    "upgraded",
    object_ba=ObjectBA(relief_weight=5.0, rotation_smooth=False),
    hand_masks=True,
    optimizer=MappingProxyType(
        {
            "hand_depth_source": "size_scene",
            "scale_anchor": "contact_follow",
            "follow_depth": "local",
            "size_fallback": "hands",
            "hand_depth_correction": "ray",
            "wrist_prior_mode": "ray",
            "contact_depth": "ray",
            "hand_mask_veto": True,
            "image_contact_gate": True,
            "image_carry_gate": True,
            "w_data_obj_rot": 30.0,
            "w_hold_obj_rot": 0.0,
        }
    ),
    refine_bias=False,
    object_refine=ObjectRefine(fit="lateral", rotation=False, trans_radius=3, soft_accept=3, run_guard=True, front_silhouette=True),
    smooth=True,
)

CHAINS = {c.name: c for c in (PRODUCTION, UPGRADED)}


class ChainMismatch(RuntimeError):
    pass


def get(chain=None) -> Chain:
    """A Chain from a Chain, a name, or None (= $EXTRACTION_CHAIN, else production)."""
    if hasattr(chain, "object_refine"):  # a Chain (also one created through another import path of this module)
        return chain
    name = chain or os.environ.get(ENV) or "production"
    if name not in CHAINS:
        raise KeyError(f"unknown extraction chain {name!r}; known: {', '.join(CHAINS)}")
    return CHAINS[name]


def add_argument(parser):
    parser.add_argument("--chain", choices=list(CHAINS), default=None, help=f"extraction chain (default: ${ENV}, else production); see gtf/extraction/chains.py")


def claim(segment_dir, chain) -> Chain:
    """Bind a segment directory to a chain before its first chain-dependent stage (object pose BA onward).

    No marker = production (production never writes one). Another chain writes `.extraction_chain` into a directory that
    has no BA / optimizer output yet. ChainMismatch if the directory already holds a different chain's outputs.
    """
    chain = get(chain)
    segment_dir = Path(segment_dir)
    marker = segment_dir / MARKER
    held = marker.read_text().strip() if marker.exists() else "production"
    if held == chain.name:
        return chain
    has_outputs = (segment_dir / "object_pose" / "object_poses.npz").exists() or (segment_dir / "optimized_mano" / "optimized.npz").exists()
    if marker.exists() or has_outputs:
        raise ChainMismatch(
            f"{segment_dir.name} holds {held}-chain outputs; the {chain.name} chain needs its own results directory (ACTION100M_RESULTS_DIR)"
        )
    if segment_dir.is_dir():
        tmp = marker.with_name(f"{MARKER}.{os.getpid()}.tmp")
        tmp.write_text(chain.name + "\n")
        os.replace(tmp, marker)
    return chain
