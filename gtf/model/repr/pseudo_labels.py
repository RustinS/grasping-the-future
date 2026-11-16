"""Contact pseudo-labels for the auxiliary contact head.

Binary contact from fingertip distances (d < 5 mm), the target of OutputHeads' contact_logits (B,T,2,5).
Computed from the relation features when the dataset builds a segment.
"""

from torch import Tensor

# Contact threshold in meters (5mm)
CONTACT_THRESHOLD = 0.005


def compute_contact_labels(
    fingertip_distances: Tensor,
    hand_valid: Tensor | None = None,
) -> Tensor:
    """Compute binary contact labels from fingertip-object distances.

    Args:
        fingertip_distances: (T, 2, 5) per-hand per-fingertip minimum distances
            to object surface, in meters. Dim 1: 0=left, 1=right. Dim 2: 5 fingertips
            matching FINGERTIP_INDICES = [4, 8, 12, 16, 20] from mano.py.
        hand_valid: (T, 2) bool, optional. Per-hand per-frame validity.
            Critical for absent hands: zeroed relation features produce
            fingertip_dists=0, which would falsely indicate contact.
            When provided, contacts for invalid hand-frames are forced False.

    Returns:
        (T, 2, 5) bool. True where fingertip distance < 5mm AND hand is valid.
    """
    contacts = fingertip_distances < CONTACT_THRESHOLD
    if hand_valid is not None:
        contacts = contacts & hand_valid.unsqueeze(-1)  # (T, 2, 1) broadcast over 5 fingers
    return contacts
