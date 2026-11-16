"""State tokenization: canonical.npz fields ↔ model token format.

Converts between axis-angle rotations (canonical.npz) and 6D rotation representation
(model tokens). Stateless utility — no learnable parameters, not an nn.Module.

Pipeline position:
  canonical.npz → gravity_align (axis-angle) → tokenize (→ 6D) → relation features → contact labels

State token layouts:
  left_state:  [orient_6d(6) | cam_t(3) | pose_6d(90)] = 99D
  right_state: [orient_6d(6) | cam_t(3) | pose_6d(90)] = 99D
  obj_state:   [rotation_6d(6) | translation(3)]        = 9D
"""

import torch
from torch import Tensor

from gtf.model.repr.rotation import axis_angle_to_6d, rotation_6d_to_axis_angle

# Layout slices of the state tokens.
# Hand state: [orient_6d(6) | cam_t(3) | pose_6d(90)] = 99D
HAND_ORIENT_SLICE = slice(0, 6)
HAND_CAM_T_SLICE = slice(6, 9)
HAND_POSE_SLICE = slice(9, 99)

# Object state: [rotation_6d(6) | translation(3)] = 9D
OBJ_ROTATION_SLICE = slice(0, 6)
OBJ_TRANSLATION_SLICE = slice(6, 9)


class StateTokenizer:
    """Converts canonical.npz fields ↔ model token format.

    Stateless: no parameters, no state. All methods are pure functions on dicts.
    Not an nn.Module — no need for device management or serialization.
    """

    def encode(self, data_dict: dict[str, Tensor], valid_frames: Tensor | None = None) -> dict[str, Tensor]:
        """Encode canonical.npz fields to model token format.

        Converts axis-angle rotations to 6D, concatenates into per-token-type
        state vectors, and constructs the presence mask.

        Does NOT compute relation features — those are a separate step.

        Args:
            data_dict: Dict with canonical.npz field names (after gravity alignment):
                - hand_{side}_orient: (T, 3) axis-angle, or None/absent
                - hand_{side}_pose: (T, 45) axis-angle, or None/absent
                - hand_{side}_cam_t: (T, 3) meters, or None/absent
                - hand_{side}_valid: (T,) bool, or None/absent
                - obj_rotation: (T, 3) axis-angle
                - obj_translation: (T, 3) meters
            valid_frames: (T,) bool, optional. True = real frame, False = padding.
                When provided, padded frames get all-False in presence_mask
                (including object slot). When None, all frames treated as real.

        Returns:
            Dict with:
                - left_state: (T, 99) [orient_6d | cam_t | pose_6d]
                - right_state: (T, 99) same layout
                - obj_state: (T, 9) [rotation_6d | translation]
                - presence_mask: (T, 5) bool [left_state, right_state, obj_state, left_rel, right_rel]
        """
        T = data_dict["obj_rotation"].shape[0]
        device = data_dict["obj_rotation"].device

        result = {}

        # --- Encode each hand ---
        for side, state_slot, rel_slot in [("left", 0, 3), ("right", 1, 4)]:
            orient_key = f"hand_{side}_orient"
            pose_key = f"hand_{side}_pose"
            cam_t_key = f"hand_{side}_cam_t"
            valid_key = f"hand_{side}_valid"

            orient = data_dict.get(orient_key)
            pose = data_dict.get(pose_key)
            cam_t = data_dict.get(cam_t_key)
            hand_valid = data_dict.get(valid_key)

            if orient is not None and hand_valid is not None and hand_valid.any():
                # Convert orient: axis-angle (T, 3) → 6D (T, 6)
                orient_6d = axis_angle_to_6d(orient)  # (T, 6)

                # Convert pose: axis-angle (T, 45) → per-joint (T, 15, 3) → 6D (T, 90)
                pose_per_joint = pose.reshape(T, 15, 3)  # (T, 15, 3)
                pose_6d = axis_angle_to_6d(pose_per_joint).reshape(T, 90)  # (T, 90)

                # Concatenate: [orient_6d(6) | cam_t(3) | pose_6d(90)] = 99D
                state = torch.cat([orient_6d, cam_t, pose_6d], dim=-1)  # (T, 99)

                # Zero out invalid frames
                state[~hand_valid] = 0.0
            else:
                state = torch.zeros(T, 99, device=device, dtype=data_dict["obj_rotation"].dtype)
                hand_valid = torch.zeros(T, dtype=torch.bool, device=device)

            state_name = f"{side}_state"
            result[state_name] = state

        # --- Encode object ---
        obj_rotation_6d = axis_angle_to_6d(data_dict["obj_rotation"])  # (T, 6)
        obj_state = torch.cat([obj_rotation_6d, data_dict["obj_translation"]], dim=-1)  # (T, 9)
        result["obj_state"] = obj_state

        # --- Build presence mask ---
        presence_mask = torch.zeros(T, 5, dtype=torch.bool, device=device)

        left_valid = data_dict.get("hand_left_valid")
        right_valid = data_dict.get("hand_right_valid")

        if left_valid is not None:
            presence_mask[:, 0] = left_valid  # left_state
            presence_mask[:, 3] = left_valid  # left_rel mirrors hand
        if right_valid is not None:
            presence_mask[:, 1] = right_valid  # right_state
            presence_mask[:, 4] = right_valid  # right_rel mirrors hand

        # Object always present for real frames
        presence_mask[:, 2] = True

        # Padded frames: all-False across all 5 slots (including object)
        if valid_frames is not None:
            presence_mask &= valid_frames.unsqueeze(-1)  # (T, 1) broadcast

        result["presence_mask"] = presence_mask
        return result

    def decode(self, tokens_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Decode model tokens back to canonical.npz field format.

        Converts 6D rotations back to axis-angle. Handles both unbatched (T, D)
        and batched (B, T, D) inputs via ... indexing.

        Note: rotation_6d_to_axis_angle on zero vectors produces NaN
        (normalize zero → NaN). Caller should decode only present tokens.

        Args:
            tokens_dict: Dict with:
                - left_state: (*, 99) or absent
                - right_state: (*, 99) or absent
                - obj_state: (*, 9)

        Returns:
            Dict with canonical field names:
                - hand_{side}_orient: (*, 3) axis-angle
                - hand_{side}_cam_t: (*, 3) meters
                - hand_{side}_pose: (*, 45) axis-angle
                - obj_rotation: (*, 3) axis-angle
                - obj_translation: (*, 3) meters
        """
        result = {}

        for side in ("left", "right"):
            state_key = f"{side}_state"
            if state_key not in tokens_dict:
                continue

            state = tokens_dict[state_key]
            batch_shape = state.shape[:-1]  # everything before the 99 dim

            # Extract slices
            orient_6d = state[..., HAND_ORIENT_SLICE]  # (*, 6)
            cam_t = state[..., HAND_CAM_T_SLICE]  # (*, 3)
            pose_6d = state[..., HAND_POSE_SLICE]  # (*, 90)

            # Convert orient: 6D → axis-angle
            result[f"hand_{side}_orient"] = rotation_6d_to_axis_angle(orient_6d)  # (*, 3)
            result[f"hand_{side}_cam_t"] = cam_t

            # Convert pose: 6D → axis-angle per joint, then flatten
            pose_6d_per_joint = pose_6d.reshape(*batch_shape, 15, 6)  # (*, 15, 6)
            pose_aa_per_joint = rotation_6d_to_axis_angle(pose_6d_per_joint)  # (*, 15, 3)
            result[f"hand_{side}_pose"] = pose_aa_per_joint.reshape(*batch_shape, 45)  # (*, 45)

        # Object
        obj_state = tokens_dict["obj_state"]
        result["obj_rotation"] = rotation_6d_to_axis_angle(obj_state[..., OBJ_ROTATION_SLICE])
        result["obj_translation"] = obj_state[..., OBJ_TRANSLATION_SLICE]

        return result


def center_translations(tokens_dict: dict[str, Tensor], origin: Tensor) -> dict[str, Tensor]:
    """Center translation components within tokenized state vectors.

    Subtracts origin from the translation slices of the state tokens.

    Args:
        tokens_dict: Dict with left_state (*, 99), right_state (*, 99),
            obj_state (*, 9), and any other keys (passed through unchanged).
        origin: (*, 3) or (B, 3) physical-frame origin to subtract. Must broadcast
            with the translation slices (typically needs unsqueeze for T dim).

    The LEFT hand's cam_t lives in the MIRRORED right-model frame (MANO mirrors the
    whole left FK output, M = diag(-1,1,1)), so the left slice subtracts ``M·o``:
    the physical left hand then moves by exactly ``o`` like every other entity —
    one shared centered physical frame. Subtracting the raw ``o`` would displace
    the physical left hand by ``(I−M)o = [2·o_x, 0, 0]`` relative to the object.
    Un-centering passes ``-origin`` through this same function.

    Returns:
        New dict with centered translation slices. Rotation and pose slices unchanged.
        Non-state keys passed through unchanged.
    """
    result = dict(tokens_dict)

    # Reshape origin for broadcasting over T dimension
    # origin: (B, 3) → (B, 1, 3) for broadcasting with (B, T, 3)
    if origin.ndim >= 1:
        origin_broadcast = origin.unsqueeze(-2)  # (..., 1, 3)
    else:
        origin_broadcast = origin

    for key in ("left_state", "right_state"):
        if key in result and result[key] is not None:
            org = origin_broadcast
            if key == "left_state":
                flip = torch.tensor([-1.0, 1.0, 1.0], device=origin.device, dtype=origin.dtype)
                org = origin_broadcast * flip  # M·o — mirrored-frame image of the physical origin
            result[key] = result[key].clone()
            result[key][..., HAND_CAM_T_SLICE] = result[key][..., HAND_CAM_T_SLICE] - org

    if "obj_state" in result and result["obj_state"] is not None:
        result["obj_state"] = result["obj_state"].clone()
        result["obj_state"][..., OBJ_TRANSLATION_SLICE] = result["obj_state"][..., OBJ_TRANSLATION_SLICE] - origin_broadcast

    return result


# Slot-to-token-key and slot-to-translation-slice mapping for center_tokens_at_present
_SLOT_TRANSLATION_MAP = {
    0: ("left_state", HAND_CAM_T_SLICE),
    1: ("right_state", HAND_CAM_T_SLICE),
    2: ("obj_state", OBJ_TRANSLATION_SLICE),
}


def center_tokens_at_present(
    tokens: dict[str, Tensor],
    obs_mask: Tensor,
    presence_mask: Tensor,
    present_frames: Tensor,
) -> tuple[dict[str, Tensor], Tensor]:
    """Center tokenized data at the present frame's manipulation midpoint.

    The origin is the midpoint of the hand centroid and the object position at the
    present frame, over the entities that are present and observed there (all present
    entities in a fully masked window), and is subtracted with center_translations.

    Args:
        tokens: Dict with left_state (B,T,99), right_state (B,T,99), obj_state (B,T,9),
            and any other keys (passed through). Also works unbatched (T, D).
        obs_mask: (B, T, 5) or (T, 5) bool. True = observed.
        presence_mask: (B, T, 5) or (T, 5) bool. True = entity present.
        present_frames: (B,) present-frame index of each sample, (1,) when unbatched
            (present_anchor.select_present_frames).

    The LEFT hand's contribution to the origin uses its PHYSICAL position
    ``M·cam_t`` (raw left cam_t is mirrored-frame) and the left subtraction is
    ``M·o`` (see center_translations), so every entity's PHYSICAL position shifts
    by the same ``o`` and the returned origin is a physical-frame point (what scene
    centering and the decoded-geometry losses expect).

    Returns:
        (centered_tokens, origin): Centered token dict and origin(s).
            origin: (B, 3) or (3,) depending on input batching.
    """
    unbatched = obs_mask.ndim == 2
    if unbatched:
        obs_mask = obs_mask.unsqueeze(0)
        presence_mask = presence_mask.unsqueeze(0)
        tokens = {k: (v.unsqueeze(0) if isinstance(v, Tensor) and v.ndim >= 1 else v) for k, v in tokens.items()}

    B = obs_mask.shape[0]
    device = obs_mask.device
    origins = []

    for b in range(B):
        present_frame = int(present_frames[b].item())

        # Fully masked window: no observed state slots at all
        is_full_mask = not obs_mask[b, :, :3].any().item()

        hand_positions = []
        obj_position = None

        for slot, (token_key, trans_slice) in _SLOT_TRANSLATION_MAP.items():
            if token_key not in tokens or tokens[token_key] is None:
                continue
            is_present = presence_mask[b, present_frame, slot].item()
            is_observed = obs_mask[b, present_frame, slot].item()
            if not is_present:
                continue
            if not is_observed and not is_full_mask:
                continue

            pos = tokens[token_key][b, present_frame, trans_slice]  # (3,)
            if slot in (0, 1):  # hand slots
                if slot == 0:
                    # Physical left position: raw left cam_t is mirrored-frame.
                    pos = pos * torch.tensor([-1.0, 1.0, 1.0], device=pos.device, dtype=pos.dtype)
                hand_positions.append(pos)
            else:  # slot 2 = object
                obj_position = pos

        # Midpoint(hand_centroid, obj_centroid) — not flat average
        hand_centroid = torch.stack(hand_positions).mean(dim=0) if hand_positions else None
        if hand_centroid is not None and obj_position is not None:
            origin = (hand_centroid + obj_position) / 2.0
        elif hand_centroid is not None:
            origin = hand_centroid
        elif obj_position is not None:
            origin = obj_position
        else:
            origin = torch.zeros(3, device=device)
        origins.append(origin)

    origin = torch.stack(origins, dim=0)  # (B, 3)
    centered = center_translations(tokens, origin)

    if unbatched:
        origin = origin.squeeze(0)
        centered = {k: (v.squeeze(0) if isinstance(v, Tensor) and v.ndim >= 1 else v) for k, v in centered.items()}

    return centered, origin
