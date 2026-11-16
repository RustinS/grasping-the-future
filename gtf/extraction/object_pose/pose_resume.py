"""Whether a finished object-pose output still matches its inputs and the current request (resume / skip decisions)."""

import json
import numpy as np
from pathlib import Path
from gtf.extraction.object_pose.pose_cache import (
    OBJECT_POSE_PIPELINE_VERSION,
    _cache_binding_matches,
    _object_pose_data_and_stat_binding,
    _object_pose_done_matches,
    _output_artifact_matches,
)
from gtf.extraction.object_pose.object_depth import DEPTH_CACHE_METADATA_NAME, _depth_output_matches, _load_matching_depth_cache_contract
from gtf.extraction.object_pose.point_tracking import _cotracker_checkpoint_sha256
from gtf.extraction.object_pose.camera_poses import _camera_model_checkpoint_sha256, _try_import_megasam


def _object_pose_outputs_valid(pose_dir, n_frames, expected_artifacts, *, expected_ref_frame_idx):
    """Validate exact output bytes and the numerical payload contract."""

    pose_dir = Path(pose_dir)
    paths = {
        "tracks": pose_dir / "tracks.npz",
        "camera_poses": pose_dir / "camera_poses.npz",
        "object_poses": pose_dir / "object_poses.npz",
        "object_points": pose_dir / "object_points.npz",
    }
    if not isinstance(expected_artifacts, dict) or set(expected_artifacts) != set(paths):
        return False
    if any(not _output_artifact_matches(paths[key], expected_artifacts[key]) for key in paths):
        return False

    with np.load(paths["tracks"], allow_pickle=False) as archive:
        if not {"tracks", "visibility", "ref_frame_idx"}.issubset(archive.files):
            return False
        tracks = np.asarray(archive["tracks"])
        visibility = np.asarray(archive["visibility"])
        ref_frame_idx = int(np.asarray(archive["ref_frame_idx"]).item())
        if (
            tracks.ndim != 3
            or tracks.shape[0] != n_frames
            or tracks.shape[1] < 1
            or tracks.shape[2] != 2
            or visibility.shape != tracks.shape[:2]
            or not np.isfinite(tracks).all()
            or not 0 <= ref_frame_idx < n_frames
            or ref_frame_idx != int(expected_ref_frame_idx)
        ):
            return False

    with np.load(paths["camera_poses"], allow_pickle=False) as archive:
        if "cam_to_world" not in archive.files:
            return False
        camera = np.asarray(archive["cam_to_world"], dtype=np.float64)
        if camera.shape != (n_frames, 4, 4) or not np.isfinite(camera).all():
            return False
        rotations = camera[:, :3, :3]
        if (
            not np.allclose(
                camera[:, 3, :],
                np.asarray([0.0, 0.0, 0.0, 1.0]),
                atol=1e-5,
            )
            or not np.allclose(
                rotations.transpose(0, 2, 1) @ rotations,
                np.eye(3)[None],
                atol=2e-3,
            )
            or np.any(np.linalg.det(rotations) <= 0.0)
        ):
            return False

    with np.load(paths["object_poses"], allow_pickle=False) as archive:
        if not {"rotations", "translations", "reprojection_errors"}.issubset(archive.files):
            return False
        rotations = np.asarray(archive["rotations"], dtype=np.float64)
        translations = np.asarray(archive["translations"], dtype=np.float64)
        reprojection = np.asarray(archive["reprojection_errors"], dtype=np.float64)
        if (
            rotations.shape != (n_frames, 3, 3)
            or translations.shape != (n_frames, 3)
            or reprojection.shape != (n_frames,)
            or not np.isfinite(rotations).all()
            or not np.isfinite(translations).all()
            or np.isinf(reprojection).any()
            or not np.allclose(
                rotations.transpose(0, 2, 1) @ rotations,
                np.eye(3)[None],
                atol=2e-3,
            )
            or np.any(np.linalg.det(rotations) <= 0.0)
        ):
            return False

    with np.load(paths["object_points"], allow_pickle=False) as archive:
        if "object_points" not in archive.files:
            return False
        points = np.asarray(archive["object_points"])
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 3 or not np.isfinite(points).all():
            return False
    return True


def object_pose_metadata_inputs_match(segment_dir, pose_dir, metadata):
    """Validate the complete current object-pose cache without model loading."""

    try:
        if not isinstance(metadata, dict) or metadata.get("pipeline_version") != OBJECT_POSE_PIPELINE_VERSION:
            return False
        n_frames = int(metadata["num_frames"])
        resolution = metadata["resolution"]
        if not isinstance(resolution, list) or len(resolution) != 2 or any(int(value) < 1 for value in resolution):
            return False
        segment_dir = Path(segment_dir)
        pose_dir = Path(pose_dir)
        metadata_path = pose_dir / "metadata.json"
        if not _object_pose_done_matches(pose_dir / ".object_pose_done", metadata_path):
            return False
        if json.loads(metadata_path.read_text()) != metadata:
            return False
        depth_metadata_path = pose_dir / DEPTH_CACHE_METADATA_NAME
        depth_contract = json.loads(depth_metadata_path.read_text())
        if (
            not isinstance(depth_contract, dict)
            or depth_contract.get("format") != "object-pose-depth-cache-contract-v1"
            or depth_contract.get("pipeline_version") != OBJECT_POSE_PIPELINE_VERSION
            or not _object_pose_done_matches(pose_dir / ".depth_done", depth_metadata_path)
            or metadata.get("depth_input_binding") != depth_contract.get("input_binding")
            or metadata.get("depth_output_binding") != depth_contract.get("output_binding")
            or not _depth_output_matches(
                pose_dir / "depth",
                depth_contract.get("output_binding"),
                n_frames,
                (int(resolution[0]), int(resolution[1])),
            )
        ):
            return False
        expected_shared = metadata.get("shared_input_binding")
        expected_stat = metadata.get("shared_input_stat_binding")
        if not all(isinstance(value, dict) and isinstance(value.get("input_digest_sha256"), str) for value in (expected_shared, expected_stat)):
            return False
        # Recompute the authoritative content binding even when stat metadata
        # is unchanged; see pose_cache._output_artifact_matches.
        current_shared, _current_stable_stat = _object_pose_data_and_stat_binding(
            video_path=segment_dir / "video.mp4",
            masks_path=segment_dir / "SAM3/masks.npz",
            depth_dir=pose_dir / "depth",
            n_frames=n_frames,
            decoded_frame_shape=(
                n_frames,
                int(resolution[0]),
                int(resolution[1]),
                3,
            ),
        )
        if current_shared != expected_shared:
            return False

        expected_tracks = metadata.get("tracks_input_binding")
        expected_camera = metadata.get("camera_input_binding")
        expected_ba = metadata.get("ba_input_binding")
        if not all(isinstance(value, dict) and isinstance(value.get("input_digest_sha256"), str) for value in (expected_tracks, expected_camera, expected_ba)):
            return False
        shared_digest = current_shared["input_digest_sha256"]
        if (
            expected_tracks.get("shared_input_digest_sha256") != shared_digest
            or expected_tracks.get("tracking_mode") != metadata.get("tracking_mode")
            or expected_tracks.get("cotracker_grid_size") != metadata.get("cotracker_grid_size")
            or expected_tracks.get("reference_frame_index") != metadata.get("ref_frame_idx")
            or expected_camera.get("shared_input_digest_sha256") != shared_digest
            or expected_ba.get("shared_input_digest_sha256") != shared_digest
            or expected_ba.get("tracks_input_digest_sha256") != expected_tracks.get("input_digest_sha256")
            or expected_ba.get("camera_input_digest_sha256") != expected_camera.get("input_digest_sha256")
        ):
            return False
        expected_artifacts = metadata.get("output_artifact_bindings")
        if not _object_pose_outputs_valid(
            pose_dir,
            n_frames,
            expected_artifacts,
            expected_ref_frame_idx=expected_tracks.get("reference_frame_index"),
        ):
            return False
        if expected_ba.get("tracks_sha256") != expected_artifacts["tracks"].get("sha256") or expected_ba.get("camera_poses_sha256") != expected_artifacts["camera_poses"].get(
            "sha256"
        ):
            return False
        with np.load(pose_dir / "tracks.npz", allow_pickle=False) as archive:
            if not _cache_binding_matches(archive, "tracks", expected_tracks):
                return False

        with np.load(pose_dir / "camera_poses.npz", allow_pickle=False) as archive:
            if not _cache_binding_matches(archive, "camera", expected_camera):
                return False
        with np.load(pose_dir / "object_poses.npz", allow_pickle=False) as archive:
            if not _cache_binding_matches(archive, "ba", expected_ba):
                return False
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def object_pose_metadata_request_matches(
    segment_dir,
    pose_dir,
    metadata,
    *,
    tracking_mode,
    cotracker_grid_size,
    use_megasam,
    require_megasam=False,
):
    """Require a completed cache to implement the requested extraction run."""

    if not object_pose_metadata_inputs_match(segment_dir, pose_dir, metadata):
        return False
    try:
        n_frames = int(metadata["num_frames"])
        height, width = (int(value) for value in metadata["resolution"])
        depth_contract = _load_matching_depth_cache_contract(
            video_path=Path(segment_dir) / "video.mp4",
            depth_dir=Path(pose_dir) / "depth",
            pose_dir=pose_dir,
            sam3d_dir=Path(segment_dir) / "sam3d",
            n_frames=n_frames,
            decoded_frame_shape=(n_frames, height, width, 3),
        )
        if (
            depth_contract is None
            or metadata.get("depth_input_binding") != depth_contract.get("input_binding")
            or metadata.get("depth_output_binding") != depth_contract.get("output_binding")
        ):
            return False
        normalized_mode = str(tracking_mode).lower()
        grid_size = int(cotracker_grid_size)
        expected_tracks = metadata["tracks_input_binding"]
        expected_camera = metadata["camera_input_binding"]
        if (
            metadata.get("tracking_mode") != normalized_mode
            or metadata.get("cotracker_grid_size") != grid_size
            or expected_tracks.get("tracking_mode") != normalized_mode
            or expected_tracks.get("cotracker_grid_size") != grid_size
            or expected_tracks.get("cotracker_checkpoint_sha256") != _cotracker_checkpoint_sha256()
        ):
            return False

        megasam_available = bool(use_megasam) and _try_import_megasam()
        if require_megasam and not megasam_available:
            return False
        requested_camera_method = "megasam" if megasam_available else "pnp"
        if (
            metadata.get("camera_method") != requested_camera_method
            or expected_camera.get("camera_method") != requested_camera_method
            or expected_camera.get("camera_model_checkpoint_sha256") != _camera_model_checkpoint_sha256(requested_camera_method)
            or expected_camera.get("format") != "object-pose-camera-input-binding-v1"
        ):
            return False
    except (OSError, TypeError, ValueError, KeyError):
        return False
    return True
