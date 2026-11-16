"""SAM 3D Objects inference: the model, full reconstruction, and the pose-only sampler (single frame and batched)."""

import os
from contextlib import nullcontext
import torch
from gtf.paths import REPO_ROOT
from gtf.utils import rprint as print

SAM3D_REPO = os.path.join(str(REPO_ROOT), "third-party", "sam-3d-objects")
SAM3D_CHECKPOINT_TAG = "hf"


def build_sam3d_inference(checkpoint_dir=None, compile_model=False):
    """Build SAM3D Inference object (loaded once, reused across segments)."""
    from inference import Inference

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(SAM3D_REPO, "checkpoints", SAM3D_CHECKPOINT_TAG)

    config_path = os.path.join(checkpoint_dir, "pipeline.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing SAM3D config: {config_path}\n"
            "Download checkpoints with:\n"
            "  hf auth login\n"
            "  hf download --repo-type model "
            "--local-dir third-party/sam-3d-objects/checkpoints/hf-download facebook/sam-3d-objects\n"
            "  mv third-party/sam-3d-objects/checkpoints/hf-download/checkpoints "
            "third-party/sam-3d-objects/checkpoints/hf"
        )

    print(f"Loading SAM3D model from {config_path}...")
    inference = Inference(config_path, compile=compile_model)
    _condition_moge_on_known_fov(inference._pipeline)
    print("SAM3D model loaded successfully")
    return inference


def _condition_moge_on_known_fov(pipeline):
    """Opt-in camera calibration: EXTRACT_KNOWN_FOV_X_DEG conditions the SAM3D
    pipeline's MoGe depth model on a known horizontal FoV (degrees) instead of
    letting it estimate intrinsics. Unset (the default) is a no-op."""
    fov = os.environ.get("EXTRACT_KNOWN_FOV_X_DEG", "").strip()
    if not fov:
        return
    import functools

    model = pipeline.depth_model.model
    base = getattr(model, "_infer_unconditioned", None) or model.infer
    model._infer_unconditioned = base
    model.infer = functools.partial(base, fov_x=float(fov))
    print(f"    known-K: MoGe conditioned on fov_x={float(fov):.3f} deg")


def run_inference(inference_obj, image, mask, seed=42):
    """Run SAM3D (with layout post-processing) on a single (image, mask) pair. Returns output dict with gaussian, mesh,
    intrinsics."""
    rgba = inference_obj.merge_mask_to_rgba(image, mask.astype(bool))
    pipeline = inference_obj._pipeline
    output = pipeline.run(
        rgba,
        None,
        seed,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=False,
        with_layout_postprocess=True,
        use_vertex_color=True,
        stage1_inference_steps=20,
        stage2_inference_steps=20,
        decode_formats=["gaussian", "mesh"],
    )

    intrinsics = output.get("intrinsics")
    if intrinsics is None:
        intrinsics = pipeline.compute_pointmap(rgba, None)["intrinsics"]
        output["intrinsics"] = intrinsics

    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
    if not (0.1 < fx < 10 and 0.1 < fy < 10 and 0.3 < cx < 0.7 and 0.3 < cy < 0.7):
        print(f"    WARNING: unusual intrinsics fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

    return output


def run_stage1_pose(pipeline, rgba, seed=42, stage1_steps=20):
    """Depth + Stage 1 + pose decode. Returns (ss_return_dict, ss_input_dict, pointmap_dict)."""
    with pipeline.device:
        pointmap_dict = pipeline.compute_pointmap(rgba, None)
        ss_input_dict = pipeline.preprocess_image(rgba, pipeline.ss_preprocessor, pointmap=pointmap_dict["pointmap"])
        if seed is not None:
            torch.manual_seed(seed)
        ss_return_dict = pipeline.sample_sparse_structure(ss_input_dict, inference_steps=stage1_steps)
        pointmap_scale = ss_input_dict.get("pointmap_scale", None)
        pointmap_shift = ss_input_dict.get("pointmap_shift", None)
        ss_return_dict.update(pipeline.pose_decoder(ss_return_dict, scene_scale=pointmap_scale, scene_shift=pointmap_shift))
        ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
    return ss_return_dict, ss_input_dict, pointmap_dict


def _batch_tensor_dicts(items):
    """Concatenate the tensor fields shared by SAM3D preprocessor outputs."""
    keys = set.intersection(*(set(item) for item in items))
    return {key: torch.cat([item[key] for item in items], dim=0) for key in keys if all(torch.is_tensor(item[key]) for item in items)}


def _slice_batch_dict(values, index, batch_size):
    """Slice fields carrying a leading batch dimension; preserve shared metadata."""
    return {key: value[index : index + 1] if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size else value for key, value in values.items()}


def _repeated_seed_noise(generator, batch_size):
    """Make a batched diffusion call use the same seeded noise as singleton calls."""
    original = generator._generate_noise_tensor

    def generate(shape, x_device=None, device=None):
        target_device = x_device if x_device is not None else device
        if isinstance(shape, tuple) and shape and shape[0] == batch_size:
            single = original((1,) + tuple(shape[1:]), target_device)
            return single.expand(shape).clone()
        return original(shape, target_device)

    return original, generate


def _sample_pose_latent(pipeline, item, stage1_steps):
    """Run only the Stage 1 sparse-structure generator (stage1_steps steps, the pipeline's CFG strengths) on a batch."""
    generator = pipeline.models["ss_generator"]
    generator.no_shortcut = True
    generator.reverse_fn.strength = pipeline.ss_cfg_strength
    generator.reverse_fn.strength_pm = pipeline.ss_cfg_strength_pm
    previous_steps = generator.inference_steps
    generator.inference_steps = stage1_steps
    image = item["image"]
    batch_size = image.shape[0]
    autocast = torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype) if image.device.type == "cuda" else nullcontext()
    try:
        with torch.no_grad(), autocast:
            if pipeline.is_mm_dit():
                latent_shape = {
                    key: (
                        batch_size,
                        value.pos_emb.shape[0],
                        value.input_layer.in_features,
                    )
                    for key, value in generator.reverse_fn.backbone.latent_mapping.items()
                }
            else:
                latent_shape = (batch_size, 4096, 8)
            condition_args, condition_kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["ss_condition_embedder"],
                item,
                pipeline.ss_condition_input_mapping,
            )
            result = generator(
                latent_shape,
                image.device,
                *condition_args,
                **condition_kwargs,
            )
            if not pipeline.is_mm_dit():
                result = {"shape": result}
            return result
    finally:
        generator.inference_steps = previous_steps


def _decode_sparse_pose_metadata(pipeline, latent, batch_size):
    """Decode each item's sparse structure one at a time, as a singleton run does, so the pose scale gets the same downsample factor."""
    from sam3d_objects.pipeline.inference_utils import (
        downsample_sparse_structure,
        prune_sparse_structure,
    )

    shape = latent.get("shape")
    decoder = pipeline.models["ss_decoder"] if "ss_decoder" in pipeline.models else None
    if decoder is None or not torch.is_tensor(shape) or shape.ndim != 3 or tuple(shape.shape[1:]) != (4096, 8):
        return [{"downsample_factor": 1, "coords": None, "coords_original": None} for _ in range(batch_size)]

    autocast = torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype) if shape.device.type == "cuda" else nullcontext()
    decoded = []
    with torch.no_grad(), autocast:
        for index in range(batch_size):
            item_shape = shape[index : index + 1]
            occupancy = decoder(item_shape.permute(0, 2, 1).contiguous().view(1, 8, 16, 16, 16))
            coords_original = torch.argwhere(occupancy > 0)[:, [0, 2, 3, 4]].int()
            coords = coords_original
            if pipeline.downsample_ss_dist > 0:
                coords = prune_sparse_structure(
                    coords,
                    max_neighbor_axes_dist=pipeline.downsample_ss_dist,
                )
            coords, factor = downsample_sparse_structure(coords)
            decoded.append(
                {
                    "coords_original": coords_original,
                    "coords": coords,
                    "downsample_factor": factor,
                }
            )
    return decoded


def run_stage1_pose_batch(pipeline, rgba_batch, seed=42, stage1_steps=10):
    """Run Stage 1 diffusion on a batch of frames; every item gets the seeded noise of a singleton call."""
    if not rgba_batch:
        return []
    batch_size = len(rgba_batch)
    with pipeline.device:
        pointmaps = [pipeline.compute_pointmap(rgba, None) for rgba in rgba_batch]
        inputs = [
            pipeline.preprocess_image(
                rgba,
                pipeline.ss_preprocessor,
                pointmap=pointmap["pointmap"],
            )
            for rgba, pointmap in zip(rgba_batch, pointmaps)
        ]
        combined = _batch_tensor_dicts(inputs)
        generator = pipeline.models["ss_generator"]
        original_noise, repeated_noise = _repeated_seed_noise(generator, batch_size)
        generator._generate_noise_tensor = repeated_noise
        try:
            if seed is not None:
                torch.manual_seed(seed)
            batched = _sample_pose_latent(pipeline, combined, stage1_steps)
        finally:
            generator._generate_noise_tensor = original_noise

        sparse_metadata = _decode_sparse_pose_metadata(
            pipeline,
            batched,
            batch_size,
        )
        outputs = []
        for index, (item, pointmap) in enumerate(zip(inputs, pointmaps)):
            output = _slice_batch_dict(batched, index, batch_size)
            output.update(sparse_metadata[index])
            output.update(
                pipeline.pose_decoder(
                    output,
                    scene_scale=item.get("pointmap_scale"),
                    scene_shift=item.get("pointmap_shift"),
                )
            )
            output["scale"] = output["scale"] * output["downsample_factor"]
            output["intrinsics"] = pointmap["intrinsics"]
            output["moge_depth"] = pointmap.get("moge_depth")
            output["moge_mask"] = pointmap.get("moge_mask")
            outputs.append(output)
    return outputs


def run_pose_only_inference(pipeline, rgba, seed=42, stage1_steps=20):
    """Stage 1 pose estimation of one frame, with its intrinsics and MoGe outputs."""
    ss_return_dict, _, pointmap_dict = run_stage1_pose(pipeline, rgba, seed, stage1_steps)
    return {
        **ss_return_dict,
        "intrinsics": pointmap_dict["intrinsics"],
        "moge_depth": pointmap_dict.get("moge_depth"),
        "moge_mask": pointmap_dict.get("moge_mask"),
    }
