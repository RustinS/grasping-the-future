"""GTF configuration using OmegaConf structured configs.

Load with:
    cfg = OmegaConf.structured(GTFConfig)
or merge with YAML:
    cfg = OmegaConf.merge(OmegaConf.structured(GTFConfig), OmegaConf.load("configs/main.yaml"))
"""

from dataclasses import dataclass, field


@dataclass
class StateDimsConfig:
    left_state: int = 99
    right_state: int = 99
    obj_state: int = 9
    left_rel: int = 14
    right_rel: int = 14


@dataclass
class ModelConfig:
    """Transformer, conditioning and scene-encoder sizes of the GTF model."""

    d_model: int = 512
    n_layers: int = 12  # every global_block_interval-th block is global, the rest factorized
    n_heads: int = 8
    d_ff: int = 2048
    n_token_types: int = 5
    state_dims: StateDimsConfig = field(default_factory=StateDimsConfig)
    # Object shape: a cached frozen Uni3D-g embedding (B, 1024) projected to this many tokens.
    n_obj_shape_tokens: int = 4
    n_scene_global_tokens: int = 8
    confidence_dim: int = 7
    max_seq_len: int = 150
    dropout: float = 0.1
    use_flash_attention: bool = True
    global_block_interval: int = 4

    # Scene encoder: LitePT-S (ScanNet-pretrained body, frozen; input stem, output
    # projection and global query branch trainable) over per-segment scene point clouds.
    scene_in_channels: int = 44  # 12 geometric + 32 semantic channels
    scene_feature_dim: int = 64  # per-point feature width after the output projection
    scene_semantic_dim: int = 32  # DINOv3 PCA width
    scene_features_dropout: float = 0.2  # training-time per-segment semantic-channel dropout
    litept_weights_path: str = "checkpoints/litept_scannet/model_best.pth"

    # Reference cadence of the timing conditioning: log(dt * dt_ref_fps) is 0 here, and
    # the temporal position embedding counts elapsed time in units of 1 / dt_ref_fps.
    dt_ref_fps: float = 30.0


@dataclass
class TrainingConfig:
    batch_size: int = 32  # per GPU
    lr: float = 1e-4
    weight_decay: float = 0.01  # AdamW, on >= 2-D weights outside embeddings and norms
    max_steps: int = 40000
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    lr_min_ratio: float = 0.1  # cosine schedule floor as a fraction of the peak lr
    window_size: int = 150  # window tensor length in frames (real frames + padding)

    # Real-time windows: each segment's window spans round(window_seconds * effective_fps) frames (at most
    # window_size, padded to it), and windows start every round(stride_seconds * effective_fps) frames.
    window_seconds: float = 5.0
    stride_seconds: float = 2.5

    # Global SE(3) augmentation: with probability aug_global_prob, one gaussian yaw about gravity (radians) and
    # one gaussian translation (meters) applied identically to hands, object, scene and support plane.
    aug_global_prob: float = 0.5
    aug_global_yaw_sigma: float = 0.35
    aug_global_trans_sigma: float = 0.0

    # Temporal stride augmentation (train split): with probability aug_stride_prob a window is re-sliced at
    # stride s ~ U{1..max_stride}, max_stride = clamp(round(effective_fps / aug_stride_target_hz), 1,
    # aug_stride_max): the same motion at a coarser cadence over the same real-time span.
    aug_stride_prob: float = 0.5
    aug_stride_target_hz: float = 10.0
    aug_stride_max: int = 6

    # Seeds model init, the per-rank training RNG, the DataLoader workers, validation and the energy-score gate.
    seed: int = 42


@dataclass
class LossConfig:
    """Loss weights and loss parameters; the defaults are the main model's.

    Weighted terms: flow matching (fm_component_weights), auxiliary contact BCE, energy score,
    support penetration, decoded x0 position/rotation/articulation/smoothness, MANO anatomy and
    pose prior, decoded contact, object-frame attachment.
    """

    # Flow-matching component weights over {hand_trans, hand_rot, hand_pose, obj_trans, obj_rot,
    # relation}: per window, a weighted mean over the active groups (missing keys weigh 1).
    fm_component_weights: dict[str, float] = field(
        default_factory=lambda: {
            "hand_trans": 0.88390513,
            "hand_rot": 0.73826554,
            "hand_pose": 1.1122007,
            "obj_trans": 0.72983978,
            "obj_rot": 0.30580312,
            "relation": 0.18482417,
        }
    )
    # Per-finger contact BCE of the contact head (ramped in by GTFConfig.loss_schedule).
    lambda_aux_contact: float = 0.1

    # Energy score on es_samples rollouts of es_steps Euler steps from noise, applied on a seeded
    # fraction es_p of training steps (gtf/model/flow/losses.py:loss_energy_score). Statistic: each
    # entity's displacement from the present frame at es_path_points intermediate frames and the
    # endpoint, relative hand-object displacement (x es_rel_weight) and endpoint relative rotations
    # (x es_rot_weight); a dense all-frame translation score is added with weight es_dense_weight.
    lambda_energy_score: float = 1.0
    es_samples: int = 3
    es_steps: int = 6
    es_p: float = 0.25
    es_rel_weight: float = 1.0
    es_path_points: int = 2
    es_rot_weight: float = 0.3
    es_dense_weight: float = 0.5

    # Support-surface penetration of the energy-score samples: object vertices more than
    # support_pen_margin_mm below their nearest horizontal scene point, within support_pen_band_mm
    # laterally of it; per frame only the support_pen_topk deepest vertices count (0 = all).
    # Applied to windows from support_pen_sources only.
    lambda_support_pen: float = 1.0
    support_pen_margin_mm: float = 10.0
    support_pen_band_mm: float = 40.0
    support_pen_topk: int = 8
    support_pen_sources: list[str] = field(default_factory=lambda: ["dexycb", "hot3d", "oakink2", "taco", "hocap", "ho3d", "show3d"])

    # Decoded x0 losses (gtf/training/trainer.py): x0_hat = x_t + (1 - t) v is integrated through the
    # delta codec and compared with the clean chain on every frame whose decoded value depends on
    # a generated slot.
    lambda_x0_pos: float = 1.0  # wrist and object translation error (m^2)
    lambda_x0_rot: float = 0.368271  # wrist and object rotation error (theta^2 / 2)
    lambda_x0_artic: float = 0.42243744  # finger joint rotation error (theta^2 / 2)
    lambda_smooth_decoded: float = 0.00032034264  # decoded acceleration^2 + jerk^2 above the clean chain's

    # Decoded MANO anatomy: swing/twist limit-violation energy above the clean trajectory's.
    lambda_anatomy: float = 0.003353271
    anatomy_twist_mcp_deg: float = 25.0
    anatomy_twist_pip_deg: float = 15.0
    anatomy_swing_deg: float = 135.0
    # Standardized MANO pose-prior energy above the clean target's; the floor bounds the
    # reference normalization's denominator (in standardized MANO coefficients).
    lambda_mano_prior: float = 0.00049797137
    mano_prior_reference_floor: float = 1.0

    # Decoded contact: |d_decoded - d_GT| of fingertip-to-surface distances, both clamped at
    # dcontact_clamp_m (m), on up to dcontact_max_frames generated frames per (window, hand) (0 = all).
    lambda_dcontact: float = 0.33456092
    dcontact_max_frames: int = 16
    dcontact_clamp_m: float = 0.10

    # Decoded object-frame attachment over GT-contact runs (gtf/model/flow/losses.py:loss_attachment).
    lambda_attach_pos: float = 0.17838569  # wrist + fingertip position in the object frame
    lambda_attach_rot: float = 0.040579505  # hand rotation in the object frame (chordal)
    lambda_attach_vel: float = 0.12129669  # object-frame relative velocity over adjacent contact frames
    attach_contact_thr: float = 0.005  # GT min fingertip distance defining contact (m)
    attach_max_frames: int = 48  # frame budget per (window, hand); 0 = all contact frames
    attach_min_run: int = 2  # contact runs shorter than this are ignored


@dataclass
class LossRampConfig:
    """Linear ramp of one loss weight (GTFConfig.loss_schedule).

    Before start_step returns start_val; at and after end_step returns end_val; linear in between.
    """

    start_step: int = 0
    end_step: int = 0
    start_val: float = 0.0
    end_val: float = 0.0


@dataclass
class MaskingRegimesConfig:
    """Sampling weights of the training masking regimes (see gtf/model/flow/masking.py)."""

    future_prediction: float = 0.19
    hand_to_object: float = 0.19
    object_to_hand: float = 0.19
    random_block: float = 0.05
    goal_infilling: float = 0.19
    # Endpoints observed, one contiguous all-entity gap masked (the temporal_infilling task).
    temporal_infilling: float = 0.19


@dataclass
class MaskingConfig:
    regimes: MaskingRegimesConfig = field(default_factory=MaskingRegimesConfig)


@dataclass
class DataConfig:
    manifest_path: str = "datasets/canonical_manifest.json"
    # Window-sampled normalization statistics (python -m gtf.data.compute_stats).
    stats_path: str = "datasets/normalization_stats.pt"
    # Frozen Uni3D object embeddings, one row per segment (gtf/data/uni3d_cache.py).
    # The artifact holds the bf16-autocast output (training) and the fp32 output
    # (evaluation without AMP).
    uni3d_cache_path: str = "datasets/uni3d_embeddings.pt"
    # Byte cap of the per-process LRU cache of preprocessed segments (0 = off).
    segment_cache_max_bytes: int = 8 << 30

    # Weighted window sampler (gtf/data/sampler.py).
    dataset_alpha: float = 0.25  # source mass exponent: a source with N windows gets mass N ** alpha
    tier_boost: dict[str, float] = field(default_factory=lambda: {"gt_mocap": 3.0, "gt_fitted": 2.0, "gt_landmark_fitted": 1.5, "extracted": 1.0})
    # Action100M share of the sampling mass, piecewise-linear in the optimizer
    # step: [[step, mass], ...]. Weight ratios inside each pool are preserved.
    sampler_action100m_mass_knots: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.23476745351278497], [20000.0, 0.23476745351278497], [32000.0, 0.02], [40000.0, 0.02]]
    )
    # Target share of left-hand valid frames among all sampled hand frames (0 = no balancing).
    hand_balance_target: float = 0.5
    # Cap on the left-window boost used to reach hand_balance_target.
    hand_balance_max_boost: float = 10.0


@dataclass
class TrainerConfig:
    """Lightning Trainer settings: hardware, checkpointing, logging, validation and data loading."""

    # Hardware; DDP when devices x num_nodes > 1. The main model trained on 2 nodes x 8 GPUs (configs/main.yaml),
    # a global batch of 16 x training.batch_size = 512 windows.
    devices: int = 1
    num_nodes: int = 1
    precision: str = "bf16-mixed"
    benchmark: bool = True  # cudnn.benchmark

    # Checkpointing: best-K by checkpoint_monitor plus a periodic save every save_every_n_steps (which owns
    # last.ckpt). val/loss_macro is the equal-weight per-source mean of the validation objective.
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    save_every_n_steps: int = 500
    save_last: bool = True
    checkpoint_monitor: str = "val/loss_macro"

    # Logging
    log_dir: str = "logs"
    log_every_n_steps: int = 50
    step_print_every: int = 100
    enable_progress_bar: bool = False

    # Validation, every val_check_interval training batches
    val_check_interval: int = 1000
    num_sanity_val_steps: int = 2

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    accumulate_grad_batches: int = 1

    # Resume: None (fresh start), "auto" (the highest complete step in the run's checkpoint directory, or a
    # fresh start when it has none), "last", or a checkpoint path.
    resume: str | None = "auto"

    # Run name (checkpoint subdirectory and logger name); None builds one from the main hyperparameters.
    run_name: str | None = None

    # Weights & Biases (TensorBoard is always on). wandb_entity None uses the account's default entity.
    use_wandb: bool = False
    wandb_project: str = "gtf"
    wandb_entity: str | None = None

    # Live evaluation: at the first validation after every live_eval_every steps, future_prediction on a fixed
    # subset of live_eval_max_samples val windows under the cfg.eval protocol. 0 disables.
    live_eval_every: int = 1000
    live_eval_max_samples: int = 64


@dataclass
class EvalConfig:
    """Evaluation protocol (gtf/eval/run_eval.py, gtf/eval/evaluate.py). The paper's tables use
    n_samples=20, n_steps=10 and amp=True with the other defaults below (configs/benchmark.yaml)."""

    checkpoint_path: str = ""  # required at runtime
    tasks: list[str] = field(
        default_factory=lambda: [
            "future_prediction",
            "hand_to_object",
            "object_to_hand",
            "temporal_infilling",
            "goal_infilling",
            "goal_hand_only",
            "goal_object_only",
        ]
    )
    n_samples: int = 5  # samples per window (single = sample 0; best-of-N over all)
    n_steps: int = 20  # Euler integration steps
    seed: int = 42  # sample i is drawn after torch.manual_seed(seed + i)
    # Action filter: an entity's window counts as an action when its max GT excursion over the
    # predicted span is at least the threshold (mm).
    action_thresholds_mm: list[float] = field(default_factory=lambda: [25.0, 50.0, 100.0, 150.0])
    future_obs_ratio: float = 0.1  # fraction of the valid frames observed in future_prediction
    infill_gap_ratio: float = 0.33  # fraction of the valid frames masked in temporal_infilling
    goal_n_prefix: int = 1  # frames observed at the start in the goal tasks
    goal_n_suffix: int = 1  # frames observed at the end (goal state) in the goal tasks
    output_dir: str = "eval_results"
    split: str = "val"
    max_samples: int = 0  # windows evaluated per task; 0 = all
    batch_size: int = 16
    num_workers: int = 4
    window_size: int = 150  # padded window length in frames
    window_seconds: float = 5.0  # window extent in seconds
    stride_seconds: float = 2.5  # window start spacing in seconds
    amp: bool = False  # bf16 autocast for generation (metrics stay fp32)
    # Seeded permutation of the window order, applied before max_samples and sharding.
    window_shuffle_seed: int = 0
    # Window sharding: shard I of N evaluates every N-th batch; `run_eval.py --merge` combines them.
    num_shards: int = 1
    shard_idx: int = 0


@dataclass
class GTFConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Per-loss linear ramps keyed by lambda_* name (gtf/training/loss_schedule.py); the auxiliary contact weight ramps
    # from ~0 to its final value over the first 5K steps.
    loss_schedule: dict[str, LossRampConfig] = field(
        default_factory=lambda: {"lambda_aux_contact": LossRampConfig(start_step=0, end_step=5000, start_val=1e-9, end_val=0.1)}
    )
