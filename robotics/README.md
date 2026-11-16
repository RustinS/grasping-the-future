# Retargeting forecasts to simulated robot hands

Code for the paper's experiment "Testing Physical Plausibility of Forecasting through Retargeting". The model's
future-prediction forecasts (MANO hands and a 6-DoF object) are retargeted to simulated XHand robot hands with
[SPIDER](https://github.com/facebookresearch/spider), and a retarget counts as a success when the simulated hands move
the free object along the forecast.

| Retargeting method | Successes | Success rate (%) |
|---|---:|---:|
| Kinematic | 15/103 | 14.6 |
| Kinematic + grasp prep | 45/103 | 43.7 |
| Dynamic (SPIDER) | 228/309 | 73.8 |

The per-trial records of the paper's run are in `results/` and print this table without any simulation.

```bash
python robotics/reproduce_table.py                  # all recordings
python robotics/reproduce_table.py --by-source      # plus one table per source dataset
```

## What is measured

- **Cohort.** The 170 windows in `cohort.json`, taken from the benchmark windows (`benchmark/*.json`).
  A window is eligible when its recorded reference has a stable set of hands and object, at least 50 mm of object
  excursion, a mesh no larger than 0.5 m, finite contiguous timing and a contiguous observed prefix (225 of 331
  windows); the cohort has one eligible window for each of the 170 recordings that have one. Hand configurations:
  50 right, 31 left, 89 bimanual.
- **Forecast.** One sample per window of the model's future-prediction task (first 10% of the 5 s window observed,
  hands and object predicted jointly; for the paper, seed 42 and 10 sampling steps).
- **Retained recordings.** The recorded human future (the *reference* arm) goes through the same pipeline. A recording
  is retained when the reference succeeds under SPIDER in at least one of three seeds: 103 of 170. All methods are
  scored on these 103 recordings, against the forecast.
- **Methods**, all applied to the same forecast:
  - *Kinematic*: SPIDER's fingertip and wrist IK (`ik_fast`), its joint targets replayed under physics, original scene
    and initial state. One execution per recording.
  - *Kinematic + grasp prep*: the grasp-prepared initial state, scene and controls that are given to SPIDER, replayed
    without trajectory optimization. One execution per recording.
  - *Dynamic (SPIDER)*: SPIDER's `examples/run_mjwp_fast.py` (preset `oakinkv2_fast`; 1,024 samples, 8 iterations,
    1.2 s horizon, 0.2 s control and knot spacing, one local retry and one revert; contact guidance and domain
    randomization off), seeds 0, 1 and 2, all counted.
- **Grasp preparation.** From observed frames only, the object counts as held when at least two fingertips are within
  3 cm of it in 80% of the first (at most ten, 50 Hz) frames. For a held object a fixed cross-entropy search
  (30 iterations x 256 candidates) finds a robot grasp and motor controls that keep the free object within 10 mm and
  0.3 rad for 3 s; the floor is lowered 0.3 m so it cannot carry the object. Afterwards the robot wrist follows the
  predicted object rigidly while the forecast hand is in contact, then blends back to the IK solution after ten frames
  without contact. Object targets are never changed.
- **Success.** Every method's controls are replayed in a fresh MuJoCo Warp world (free object with gravity, no object
  actuation, no state resets). A trial succeeds if the replay covers the whole clip and the object's mean position and
  rotation errors to the forecast over the predicted frames are below 0.10 m and 0.50 rad. Failed initializations and
  incomplete replays count as failures.

## Setup

1. **SPIDER and its environment.** SPIDER is CC BY-NC 4.0 and is not included; this clones it unmodified at commit
   `44717007de41cbef7565dff7ff9f4453557a2d3d` into `robotics/spider/` and, with `env`, creates `robotics/.venv`
   (Python 3.12) from `requirements.txt` (MuJoCo 3.7.0, mujoco-warp 3.7.0.1, warp-lang 1.12.1, mink 1.1.0,
   coacd 1.0.13, hydra-core 1.3.2, torch 2.13.0):

   ```bash
   bash robotics/third_party.sh env       # PYTHON=/path/to/python3.12 if it is not on PATH
   ```

   `SPIDER_ROOT=/path/to/spider` points the scripts at another clone. The simulation steps need an NVIDIA GPU.
2. **Project environment** (for `prepare.py` only): the repository's own environment, with MANO's `MANO_RIGHT.pkl` in
   `third-party/mano/` (see `third-party/setup.sh`).
3. **Forecast files**, in the format below. The benchmark writes them for the released model (repository root,
   project environment):

   ```bash
   for pop in mocap action100m; do
     python -m gtf.eval.benchmark run --checkpoint checkpoints/gtf_main.ckpt --population $pop --tasks future_prediction \
       --forecasts --out results/benchmark
   done
   ```

## Forecast input format

`prepare.py --forecast` takes one or more files saved with `torch.save` (for the paper: one for the MoCap windows,
one for the Action100M windows). Each is a dict with `task = "future_prediction"` and `rows`, a list with one dict
per benchmark window. `T` is the padded window length (150), `V`/`F` the object mesh's vertex and face counts.

| Key | Type, shape | Meaning |
|---|---|---|
| `identity` | `[source, segment_id, first_frame]` | window id, as in `benchmark/*.json` |
| `frame_times_s` | float64 `(T,)` | frame timestamps in seconds (any origin), strictly increasing over valid frames |
| `valid_frames` | bool `(T,)` | real frames; must be a prefix (`n = valid_frames.sum()`) |
| `presence_mask` | bool `(T, 5)` | entity present, token order `[left hand, right hand, object, left relation, right relation]` (first three used) |
| `obs_mask` | bool `(T, 5)` | entity given to the model; the observed frames must be a contiguous prefix |
| `decoded` | dict of float32 tensors | the forecast (sample 0), keys below |
| `gt_decoded` | dict of float32 tensors | the recorded reference, same keys |
| `betas_left`, `betas_right` | float32 `(10,)` | MANO shape |
| `obj_canonical_verts` | float32 `(V, 3)` | object mesh vertices in the object frame |
| `obj_scale` | float32 `(3,)` | per-axis scale; `obj_canonical_verts * obj_scale` is the mesh in meters |
| `obj_faces` | int64 `(F, 3)` | object mesh faces |

`decoded` and `gt_decoded` hold, per frame:

| Key | Shape | Meaning |
|---|---|---|
| `obj_translation` | `(T, 3)` | object position, m |
| `obj_rotation` | `(T, 3)` | object orientation, axis-angle; object-frame vertices map to `R @ (verts * scale) + t` |
| `hand_{left,right}_orient` | `(T, 3)` | MANO global orientation, axis-angle |
| `hand_{left,right}_pose` | `(T, 45)` | MANO finger pose, 15 joints x axis-angle (flat hand mean) |
| `hand_{left,right}_cam_t` | `(T, 3)` | MANO translation, m (added after forward kinematics, as in `gtf.model.repr.mano.MANOHelper`) |

All poses of a row share one metric, gravity-aligned frame with gravity along +Y (-Y up), the model's evaluation frame;
the origin is free (`prepare.py` moves the reference object's first position to x = y = 0 and its lowest vertex to
z = 0, with +Z up). Left hands follow the project convention: parameters of the right MANO model, mirrored in x by
`MANOHelper(..., "left")`. Over the observed prefix `decoded` must equal `gt_decoded` (within 2e-5) for the object and
every present hand. Only the present hands need meaningful values. Other keys in the file are ignored.

## Running

From the repository root:

```bash
F=results/benchmark                                   # written by gtf.eval.benchmark run --forecasts
python robotics/prepare.py --forecast $F/mocap/future_prediction.pt $F/action100m/future_prediction.pt
robotics/.venv/bin/python robotics/run.py --gpus 0 1 2 3 --workers-per-gpu 2 --cpu-workers 8 --threads 16
python robotics/reproduce_table.py robotics/output/trials.csv
```

`prepare.py` (project environment) checks every cohort window against the forecast files, writes `cohort.json` and
`protocol.json`, and exports both arms of every episode into SPIDER's MANO keypoint format. `--all-eligible` replaces
the paper's cohort with every eligible window of the forecast files (one per segment); `--out` changes the output
folder (default `robotics/output/`).

`run.py` runs the remaining steps for all episodes, one process each, in the SPIDER environment, and ends with
`summarize.py`. It resumes from receipts when rerun. The steps can also be run by hand:

```bash
robotics/.venv/bin/python robotics/preprocess.py ep000                   # CPU: scene, IK, grasp preparation
robotics/.venv/bin/python robotics/evaluate.py ep000 --arm reference     # GPU: 3 SPIDER seeds + replays
robotics/.venv/bin/python robotics/evaluate.py ep000 --arm forecast      # GPU, retained recordings only
robotics/.venv/bin/python robotics/summarize.py                          # tables, trials.csv, results.json
```

Cost, measured on B300 GPUs: preprocessing takes a median 75 s per recording (up to about 8 min with 16 threads),
one SPIDER seed a median 3 min (up to about 25 min); the paper's run made 861 SPIDER runs, 54 process-hours in all,
with several processes sharing each GPU.

## Output

Under `--out` (default `robotics/output/`):

| Path | Content |
|---|---|
| `cohort.json`, `protocol.json` | episodes (window id, hand configuration, observed frames) and the fixed recipe |
| `arms/{reference,forecast}/dataset/processed/forecast/` | SPIDER's dataset layout: `mano/` keypoints, `assets/` meshes and the XHand model, `xhand/<mode>/<episode>/` scenes and per-seed outputs (`trajectory_kinematic*.npz`, `trajectory_mjwp_fast.npz`, `run.json`, `replay.json`, `trajectory_replay.npz`) |
| `arms/<arm>/episodes/<episode>/` | `generated.npz` (resampled joints and object poses), `export.json` (timing, world transform) |
| `preprocess/<episode>/status.json`, `grasp_init/<episode>/` | preprocessing receipts and the prepared grasp |
| `qualification/<episode>.json` | reference outcome per seed and whether the recording is retained |
| `baselines/{native_ik,adapted_ik}/forecast/<episode>/replay.json` | the two kinematic baselines |
| `trials.csv`, `qualification.csv`, `results.json` | written by `summarize.py`; `results.json` adds recording-bootstrap 95% intervals, per-source and per-hand-configuration rates, and the stricter set of recordings whose reference succeeds in all three seeds |

`results/trials.csv` and `results/qualification.csv` are those two tables from the paper's run, reduced to episode
ids, source, hand configuration, flags and errors. `trials.csv` methods: `native_ik` = Kinematic, `adapted_ik` =
Kinematic + grasp prep, `spider` = Dynamic (SPIDER).

## Files

| File | Environment | Role |
|---|---|---|
| `common.py` | both | constants, output layout, resampling, scoring |
| `prepare.py` | project | cohort and export |
| `preprocess.py` | SPIDER | SPIDER decompose, generate_xml and ik_fast; contact; grasp search and adapter |
| `evaluate.py` | SPIDER | SPIDER trials, independent replays, the two kinematic baselines |
| `run.py` | either | single-node runner (one process per CPU or GPU worker) |
| `summarize.py` | either | success rates |
| `reproduce_table.py` | standard library | the paper's table from a `trials.csv` |
| `third_party.sh`, `requirements.txt` | | SPIDER clone and environment |
| `cohort.json` | | the paper's 170 windows |
| `results/` | | the paper's per-trial records |

## Notes

- The success test is relative to the forecast, not to the recorded future: it measures whether a robot can carry
  out the predicted motion, not whether the prediction is accurate.
- On 43 of the 103 retained recordings an object that simply stayed where it was when the prediction starts would
  already be within the thresholds (an upper bound: a held object would fall). On the other 60 the order of the
  three methods is the same and the gaps are larger: Kinematic 2/60, Kinematic + grasp prep 15/60, SPIDER 115/180.
- Contact simulation on the GPU is not bitwise deterministic, so a rerun can flip borderline trials.
- Action100M references are reconstructed from video, not measured. A failed reference means failure under this
  pipeline and budget, not that the human motion cannot be transferred.
- The XHand has an actuated floating base; nothing here says whether an arm could reach these poses.
