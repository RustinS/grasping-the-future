# Extraction pipeline

Turns an RGB clip of hands manipulating an object into a 3D trajectory: two MANO hands and one rigid object at about
10 keyframes per second, written as `results/<segment>/canonical/canonical.npz` (format: `docs/data.md`). The
orchestrator for Action100M is `gtf/extraction/action100m/pipeline.py`; the stages live in `gtf/extraction/`.

```
find → fetch → validate → cutcheck → hands → molmo2 → sam3 → sam3d
     → object_pose → sam3dc (fallback) → optimize → refine → export_canonical → quality_gate → reground
```

| Step | What it does | Code |
|---|---|---|
| `find` | Search the Action100M annotations for manipulation segments (keyword, or object discovery with an LLM) | `action100m/find_segments.py` |
| `fetch` | Cut the segment from the local source video and normalize it to H.264 | `action100m/fetch_segments.py` |
| `validate` | Ask Molmo2 whether an object is manipulated | `frontend/run_valid_action.py` |
| `cutcheck` | Detect shot cuts; keep the first shot and harvest later shots as `__shotN` segments | `frontend/detect_cuts.py` |
| `hands` | Detect and reconstruct both hands with WiLoR (its own environment); writes `hands/` | `hands/run_wilor_hands.py` |
| `molmo2` | Point at the object on hand-guided candidate frames | `frontend/run_molmo2.py`, `pointing_frames.py` |
| `sam3` | Track the object mask through the video | `frontend/run_sam3.py` |
| `sam3d` | Reconstruct the object mesh and its pose at keyframes (SAM 3D Objects, MoGe depth) | `sam3d/run_sam3d.py` |
| `object_pose` | Camera motion (MegaSaM) and the object's 6-DoF trajectory by bundle adjustment over CoTracker tracks | `object_pose/` |
| `sam3dc` | Temporal SAM 3D reconstruction, used only when the bundle adjustment fails | `sam3d/run_sam3d_c.py` |
| `optimize` | Joint optimization of both hands (51 DoF each) and the object pose: image, depth, contact, penetration and smoothness terms | `optimizer/` |
| `refine` | Temporal refinement: hand-rigidity bridge, learned per-segment bias, filters, depth repair from object size | `stage2/` |
| `export_canonical` | Crop interaction windows, refine the object pose against its masks, enforce grasp coherence, resolve penetration, write the trajectory | `export/` |
| `quality_gate` | 21 physical, image-alignment, interaction and motion gates on the exported window | `gates/quality_gate.py` |
| `reground` | Re-ask Molmo2 where the object is and reject trajectories of the wrong object (gate 21) | `gates/reground.py` |

## The two chains

From the bundle adjustment on, a run uses one of two chains, defined in `gtf/extraction/chains.py`:

| Stage | `production` (default) | `upgraded` |
|---|---|---|
| Object pose | Reprojection and rotation smoothness | Adds a depth-relief term, no rotation smoothness |
| RGB hand masks | Not used | SAM 3 "hand" masks at the keyframes |
| Hand depth | MoGe depth at the hand | From the hand's apparent size, with the scene depth where the hand is small; corrections along the viewing ray |
| Object depth and scale | MoGe | The object follows the hand while it is held |
| Contact | Targets from 3D proximity | The hand slides along its viewing ray onto the object where the video shows contact; contact and hold terms gated by the image |
| Object rotation | Free against the bundle-adjusted chain | Kept on the bundle-adjusted chain |
| Refinement | With the learned bias | Without it |
| Export | Full 6-DoF refinement of the object against its masks | Image-plane slide only, then a trajectory smoother |

`production` is the default and the chain the paper's training data and benchmark were extracted with. `upgraded`
places the hands closer to the hands in the video, with less jitter, steadier object rotation and tighter grasps. The
quality-gate thresholds are calibrated on `production` outputs: check the gate pass rates before running `upgraded` at
scale.

Both chains take gravity from the support plane that the hands and the object rest on, pooled over about eight
keyframes; when no plane is supported, the segment has no gravity (`has_gravity: false`). A segment folder belongs to
one chain (`chains.claim` refuses mixing): run `upgraded` into its own results folder.

```bash
python -m gtf.extraction.action100m.pipeline --segment NAME                                     # production
ACTION100M_RESULTS_DIR=$PWD/results_upgraded python -m gtf.extraction.action100m.pipeline --segment NAME --chain upgraded
python -m gtf.extraction.optimizer.optimize_mano --segment=NAME --chain upgraded                  # one stage
```

## Running

Inputs: the Action100M annotation parquet files in `datasets/action100m/` and the source videos in `--video-dir` (one
file per YouTube id: `<id>.mp4`, `.mkv` or `.webm`). Without `--object`, `find` discovers the manipulated object of each
segment and validation and pointing are object-agnostic; with `--object NAME` only segments about that object run.

```bash
python -m gtf.extraction.action100m.pipeline --video-dir /data/videos --limit 100
python -m gtf.extraction.action100m.pipeline --start-from sam3d --force           # rerun from a step
python -m gtf.extraction.action100m.pipeline --segment NAME --stop-after optimize
bash scripts/run_split_extraction.sh segments.txt 4 0,1,2,3 24                   # one node: GPU stages from shared queues, CPU export and gates
```

Every step skips segments that already have its outputs; `--force` recomputes. Segments that fail validation, have no
hands, or fail a gate are rejected and recorded in their folder. The split driver stops after `quality_gate`; the
re-grounding verifier needs a GPU and runs afterwards:
`python -m gtf.extraction.action100m.pipeline --segment-file segments.txt --start-from reground`.

## Settings

A script has command-line flags only for what a run usually changes. Every other option is a setting, changed with
`--set KEY=VALUE` (repeatable); `--help` lists the keys, and the `SETTINGS` table of the script gives their defaults
and meanings:

```bash
python -m gtf.extraction.optimizer.optimize_mano --segment=NAME --set contact_threshold=0.01
python -m gtf.extraction.action100m.pipeline --object cup --set max_duration=10
```

A chain sets some optimizer settings; one `--set` overrides one of them. Environment variables:

| Variable | What it does |
|---|---|
| `EXTRACTION_CHAIN` | Chain of a whole run when `--chain` is not passed |
| `ACTION100M_RESULTS_DIR` | Results folder (default `results/`) |
| `EXTRACT_KNOWN_K`, `EXTRACT_KNOWN_FOV_X_DEG` | True intrinsics of a calibrated camera instead of the estimated focal length |
| `OBJREF_SEED_WORKERS`, `EXTRACTION_WILOR_WORKERS`, `OBJECT_POSE_DEPTH_WRITERS`, `EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS`, `EXTRACTION_OBJECT_POSE_BA_MEMORY_FRACTION`, `VALID_ACTION_VIDEO_BATCH_SIZE`, `REGROUND_DRAW_BATCH_SIZE` | Worker counts and batch sizes; results do not depend on them |

## Outputs of a segment

`results/<segment>/` holds the clip (`video.mp4`, with the uncut clip in `video_full.mp4` when `cutcheck` truncated it,
and the verdict in `cut_check.json`), `metadata.json`, one folder per step (`valid_action/`, `hands/`, `molmo2/`,
`SAM3/`, `sam3d/`, `object_pose/`, `sam3d_c/` when the fallback ran, `hand_masks/` in the `upgraded` chain,
`optimized_mano/`, `optimized_stage2/`), the trajectory in `canonical/`, the gate verdict in `quality/` and the
re-grounding evidence in `reground/`. Later shots that `cutcheck` harvests are segments of their own
(`<segment>__shotN/`); the interaction windows of an export are child segments `<segment>__iwN/` that link to their
parent's inputs and carry their own `canonical/` and `quality/`.

Several GPU stages can run in one process. Stages that need fp32 (matrix inverses, SVDs, SAM 3D decoding) run under
`torch.amp.autocast("cuda", enabled=False)`, so a bf16 autocast left enabled by an earlier stage cannot reach them; a
new stage needs the same guard.
