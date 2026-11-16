#!/usr/bin/env bash
# GPU/CPU split extraction driver: runs the Action100M pipeline (gtf.extraction.action100m.pipeline) on a list of segments on one multi-GPU node.
#
# GPU path: the front end (validate → molmo2) in static shards, one per GPU; then SAM3 and the heavy stages
#   (sam3d → object_pose → sam3dc → optimize → refine) from shared SQLite queues (gtf/extraction/action100m/stage_queue.py), one
#   block of segments at a time (gtf/extraction/batch/tail_blocks.py).
# CPU path: export_canonical → quality_gate per segment without CUDA (gtf/extraction/batch/cpu_tail_queue.py), starting as soon
#   as the block that contains the segment has finished on the GPUs.
#
# Export and gate give the same results on CPU and GPU (contact projection uses CPU Adam). The re-grounding verifier
# (gate 21) needs a GPU and is not part of this driver; run it afterwards:
#   .venv/bin/python -m gtf.extraction.action100m.pipeline --segment-file SEG_LIST.txt --start-from reground
#
# Usage:
#   bash scripts/run_split_extraction.sh SEG_LIST.txt [NGPU] [GPU_IDS] [NCPU] [EXTRA_ARGS...]
#     SEG_LIST.txt : one segment name per line
#     NGPU         : number of GPU shards            (default 6)
#     GPU_IDS      : comma-separated GPU ids         (default 0,1,2,3,4,5)
#     NCPU         : CPU-pool worker count           (default 16)
#     EXTRA_ARGS   : forwarded to the pipeline (e.g. --force, --early-exit, --chain upgraded)
#
# Environment: EXTRACTION_PYTHON (default .venv/bin/python), EXTRACTION_LOGDIR (default logs/split_<time>),
#   ACTION100M_RESULTS_DIR, EXTRACTION_CHAIN, worker counts EXTRACTION_{WILOR,SAM3}_WORKERS (per GPU, default 2),
#   EXTRACTION_OBJREF_SEED_WORKERS (default 2), EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS (0 or 1, default 1),
#   EXTRACTION_TAIL_BLOCKS (default 4), EXTRACTION_TAIL_CPU_GPU_RATIO, EXTRACTION_CUDA_HOME,
#   EXTRACTION_CPU_TAIL_TIMEOUT_SECONDS (cpu_tail_queue.py), EXTRACTION_SKIP_BA_HEALTH=1.
#
# Example:
#   bash scripts/run_split_extraction.sh segs.txt 4 0,1,2,3 24 --early-exit

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${EXTRACTION_PYTHON:-.venv/bin/python}
if [[ ! -x "$PYTHON" ]]; then
  echo "[split] ERROR: extraction Python is not executable: $PYTHON" >&2
  exit 2
fi

SEG_LIST=${1:?usage: run_split_extraction.sh SEG_LIST.txt [NGPU] [GPU_IDS] [NCPU] [EXTRA...]}
NGPU=${2:-6}
GPU_IDS=${3:-0,1,2,3,4,5}
NCPU=${4:-16}
OBJREF_SEED_WORKERS=${EXTRACTION_OBJREF_SEED_WORKERS:-2}
WILOR_PER_GPU=${EXTRACTION_WILOR_WORKERS:-2}
SAM3_PER_GPU=${EXTRACTION_SAM3_WORKERS:-2}
OBJECT_POSE_BA_WORKERS=${EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS:-1}
TAIL_BLOCKS=${EXTRACTION_TAIL_BLOCKS:-4}
# CPU-tail work per unit of heavy-GPU-stage work, in (0, 1]: it sets the block sizes, which decrease geometrically
# (about 39 / 28 / 20 / 14% of the work for 4 blocks; gtf/extraction/batch/tail_blocks.py).
TAIL_CPU_GPU_RATIO=${EXTRACTION_TAIL_CPU_GPU_RATIO:-0.713223140496}
shift $(( $# > 4 ? 4 : $# ))
EXTRA=("$@")

# The stage commands set their own control arguments: strip those from the forwarded list and keep
# behavioral flags such as --force / --early-exit.
CLEAN_EXTRA=()
OPTIMIZER_PER_GPU=4
_skip_next=0
_capture=""
for _arg in "${EXTRA[@]}"; do
  if [ "$_skip_next" -eq 1 ]; then
    if [ "$_capture" = "optimizer-workers" ]; then
      OPTIMIZER_PER_GPU=$_arg
    fi
    _skip_next=0
    _capture=""
    continue
  fi
  case "$_arg" in
    --optimizer-workers)
      _skip_next=1
      _capture="optimizer-workers"
      ;;
    --optimizer-workers=*)
      OPTIMIZER_PER_GPU=${_arg#*=}
      ;;
    --start-from|--stop-after|--segment|--segment-file|--stage-queue-db|--stage-worker-id)
      _skip_next=1
      ;;
    --start-from=*|--stop-after=*|--segment=*|--segment-file=*|--stage-queue-db=*|--stage-worker-id=*)
      ;;
    *) CLEAN_EXTRA+=("$_arg") ;;
  esac
done
if [ "$_skip_next" -eq 1 ]; then
  echo "[split] ERROR: missing value for the final control argument" >&2
  exit 2
fi
if ! [[ "$OPTIMIZER_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "[split] ERROR: --optimizer-workers must be a positive integer" >&2
  exit 2
fi
if ! [[ "$NCPU" =~ ^[1-9][0-9]*$ && "$OBJREF_SEED_WORKERS" =~ ^[1-9][0-9]*$ && "$WILOR_PER_GPU" =~ ^[1-9][0-9]*$ && "$SAM3_PER_GPU" =~ ^[1-9][0-9]*$ && "$TAIL_BLOCKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[split] ERROR: CPU, object-refiner, WiLoR, SAM3, and tail-block counts must be positive integers" >&2
  exit 2
fi
if ! [[ "$OBJECT_POSE_BA_WORKERS" =~ ^[01]$ ]]; then
  echo "[split] ERROR: async object-pose BA workers must be 0 or 1" >&2
  exit 2
fi
if ! TAIL_CPU_GPU_RATIO=$("$PYTHON" -m gtf.extraction.batch.tail_blocks validate-ratio \
  --cpu-gpu-work-ratio "$TAIL_CPU_GPU_RATIO"); then
  echo "[split] ERROR: invalid EXTRACTION_TAIL_CPU_GPU_RATIO" >&2
  exit 2
fi

NVJITLINK=$("$PYTHON" -c 'import glob, os, sys; print(next(iter(glob.glob(os.path.join(sys.prefix, "lib", "python*", "site-packages", "nvidia", "nvjitlink", "lib", "libnvJitLink.so.12"))), ""))')
if [[ -n "$NVJITLINK" ]]; then
  # Preload the environment's own nvJitLink (an absolute path: JIT builders change their working
  # directory before spawning nvcc).
  export LD_PRELOAD="$NVJITLINK${LD_PRELOAD:+:$LD_PRELOAD}"
fi

CUDA_ROOT=${EXTRACTION_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}
if [[ ! -x "$CUDA_ROOT/bin/nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_ROOT=/usr/local/cuda
fi
if [[ ! -x "$CUDA_ROOT/bin/nvcc" ]]; then
  echo "[split] ERROR: CUDA compiler is not executable: $CUDA_ROOT/bin/nvcc" >&2
  exit 2
fi
export CUDA_HOME="$CUDA_ROOT"
export PYOPENGL_PLATFORM=egl
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
# Each front-end process owns one GPU; its persistent WiLoR runners share that GPU.
export EXTRACTION_WILOR_WORKERS="$WILOR_PER_GPU"
export EXTRACTION_OBJECT_POSE_ASYNC_BA_WORKERS="$OBJECT_POSE_BA_WORKERS"

mapfile -t SEGS < <(grep -v '^\s*$' "$SEG_LIST")
echo "[split] ${#SEGS[@]} segments, ${NGPU} GPU shards (${GPU_IDS}), ${WILOR_PER_GPU} WiLoR and ${SAM3_PER_GPU} SAM3 workers/GPU, ${OBJECT_POSE_BA_WORKERS} async object-pose BA worker(s), ${TAIL_BLOCKS} heavy flow-shop blocks at CPU/GPU ratio ${TAIL_CPU_GPU_RATIO}, ${NCPU} CPU workers x ${OBJREF_SEED_WORKERS} object-refiner seed threads"
[ ${#SEGS[@]} -eq 0 ] && { echo "[split] nothing to do"; exit 0; }
LOGDIR=${EXTRACTION_LOGDIR:-logs/split_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$LOGDIR"
ATTEMPT_ID=$(basename "$LOGDIR")
TAIL_BLOCK_DIR="$LOGDIR/tail_blocks"
TAIL_BLOCK_READY_DIR="$TAIL_BLOCK_DIR/ready"
TAIL_BLOCK_PLAN="$TAIL_BLOCK_DIR/plan.json"
mkdir -p "$TAIL_BLOCK_READY_DIR"

IFS=',' read -ra GPUS <<< "$GPU_IDS"

# ── Occupied-GPU guard ───────────────────────────────────────────────────────
# A Molmo2 shard placed on a GPU that another job already fills gets CPU-offloaded and fails
# its whole slice. At launch, any requested GPU with > 40 GB in use (the pipeline peaks at
# ~38 GB) is swapped for the least-used GPU outside the requested list; if none qualifies,
# the shard doubles up on the least-used requested GPU.
declare -A GPU_USED
while IFS=',' read -r _idx _mem; do GPU_USED[${_idx// /}]=${_mem// /}; done \
  < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
for _gi in "${!GPUS[@]}"; do
  _g=${GPUS[$_gi]}
  if [ "${GPU_USED[$_g]:-0}" -gt 40000 ]; then
    _list=",${GPUS[*]},"; _list=${_list// /,}
    _best=""; _best_used=40000; _best2=""; _best2_used=40000
    for _idx in "${!GPU_USED[@]}"; do
      if [ "$_idx" = "$_g" ]; then continue; fi
      if [[ "$_list" != *",$_idx,"* ]]; then
        if [ "${GPU_USED[$_idx]}" -lt "$_best_used" ]; then _best=$_idx; _best_used=${GPU_USED[$_idx]}; fi
      else
        if [ "${GPU_USED[$_idx]}" -lt "$_best2_used" ]; then _best2=$_idx; _best2_used=${GPU_USED[$_idx]}; fi
      fi
    done
    if [ -z "$_best" ] && [ -n "$_best2" ]; then
      _best=$_best2; _best_used=$_best2_used
      echo "[split] WARNING: no free GPU outside the requested list; doubling up on GPU $_best"
    fi
    if [ -n "$_best" ]; then
      echo "[split] GPU $_g occupied (${GPU_USED[$_g]} MiB used) -> shard remapped to GPU $_best (${_best_used} MiB used)"
      GPUS[$_gi]=$_best
      GPU_USED[$_best]=$(( _best_used + 38000 ))
    else
      echo "[split] WARNING: GPU $_g occupied (${GPU_USED[$_g]} MiB used) and every alternative is full; the shard will likely fail"
    fi
  fi
done

# A remap can put two front-end shards on one physical GPU; the heavy stages run one
# resident model (or the optimizer workers) per physical GPU.
QUEUE_GPUS=()
declare -A _queue_gpu_seen
for _g in "${GPUS[@]}"; do
  if [ -z "${_queue_gpu_seen[$_g]:-}" ]; then
    QUEUE_GPUS+=("$_g")
    _queue_gpu_seen[$_g]=1
  fi
done

PHASE_START="$LOGDIR/.phase_a_started"
PHASE_DONE="$LOGDIR/.phase_a_done"
PHASE_SUCCESS="$LOGDIR/.phase_a_succeeded"
CPU_TAIL_DONE="$LOGDIR/.cpu_tail_done"
CPU_TAIL_PID=""
# A reused log directory can hold markers of an interrupted run: remove them first; only the
# successful path below writes PHASE_SUCCESS.
rm -f "$PHASE_START" "$PHASE_DONE" "$PHASE_SUCCESS" "$CPU_TAIL_DONE"
cleanup_tail() {
  touch "$PHASE_DONE"
  if [ -n "${CPU_TAIL_PID:-}" ] && kill -0 "$CPU_TAIL_PID" 2>/dev/null; then
    kill "$CPU_TAIL_PID" 2>/dev/null || true
    wait "$CPU_TAIL_PID" 2>/dev/null || true
  fi
  if [ ! -e "$CPU_TAIL_DONE" ]; then
    _tail_done_tmp="${CPU_TAIL_DONE}.tmp.$$"
    printf '143\n' > "$_tail_done_tmp"
    mv "$_tail_done_tmp" "$CPU_TAIL_DONE"
  fi
}
trap cleanup_tail EXIT
trap 'exit 130' INT TERM

# ── Phase A1: front end through Molmo2, statically sharded ───────────────
# SAM3 runs from the shared queue below instead: two resident predictors per GPU overlap
# their CPU / video-decode gaps, and dynamic claims balance long segments.
pids=()
for ((g = 0; g < NGPU; g++)); do
  ARGS=()
  for ((i = g; i < ${#SEGS[@]}; i += NGPU)); do
    ARGS+=("--segment=${SEGS[$i]}")
  done
  [ ${#ARGS[@]} -eq 0 ] && continue
  CUDA_VISIBLE_DEVICES=${GPUS[$((g % ${#GPUS[@]}))]} \
    "$PYTHON" -u -m gtf.extraction.action100m.pipeline \
      --start-from validate --stop-after molmo2 "${ARGS[@]}" "${CLEAN_EXTRA[@]}" \
      > "$LOGDIR/gpu_front$g.log" 2>&1 &
  pids+=($!)
done
echo "[split] front-end launched: ${#pids[@]} GPU shards"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "[split] ERROR: a front-end shard exited nonzero; aborting before dependent stages" >&2
  exit 1
fi

# Fold cutcheck-harvested shot children ({seg}__shotN) into the segment list so the
# queued heavy stages + CPU tail (which re-reads the file) process them too.
_n_shots=0
RESULTS_ROOT=${ACTION100M_RESULTS_DIR:-$PWD/results}
while IFS= read -r _cname; do
  [ -z "$_cname" ] && continue
  echo "$_cname" >> "$SEG_LIST"
  _n_shots=$((_n_shots + 1))
done < <(
  "$PYTHON" - "$SEG_LIST" "$RESULTS_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

roster_path, results_root = Path(sys.argv[1]), Path(sys.argv[2])
names = [
    line.strip()
    for line in roster_path.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
seen = set(names)
for parent in names:
    if re.search(r"__shot[0-9]+$", parent):
        continue
    try:
        record = json.loads((results_root / parent / "cut_check.json").read_text())
    except (OSError, json.JSONDecodeError):
        continue
    children = record.get("shot_children", [])
    if not isinstance(children, list):
        continue
    pattern = re.compile(re.escape(parent) + r"__shot[0-9]+$")
    for child in children:
        # Only the child names cutcheck recorded (no glob over the results directory).
        if (
            isinstance(child, str)
            and pattern.fullmatch(child)
            and child not in seen
            and (results_root / child).is_dir()
        ):
            seen.add(child)
            print(child)
PY
)
# an if-block, not a bare && list: under set -e a false AND-list ends the script
if [ "$_n_shots" -gt 0 ]; then
  echo "[split] folded $_n_shots harvested shot children into the segment list"
fi

# --early-exit runs once, as its own pass, after SAM3 (below); the stage workers do not get it.
_has_early_exit=0
STAGE_EXTRA=()
for _arg in "${CLEAN_EXTRA[@]}"; do
  if [ "$_arg" = "--early-exit" ]; then
    _has_early_exit=1
  else
    STAGE_EXTRA+=("$_arg")
  fi
done
run_queued_stage() { # stage, workers per GPU, optional segment file, optional log tag
  local stage=$1 per_gpu=$2 work_segments=${3:-$SEG_LIST} tag=${4:-}
  local queue_name="${stage}${tag}" db="$LOGDIR/${stage}${tag}_queue.sqlite"
  "$PYTHON" -m gtf.extraction.action100m.stage_queue init \
    --db "$db" --stage "$stage" --segment-file "$work_segments"
  local stage_pids=()
  for ((g = 0; g < ${#QUEUE_GPUS[@]}; g++)); do
    for ((w = 0; w < per_gpu; w++)); do
      local worker="${queue_name}_g${g}_w${w}"
      CUDA_VISIBLE_DEVICES=${QUEUE_GPUS[$g]} \
        "$PYTHON" -u -m gtf.extraction.action100m.pipeline \
          --segment-file "$work_segments" --start-from "$stage" --stop-after "$stage" \
          --stage-queue-db "$db" --stage-worker-id "$worker" --optimizer-workers 1 \
          "${STAGE_EXTRA[@]}" > "$LOGDIR/${worker}.log" 2>&1 &
      stage_pids+=($!)
    done
  done
  echo "[split] $queue_name queue launched: ${#stage_pids[@]} workers"
  local primary_fail=0
  for p in "${stage_pids[@]}"; do wait "$p" || primary_fail=1; done
  local requeued initial_requeued recovery_fail=0
  requeued=$("$PYTHON" -m gtf.extraction.action100m.stage_queue requeue --db "$db" --stage "$stage")
  initial_requeued=$requeued
  if [ "$requeued" -ne 0 ]; then
    # Recovery is one process on one GPU: a segment whose worker failed (for example a second
    # resident SAM3 predictor out of GPU memory) is retried here without model overlap.
    echo "[split] $queue_name recovering $requeued jobs abandoned by failed workers with one fresh worker"
    CUDA_VISIBLE_DEVICES=${QUEUE_GPUS[0]} \
      "$PYTHON" -u -m gtf.extraction.action100m.pipeline \
        --segment-file "$work_segments" --start-from "$stage" --stop-after "$stage" \
        --stage-queue-db "$db" --stage-worker-id "${queue_name}_recovery" --optimizer-workers 1 \
        "${STAGE_EXTRA[@]}" > "$LOGDIR/${queue_name}_recovery.log" 2>&1 || recovery_fail=1
    requeued=$("$PYTHON" -m gtf.extraction.action100m.stage_queue requeue --db "$db" --stage "$stage")
  fi
  local stage_status
  stage_status=$("$PYTHON" -m gtf.extraction.action100m.stage_queue status --db "$db" --stage "$stage")
  echo "$stage_status"
  local stage_fail=0
  if [[ "$stage_status" == *"pending="* || "$stage_status" == *"running="* ]]; then
    echo "[split] WARNING: $queue_name queue did not drain completely"
    stage_fail=1
  fi
  if [ "$requeued" -ne 0 ]; then
    echo "[split] WARNING: $queue_name still has $requeued abandoned jobs after recovery"
    stage_fail=1
  fi
  if [ "$primary_fail" -ne 0 ] || [ "$recovery_fail" -ne 0 ]; then
    if [ "$stage_fail" -eq 0 ]; then
      # A segment is acknowledged only after its completion marker exists, so a fully
      # drained queue means success even if a worker failed during model load or teardown.
      echo "[split] WARNING: $queue_name worker exit(s) were recovered; queue drained cleanly (initially requeued=$initial_requeued)"
    fi
  fi
  # Do not end the function with a bare `[ ... ] && ...` list: with stage_fail=0 it returns 1,
  # and set -e then ends the script after the first clean stage.
  if [ "$stage_fail" -ne 0 ]; then
    echo "[split] ERROR: $queue_name had worker failures; aborting before dependent stages (see logs)" >&2
    return 1
  fi
}

# ── Phase A2: SAM3 from the shared queue ───────────────────────────────
# Each worker owns its predictor; a segment is acknowledged only after .sam3_done exists.
run_queued_stage sam3 "$SAM3_PER_GPU"

# The interaction proxy runs after SAM3 has finished, so it never reads a stale object mask.
if [ "$_has_early_exit" -eq 1 ]; then
  CUDA_VISIBLE_DEVICES="" "$PYTHON" -u -m gtf.extraction.action100m.pipeline \
    --segment-file "$SEG_LIST" --early-exit --early-exit-only "${STAGE_EXTRA[@]}" \
    > "$LOGDIR/early_exit.log" 2>&1 || echo "[split] WARNING: early-exit pass failed"
fi

# The CPU tail starts only after the whole front end has finished, so outputs of an earlier
# run cannot pass its freshness checks before this run reaches the segment. It still overlaps
# every heavy GPU stage.
touch "$PHASE_START"
(
  set +e
  "$PYTHON" -u -m gtf.extraction.batch.cpu_tail_queue \
    --segment-file "$SEG_LIST" --phase-start "$PHASE_START" --phase-done "$PHASE_DONE" \
    --phase-success "$PHASE_SUCCESS" \
    --log-dir "$LOGDIR" --attempt-id "$ATTEMPT_ID" \
    --tail-plan "$TAIL_BLOCK_PLAN" --block-ready-dir "$TAIL_BLOCK_READY_DIR" \
    --workers "$NCPU" \
    --seed-workers "$OBJREF_SEED_WORKERS" -- "${CLEAN_EXTRA[@]}"
  _tail_rc=$?
  _tail_done_tmp="${CPU_TAIL_DONE}.tmp.$$"
  printf '%s\n' "$_tail_rc" > "$_tail_done_tmp"
  mv "$_tail_done_tmp" "$CPU_TAIL_DONE"
  exit "$_tail_rc"
) > "$LOGDIR/cpu_tail.log" 2>&1 &
CPU_TAIL_PID=$!

# The heavy stages run block by block (each block through refine), so the CPU pool can export
# and gate a finished block while the GPUs work on the next one. A block's ready marker is
# written only after every GPU stage of that block has finished.
#
# The BA health check runs once over the whole list below, before PHASE_SUCCESS. CPU work on
# early blocks can be wasted when that check fails, but such a run never succeeds.
_tail_plan_result=$("$PYTHON" -m gtf.extraction.batch.tail_blocks plan \
  --segment-file "$SEG_LIST" --output-dir "$TAIL_BLOCK_DIR" \
  --blocks "$TAIL_BLOCKS" --attempt-id "$ATTEMPT_ID" \
  --phase-start "$PHASE_START" --results-root "$RESULTS_ROOT" \
  --cpu-gpu-work-ratio "$TAIL_CPU_GPU_RATIO")
IFS=$'\t' read -r _tail_plan_path _tail_block_count <<< "$_tail_plan_result"
if [ "$_tail_plan_path" != "$TAIL_BLOCK_PLAN" ] || ! [[ "$_tail_block_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "[split] ERROR: heavy-block planner returned an invalid plan: $_tail_plan_result" >&2
  exit 1
fi
for ((block = 0; block < _tail_block_count; block++)); do
  _block_tag=$(printf '_b%03d' "$block")
  _block_file="$TAIL_BLOCK_DIR/block_$(printf '%03d' "$block").segments"
  run_queued_stage sam3d 1 "$_block_file" "$_block_tag"
  run_queued_stage object_pose 1 "$_block_file" "$_block_tag"
  run_queued_stage sam3dc 1 "$_block_file" "$_block_tag"
  run_queued_stage optimize "$OPTIMIZER_PER_GPU" "$_block_file" "$_block_tag"
  run_queued_stage refine 1 "$_block_file" "$_block_tag"
  "$PYTHON" -m gtf.extraction.batch.tail_blocks publish-ready \
    --plan "$TAIL_BLOCK_PLAN" --ready-dir "$TAIL_BLOCK_READY_DIR" \
    --block-index "$block" --attempt-id "$ATTEMPT_ID" \
    --segment-file "$SEG_LIST" --phase-start "$PHASE_START"
done

# The BA health check over the whole segment list, before the global success marker.
if [ "${EXTRACTION_SKIP_BA_HEALTH:-0}" != "1" ]; then
  "$PYTHON" -m gtf.extraction.batch.check_ba_health --seg-file "$SEG_LIST" || {
    echo "[split] ABORT: shard-wide BA health check failed (see above); set EXTRACTION_SKIP_BA_HEALTH=1 to override"
    exit 1
  }
fi

# Success: the CPU tail can take every remaining segment. The failure cleanup writes
# PHASE_DONE but never this marker.
touch "$PHASE_SUCCESS"
touch "$PHASE_DONE"
wait "$CPU_TAIL_PID" || echo "[split] WARNING: CPU tail had failures (see cpu_tail.log)"
CPU_TAIL_PID=""
trap - EXIT INT TERM

echo "[split] done. Logs: $LOGDIR/"
n_pass=$(grep -l "passed" "$LOGDIR"/cpu_*.log 2>/dev/null | wc -l || true)
echo "[split] $n_pass per-segment CPU logs report a quality-gate result"
