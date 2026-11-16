#!/usr/bin/env bash
# Build the training corpus from the raw datasets.
#
# Stages, in order:
#   ingest     motion-capture loaders  -> results/<segment>/{canonical/, video.mp4}
#   split      interaction windows (__iwN children; SHOW3D thinned from 60 to 30 Hz)
#   manifest   datasets/canonical_manifest.json (the segment list the scene stage reads)
#   pca        DINOv3 PCA basis datasets/scene_dinov3_pca.pt (kept when it exists: a trained
#              model expects the basis it was trained with)
#   scene      per-frame scene caches results/<segment>/canonical/scene_cache_v2.h5
#   gravity    support-plane gravity from the scene caches of segments without one (HO-3D /
#              H2O-3D segments missing from the HO-3D loader's gravity table); an updated
#              segment's cache is deleted ...
#   scene      ... and rebuilt gravity-aligned by this second pass (valid caches are skipped)
#   manifest2  the manifest again, now recording the scene caches
#   uni3d      frozen Uni3D object embeddings -> data.uni3d_cache_path of the config
#   stats      window-sampled normalization statistics -> data.stats_path of the config
#
# Raw datasets are read from datasets/<name>/ (each loader's --start-from download step
# stages them; DexYCB, H2O and HO-3D need a manual download). MANO model files are read
# from third-party/mano/ (or $MANO_DIR). Action100M segments come from
# python -m gtf.extraction.action100m.pipeline, run into the same results directory before the
# manifest stage.
#
# Estimated-depth variant (the paper's comparison with methods that see RGB only): after the
# corpus is built, MoGe-2 depth replaces the sensor depth of DexYCB, H2O, HO-3D, HO-Cap and
# TACO in a separate cache tree and manifest; the normal caches are untouched:
#   python -m gtf.datasets.scene.preprocess --depth estimated --cache-root datasets/scene_estimated_depth
#   python -m gtf.datasets.build_manifest --scene-cache-root datasets/scene_estimated_depth \
#       --output datasets/scene_estimated_depth/canonical_manifest.json
#
# Usage:
#   bash scripts/build_corpus.sh CONFIG.yaml                 # every stage
#   bash scripts/build_corpus.sh CONFIG.yaml scene manifest2 # selected stages
# Environment: PYTHON (default: python), NUM_GPUS (default: 1),
#   SCENE_CPU_WORKERS (default: 4), LOADERS (default: all eight).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
CONFIG="${1:?usage: build_corpus.sh CONFIG.yaml [STAGE ...]}"
shift
STAGES=("$@")
[[ ${#STAGES[@]} -gt 0 ]] || STAGES=(ingest split manifest pca scene gravity scene manifest2 uni3d stats)

PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-1}"
SCENE_CPU_WORKERS="${SCENE_CPU_WORKERS:-4}"
LOADERS="${LOADERS:-dexycb hot3d taco oakink2 show3d ho3d h2o hocap}"
MANO_DIR="${MANO_DIR:-third-party/mano}"
export MANO_DIR

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

ingest() {
  for name in $LOADERS; do
    log "ingest $name"
    case "$name" in
      oakink2 | show3d) "$PYTHON" -m "gtf.datasets.$name" --start-from convert --max-seqs 0 ;;
      hocap) "$PYTHON" -m gtf.datasets.hocap --start-from convert --subject all ;;
      *) "$PYTHON" -m "gtf.datasets.$name" --start-from convert ;;
    esac
  done
}

split() {
  "$PYTHON" -m gtf.datasets.split_interaction_segments --datasets taco oakink2 show3d ho3d h2o hocap
}

manifest() {
  "$PYTHON" -m gtf.datasets.build_manifest
}

pca() {
  if [[ -f datasets/scene_dinov3_pca.pt ]]; then
    log "keeping the existing datasets/scene_dinov3_pca.pt"
  else
    "$PYTHON" -m gtf.datasets.scene.fit_pca
  fi
}

scene() {
  local pids=()
  for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" "$PYTHON" -m gtf.datasets.scene.preprocess \
      --num-shards "$NUM_GPUS" --shard-idx "$i" --scene-cpu-workers "$SCENE_CPU_WORKERS" --strict &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

gravity() {
  "$PYTHON" -m gtf.datasets.estimate_scene_gravity
}

uni3d() {
  "$PYTHON" -m gtf.datasets.build_uni3d_cache --config "$CONFIG"
}

stats() {
  "$PYTHON" -m gtf.data.compute_stats --config "$CONFIG" --mano-root "$MANO_DIR"
}

for stage in "${STAGES[@]}"; do
  log "stage: $stage"
  case "$stage" in
    ingest | split | manifest | pca | scene | gravity | uni3d | stats) "$stage" ;;
    manifest2) manifest ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
  esac
done
log "done: ${STAGES[*]}"
