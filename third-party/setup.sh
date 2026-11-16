#!/bin/bash
# Fetch and build the third-party code this repository depends on, at pinned commits.
#
#   bash third-party/setup.sh model        # what training and evaluation need (LitePT, MANO check)
#   bash third-party/setup.sh corpus       # + Uni3D, to build the object-embedding cache of a training corpus
#   bash third-party/setup.sh extraction   # + MegaSaM, SAM 3D Objects, Kaolin and WiLoR, to extract trajectories from video
#   bash third-party/setup.sh all
#
# Run from the repository root with the project environment active (see README.md).
# MANO is not downloaded: register at https://mano.is.tue.mpg.de and place MANO_RIGHT.pkl (and MANO_LEFT.pkl)
# in third-party/mano/. Gated Hugging Face models (SAM 3D Objects, DINOv3) need `hf auth login` and an
# accepted license on their model pages.
set -euo pipefail
cd "$(dirname "$0")"
TP="$(pwd)"
PY="${PYTHON:-python}"
CUDA_ARCHS="${TORCH_CUDA_ARCH_LIST:-8.0;9.0;10.0}"
WHAT="${1:-model}"

pip_install() {  # into the project environment: uv when available (uv environments have no pip), else pip
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" "$@"
  else
    "$PY" -m pip install "$@"
  fi
}

clone() {  # clone <dir> <url> <commit> [--recursive]
  local dir=$1 url=$2 commit=$3
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet "$url" "$dir"
  fi
  git -C "$dir" checkout --quiet "$commit"
  if [ "${4:-}" = "--recursive" ]; then
    git -C "$dir" submodule update --quiet --init --recursive
  fi
  echo "  $dir @ $commit"
}

apply_patch() {  # apply_patch <dir> <patch>; idempotent
  if git -C "$1" apply --check "$TP/patches/$2" 2>/dev/null; then
    git -C "$1" apply "$TP/patches/$2"
    echo "  patched $1 with $2"
  else
    echo "  $1: $2 already applied (or does not apply cleanly; check with git -C $1 diff)"
  fi
}

check_mano() {
  if [ ! -f mano/MANO_RIGHT.pkl ]; then
    echo "MISSING third-party/mano/MANO_RIGHT.pkl: download MANO from https://mano.is.tue.mpg.de" >&2
    return 1
  fi
  echo "  MANO found"
}

setup_model() {
  echo "[model]"
  # LitePT scene backbone (MIT). The code imports `litept` and `libs.pointrope` from third-party/.
  clone LitePT https://github.com/prs-eth/LitePT f0cc769
  ln -sfn LitePT/litept litept
  ln -sfn LitePT/libs libs
  # The MANO layer (manopth) and chumpy are dependencies of the package (pyproject.toml).
  check_mano || true
}

setup_corpus() {
  echo "[corpus]"
  # Uni3D object encoder (MIT), used once to embed every object mesh of a corpus.
  clone uni3d https://github.com/baaivision/Uni3D 64e03c3
  pip_install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
  mkdir -p uni3d/weights/modelzoo/uni3d-g
  [ -f uni3d/weights/modelzoo/uni3d-g/model.pt ] || \
    hf download BAAI/Uni3D modelzoo/uni3d-g/model.pt --local-dir uni3d/weights
}

setup_extraction() {
  echo "[extraction]"
  # MegaSaM camera tracking (Apache-2.0; DROID-SLAM and lietorch BSD-3), patched to build with current PyTorch
  # (third-party/README.md lists the changes).
  clone mega-sam https://github.com/mega-sam/mega-sam a27b4e6 --recursive
  apply_patch mega-sam megasam.patch
  (cd mega-sam/base && TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" $PY setup.py install)
  (cd mega-sam/base/thirdparty/lietorch && TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" $PY setup.py install)

  # SAM 3D Objects (SAM License), patched as listed in third-party/README.md.
  clone sam-3d-objects https://github.com/facebookresearch/sam-3d-objects 81a8237
  apply_patch sam-3d-objects sam3d.patch
  pip_install -e sam-3d-objects
  if [ ! -f sam-3d-objects/checkpoints/hf/pipeline.yaml ]; then
    # The model repository keeps its files under checkpoints/.
    hf download facebook/sam-3d-objects --local-dir sam-3d-objects/checkpoints/hf-download
    mkdir -p sam-3d-objects/checkpoints/hf
    mv sam-3d-objects/checkpoints/hf-download/checkpoints/* sam-3d-objects/checkpoints/hf/
    rm -rf sam-3d-objects/checkpoints/hf-download
  fi
  # The pipeline's depth model is the MetricAnything student (same MoGe-2 interface).
  sed -i 's#^\(\s*pretrained_model_name_or_path:\s*\).*moge.*$#\1yjh001/metricanything_student_pointmap#I' \
    sam-3d-objects/checkpoints/hf/pipeline.yaml

  # Kaolin (Apache-2.0), built from source for the penetration and contact geometry of the optimizer.
  clone kaolin https://github.com/NVIDIAGameWorks/kaolin 9921a79 --recursive
  IGNORE_TORCH_VER=1 TORCH_CUDA_ARCH_LIST="$CUDA_ARCHS" pip_install --no-build-isolation -e kaolin

  # WiLoR hand detection and reconstruction (CC-BY-NC-ND-4.0), in its own environment.
  mkdir -p wilor
  clone wilor/WiLoR https://github.com/rolpotamias/WiLoR fcb9113
  if [ ! -x wilor/.venv/bin/python ]; then
    uv venv --python 3.11 wilor/.venv
    uv pip install -p wilor/.venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
    uv pip install -p wilor/.venv/bin/python "numpy<2" opencv-python pyrender pytorch-lightning scikit-image \
      "smplx==0.1.28" yacs timm einops "ultralytics>=8.4" trimesh scipy
  fi
  mkdir -p wilor/WiLoR/pretrained_models
  for f in detector.pt wilor_final.ckpt; do
    [ -f "wilor/WiLoR/pretrained_models/$f" ] || \
      wget -q "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/$f" -P wilor/WiLoR/pretrained_models/
  done
  # WiLoR reads MANO through smplx; give it a copy without chumpy objects.
  if check_mano && [ ! -f wilor/WiLoR/mano_data/MANO_RIGHT.pkl ]; then
    $PY - <<'EOF'
import pickle
import numpy as np
import scipy.sparse as sp
d = pickle.load(open("mano/MANO_RIGHT.pkl", "rb"), encoding="latin1")
out = {k: (v if isinstance(v, str) or sp.issparse(v) else np.array(v)) for k, v in d.items()}
pickle.dump(out, open("wilor/WiLoR/mano_data/MANO_RIGHT.pkl", "wb"), protocol=2)
EOF
  fi
}

mkdir -p mano
case "$WHAT" in
  model) setup_model ;;
  corpus) setup_model; setup_corpus ;;
  extraction) setup_model; setup_extraction ;;
  all) setup_model; setup_corpus; setup_extraction ;;
  *) echo "usage: bash third-party/setup.sh {model|corpus|extraction|all}" >&2; exit 2 ;;
esac
echo "done"
