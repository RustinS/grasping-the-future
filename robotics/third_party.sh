#!/usr/bin/env bash
# Fetch SPIDER (facebookresearch/spider, CC BY-NC 4.0) at the commit the paper used, unmodified, into robotics/spider/.
#
#   bash robotics/third_party.sh        # clone and check out the pinned commit
#   bash robotics/third_party.sh env    # also create the SPIDER environment robotics/.venv (needs python3.12; set PYTHON)
#
# The scripts find the clone through common.SPIDER (robotics/spider, or the SPIDER_ROOT environment variable).
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PIN=44717007de41cbef7565dff7ff9f4453557a2d3d
DEST="${SPIDER_ROOT:-$HERE/spider}"

if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/facebookresearch/spider.git "$DEST"
fi
git -C "$DEST" checkout --quiet --detach "$PIN"
test "$(git -C "$DEST" rev-parse HEAD)" = "$PIN"
if [[ -n "$(git -C "$DEST" status --porcelain --untracked-files=no)" ]]; then
  echo "warning: $DEST has local modifications; the experiment uses SPIDER unmodified" >&2
fi
echo "SPIDER @ $PIN in $DEST"

if [[ "${1:-}" == "env" ]]; then
  PYTHON="${PYTHON:-python3.12}"
  if [[ ! -x "$HERE/.venv/bin/python" ]]; then
    "$PYTHON" -m venv "$HERE/.venv"
  fi
  "$HERE/.venv/bin/python" -m pip install --upgrade pip
  "$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements.txt"
  echo "SPIDER environment: $HERE/.venv"
fi
