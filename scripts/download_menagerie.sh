#!/usr/bin/env bash
# --------------------------------------------------------------------------
# download_menagerie.sh
#
# Downloads the Franka Emika Panda model from mujoco_menagerie and places it
# in the local models/ directory as a fallback for when mujoco_menagerie is
# not installed at the system level.
#
# Usage:
#   cd mujoco_learn/mujoco_panda
#   bash scripts/download_menagerie.sh
# --------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(realpath "${SCRIPT_DIR}/../models")"
MENAGERIE_REPO="https://github.com/google-deepmind/mujoco_menagerie.git"
PANDA_SRC="franka_emika_panda"

echo "==> Models target: ${MODELS_DIR}"

if [ "${1:-}" = "--full" ]; then
    echo "==> Cloning full mujoco_menagerie ..."
    TMPDIR="$(mktemp -d)"
    git clone --depth 1 "${MENAGERIE_REPO}" "${TMPDIR}/menagerie"
    cp -r "${TMPDIR}/menagerie/${PANDA_SRC}" "${MODELS_DIR}/"
    rm -rf "${TMPDIR}"
    echo "==> Done. Panda model at ${MODELS_DIR}/${PANDA_SRC}/"
    exit 0
fi

echo "==> Sparse-checkout of ${PANDA_SRC} from mujoco_menagerie ..."
TMPDIR="$(mktemp -d)"
cd "${TMPDIR}"
git clone --depth 1 --filter=blob:none --sparse "${MENAGERIE_REPO}" menagerie
cd menagerie
git sparse-checkout set "${PANDA_SRC}"
cp -r "${PANDA_SRC}" "${MODELS_DIR}/"
rm -rf "${TMPDIR}"

echo "==> Done. Panda model at ${MODELS_DIR}/${PANDA_SRC}/"
