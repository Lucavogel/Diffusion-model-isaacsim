#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_latest_checkpoint.sh [device] [checkpoint_path] [output_dir]
# Examples:
#   bash scripts/eval_latest_checkpoint.sh cuda:0
#   bash scripts/eval_latest_checkpoint.sh cuda:0 /workspace/diffusion_policy/data/outputs/.../latest.ckpt

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DP_DIR="${ROOT_DIR}/diffusion_policy"

DEVICE="${1:-cuda:0}"
CHECKPOINT="${2:-}"
OUTPUT_DIR="${3:-}"

if [[ ! -d "${DP_DIR}" ]]; then
    echo "Error: diffusion_policy directory not found at ${DP_DIR}" >&2
    exit 1
fi

if [[ -z "${CHECKPOINT}" ]]; then
    # Prefer explicit latest.ckpt files, then fallback to any .ckpt
    CHECKPOINT="$(find "${DP_DIR}/data/outputs" -type f -name "latest.ckpt" 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "${CHECKPOINT}" ]]; then
        CHECKPOINT="$(find "${DP_DIR}/data/outputs" -type f -name "*.ckpt" 2>/dev/null | sort | tail -n 1 || true)"
    fi
fi

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "Error: no checkpoint found. Pass a checkpoint path as 2nd argument." >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    ts="$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DP_DIR}/data/eval_outputs/e_waste_eval_${ts}"
fi

echo "[eval] checkpoint: ${CHECKPOINT}"
echo "[eval] output_dir: ${OUTPUT_DIR}"
echo "[eval] device: ${DEVICE}"

cd "${DP_DIR}"
python eval.py --checkpoint "${CHECKPOINT}" --output_dir "${OUTPUT_DIR}" --device "${DEVICE}"

echo "[eval] done"
echo "[eval] log: ${OUTPUT_DIR}/eval_log.json"
