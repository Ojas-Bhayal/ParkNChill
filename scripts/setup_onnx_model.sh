#!/usr/bin/env bash
# Downloads the referenced YOLOX-s ONNX model only when missing, or when --force is supplied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/../models"
TARGET_FILE="${TARGET_DIR}/vehicle_detector.onnx"
MODEL_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
FORCE="${1:-}"

mkdir -p "${TARGET_DIR}"
if [[ -s "${TARGET_FILE}" && "${FORCE}" != "--force" ]]; then
  SIZE="$(wc -c < "${TARGET_FILE}" | tr -d ' ')"
  if (( SIZE > 1000000 )); then
    echo "Existing vehicle model found; leaving it unchanged."
    echo "File: ${TARGET_FILE} (${SIZE} bytes)"
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "${TARGET_FILE}"; else shasum -a 256 "${TARGET_FILE}"; fi
    echo "Use --force only to replace this file."
    exit 0
  fi
fi

echo "Downloading YOLOX-s ONNX model from: ${MODEL_URL}"
curl -fL "${MODEL_URL}" -o "${TARGET_FILE}"
SIZE="$(wc -c < "${TARGET_FILE}" | tr -d ' ')"
if (( SIZE < 1000000 )); then echo "Downloaded model is unexpectedly small (${SIZE} bytes)." >&2; exit 1; fi
if command -v sha256sum >/dev/null 2>&1; then sha256sum "${TARGET_FILE}"; else shasum -a 256 "${TARGET_FILE}"; fi
echo "Installed ${TARGET_FILE}; run 'mvn clean test' and 'mvn javafx:run'."
