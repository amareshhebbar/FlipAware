#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
GEN_PY="$PY"
if [ -x /workspace/.venv_gen/bin/python ]; then
  GEN_PY=/workspace/.venv_gen/bin/python
fi

stop_pod() {
  if [ "${FLIP_STOP_POD:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap stop_pod EXIT

STAGES=(s00_gpu_check s01_data s02_generate s03_filter s04_splits s05_dataset s06_models s07_finalize)
FROM="${FLIP_FROM:-0}"

for i in "${!STAGES[@]}"; do
  [ "$i" -lt "$FROM" ] && continue
  st="${STAGES[$i]}"
  py="$PY"
  if [ "$st" = "s02_generate" ]; then
    py="$GEN_PY"
  fi
  echo "=== stage $i $st"
  "$py" -m "flipaware.$st"
done

echo "=== all stages finished"