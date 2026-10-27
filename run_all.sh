#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
GEN_PY="$PY"
if [ -x /workspace/.venv_gen/bin/python ]; then
  GEN_PY=/workspace/.venv_gen/bin/python
fi
export PYTHONUNBUFFERED=1

WORK=$("$PY" -m flipaware.common workdir)
LOGDIR="$WORK/logs"
mkdir -p "$LOGDIR"
echo "logs: $LOGDIR"

stop_pod() {
  if [ "${FLIP_STOP_POD:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap stop_pod EXIT

STAGES=(
  "s00_gpu_check"
  "s01_data"
  "gen:s02_generate --set main"
  "s03_filter --set main"
  "s04_splits"
  "gen:s02_generate --set xgen"
  "s03_filter --set xgen"
  "s07_dataset"
  "s08_models"
  "s09_human"
  "s10_analysis"
  "s11_finalize"
  "s12_logs"
)
FROM="${FLIP_FROM:-0}"

for i in "${!STAGES[@]}"; do
  [ "$i" -lt "$FROM" ] && continue
  entry="${STAGES[$i]}"
  py="$PY"
  if [[ "$entry" == gen:* ]]; then
    py="$GEN_PY"
    entry="${entry#gen:}"
  fi
  read -r mod args <<< "$entry"
  tag="$(printf '%02d' "$i")_${mod}${args:+_${args//[^a-z]/}}"
  echo "=== stage $i $mod ${args:-}" | tee -a "$LOGDIR/$tag.log"
  "$py" -m "flipaware.$mod" ${args:-} 2>&1 | tee -a "$LOGDIR/$tag.log"
done

echo "=== all stages finished"
