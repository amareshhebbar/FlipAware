#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -q -U pip
python -m pip install -q -r requirements-base.txt
python -m pip install -q -U -r requirements.txt

PLATFORM=$(python -m flipaware.common)
echo "platform: $PLATFORM"

if [ "$PLATFORM" = "runpod" ]; then
  CC=$(python -c "import torch; print(torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0)")
  if [ "$CC" -ge 8 ] && [ ! -x /workspace/.venv_gen/bin/python ]; then
    python -m venv /workspace/.venv_gen
    /workspace/.venv_gen/bin/pip install -q -U pip
    /workspace/.venv_gen/bin/pip install -q vllm pyyaml
  fi
fi

TRACKERS="${FLIP_TRACKERS:-tensorboard,mlflow}"
for t in ${TRACKERS//,/ }; do
  case "$t" in
    mlflow) python -m pip install -q mlflow "pyarrow<26" || echo "mlflow install failed, it will be skipped" ;;
    aim) python -m pip install -q aim || echo "aim install failed, it will be skipped" ;;
    clearml) python -m pip install -q clearml || echo "clearml install failed, it will be skipped" ;;
  esac
done
echo "trackers: $TRACKERS"

if [ "${FLIP_ONNX:-0}" = "1" ]; then
  python -m pip install -q "optimum[onnxruntime]" || echo "optimum install failed, onnx export will be skipped"
fi

if [ "${FLIP_MTEB:-0}" = "1" ]; then
  python -m pip install -q mteb || echo "mteb install failed, mteb will be skipped"
fi

python -c "import sentence_transformers, transformers, peft, datasets; print('sentence_transformers', sentence_transformers.__version__, 'transformers', transformers.__version__)"
echo "setup done"
