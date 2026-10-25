set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -q -U pip
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

python -c "import sentence_transformers, transformers, peft, datasets; print('sentence_transformers', sentence_transformers.__version__, 'transformers', transformers.__version__)"
echo "setup done"
