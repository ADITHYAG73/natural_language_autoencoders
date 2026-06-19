#!/bin/bash
# One-shot datagen env setup on a RunPod pod (everything on /workspace).
# NOT Miles, NOT sglang — data-gen is decoupled. Just uv + transformers>=5.5 + datagen deps.
set -e
cd /workspace
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
source "$HOME/.local/bin/env"
[ -d natural_language_autoencoders ] || git clone -q https://github.com/ADITHYAG73/natural_language_autoencoders.git
cd natural_language_autoencoders
uv venv >/dev/null 2>&1 || true
echo "=== installing deps (slow part — torch CUDA + transformers + datasets) ==="
uv pip install --python .venv/bin/python \
    "transformers>=5.5" torch pyarrow datasets anthropic accelerate \
    numpy pyyaml orjson safetensors huggingface_hub 2>&1 | tail -4
echo "=== verify ==="
.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
echo "=== disk on /workspace after install ==="
df -h /workspace | tail -1
