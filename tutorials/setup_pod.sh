#!/bin/bash
# =============================================================================
# NLA Pod Setup — strictly follows docs/setup.md (inference only path)
# https://github.com/ADITHYAG73/natural_language_autoencoders/blob/main/docs/setup.md
#
# Run this once on a fresh RunPod instance:
#   bash tutorials/setup_pod.sh
# =============================================================================

set -e

echo "============================================"
echo " NLA Pod Setup (inference only)"
echo " Following: docs/setup.md"
echo "============================================"

# -----------------------------------------------------------------------------
# 0. Clone the repo (if not already inside it)
# -----------------------------------------------------------------------------
if [ ! -f "nla_inference.py" ]; then
    echo ""
    echo "[0] Cloning repo..."
    git clone https://github.com/ADITHYAG73/natural_language_autoencoders.git
    cd natural_language_autoencoders
else
    echo ""
    echo "[0] Already inside repo."
fi

# -----------------------------------------------------------------------------
# 1. Install dependencies — exactly as docs/setup.md says
#    NOTE: pin torch to cu124 to avoid sgl-kernel cu12/cu13 conflict
# -----------------------------------------------------------------------------
echo ""
echo "[1] Installing dependencies (from docs/setup.md)..."

pip install -q uv  # install uv if not present

uv pip install torch transformers safetensors httpx orjson pyyaml numpy \
    --index-url https://download.pytorch.org/whl/cu124

uv pip install "sglang[all]>=0.5.6"

echo "    Done."

# -----------------------------------------------------------------------------
# 2. Launch SGLang server — exactly as README quick start says
#    (run in background, log to sglang.log)
# -----------------------------------------------------------------------------
echo ""
echo "[2] Launching SGLang server with AV checkpoint..."
echo "    model : kitft/nla-qwen2.5-7b-L20-av"
echo "    port  : 30000"
echo "    --disable-radix-cache is required (input_embeds, no token IDs)"
echo "    Logs  : sglang.log"
echo ""

python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --dtype bfloat16 > sglang.log 2>&1 &

echo "    SGLang PID: $!"
echo "    Waiting 60s for server to be ready..."
sleep 60

# health check
python -c "
import httpx
try:
    r = httpx.get('http://localhost:30000/health', timeout=10)
    print('    SGLang server is UP and healthy.')
except Exception as e:
    print(f'    Server not yet ready: {e}')
    print('    Check sglang.log — wait a bit more before running hello_world.py')
"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Now run:"
echo "   python tutorials/hello_world.py"
echo ""
echo " Or follow the README quick start directly:"
echo "   python nla_inference.py kitft/nla-qwen2.5-7b-L20-av \\"
echo "       --sglang-url http://localhost:30000 \\"
echo "       --parquet path/to/activations.parquet"
echo ""
echo " To watch SGLang logs:  tail -f sglang.log"
echo "============================================"
