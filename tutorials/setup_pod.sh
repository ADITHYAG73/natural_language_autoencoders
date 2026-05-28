#!/bin/bash
# =============================================================================
# NLA Pod Setup Script
# Run this once on a fresh RunPod instance to get everything ready.
# Usage: bash setup_pod.sh
# =============================================================================

set -e  # stop immediately if any command fails

echo "============================================"
echo " NLA Pod Setup"
echo "============================================"

# -----------------------------------------------------------------------------
# 1. System deps
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl unzip

# -----------------------------------------------------------------------------
# 2. Install Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers safetensors httpx orjson pyyaml numpy pyarrow datasets anthropic
pip install -q "sglang[all]>=0.5.6"

echo "      torch + sglang installed."

# -----------------------------------------------------------------------------
# 3. Clone the repo
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Cloning NLA repo..."
if [ ! -d "natural_language_autoencoders" ]; then
    git clone https://github.com/ADITHYAG73/natural_language_autoencoders.git
else
    echo "      Repo already exists, pulling latest..."
    cd natural_language_autoencoders && git pull && cd ..
fi
cd natural_language_autoencoders

# -----------------------------------------------------------------------------
# 4. Download Qwen2.5-7B-Instruct (base model for activation extraction)
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Downloading Qwen2.5-7B-Instruct (base model)..."
echo "      This is ~15GB — will take a few minutes..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('  Downloading model weights...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='bfloat16')
print('  Qwen2.5-7B-Instruct downloaded.')
"

# -----------------------------------------------------------------------------
# 5. Download NLA checkpoints (AV + AR for Qwen layer 20)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Downloading NLA checkpoints (AV + AR)..."
echo "      AV checkpoint: kitft/nla-qwen2.5-7b-L20-av"
echo "      AR checkpoint: kitft/nla-qwen2.5-7b-L20-ar"
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('  Downloading AV tokenizer + weights...')
AutoTokenizer.from_pretrained('kitft/nla-qwen2.5-7b-L20-av')
AutoModelForCausalLM.from_pretrained('kitft/nla-qwen2.5-7b-L20-av', torch_dtype='bfloat16')
print('  AV downloaded.')
print('  Downloading AR tokenizer + weights...')
AutoTokenizer.from_pretrained('kitft/nla-qwen2.5-7b-L20-ar')
# AR is a truncated model — load via AutoModel not AutoModelForCausalLM
from transformers import AutoModel
AutoModel.from_pretrained('kitft/nla-qwen2.5-7b-L20-ar', torch_dtype='bfloat16')
print('  AR downloaded.')
"

# -----------------------------------------------------------------------------
# 6. Launch SGLang server with AV checkpoint
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Launching SGLang server with AV checkpoint..."
echo "      Server will run in the background on port 30000."
echo "      --disable-radix-cache is required (input_embeds have no token IDs)"

python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 \
    --disable-radix-cache \
    --dtype bfloat16 &

SGLANG_PID=$!
echo "      SGLang PID: $SGLANG_PID"
echo "      Waiting 30 seconds for server to be ready..."
sleep 30

# quick health check
python -c "
import httpx, sys
try:
    r = httpx.get('http://localhost:30000/health', timeout=10)
    print('  SGLang server is UP.')
except Exception as e:
    print(f'  WARNING: server not responding yet: {e}')
    print('  Wait a bit more and then run: python tutorials/hello_world.py')
    sys.exit(0)
"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo " Next step — run the hello world:"
echo "   python tutorials/hello_world.py"
echo ""
echo " SGLang server is running on http://localhost:30000"
echo " To stop it:  kill $SGLANG_PID"
echo ""
