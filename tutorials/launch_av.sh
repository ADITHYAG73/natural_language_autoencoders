#!/bin/bash
export HF_HOME=/workspace/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace/natural_language_autoencoders
exec python -m sglang.launch_server --model-path kitft/nla-gemma3-12b-L32-av \
  --port 30000 --disable-radix-cache --mem-fraction-static 0.35 --trust-remote-code \
  --attention-backend triton --sampling-backend pytorch
