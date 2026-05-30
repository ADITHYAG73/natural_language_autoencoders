"""
Tutorial — Hello World
======================
The simplest possible end-to-end NLA run:

    sentence → extract activation vector (layer 20 of Qwen2.5-7B)
             → AV describes it in English   (NLAClient → SGLang)
             → AR scores the description     (NLACritic → MSE + cosine)

Prereq: SGLang server already running with the AV checkpoint
(setup_pod.sh does this). Then:

    python tutorials/hello_world.py
    python tutorials/hello_world.py --sentence "The cat sat on the mat."

API NOTE — verified against nla_inference.py:
  - NLAClient(checkpoint_dir, sglang_url=...)        # loads its own config
  - client.generate(activation) -> str              # the explanation text
  - NLACritic(checkpoint_dir, device=...)
  - critic.score(explanation, original) -> (mse, cos)   # a TUPLE, not a dict
  - Both classes read nla_meta.yaml from a LOCAL dir, so we snapshot_download
    the checkpoints first and pass the local paths.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sentence", type=str,
                    default="He saw a carrot and had to grab it,",
                    help="Sentence to extract activation from")
parser.add_argument("--token-index", type=int, default=-1,
                    help="Which token position to extract (-1 = last token)")
parser.add_argument("--layer", type=int, default=20,
                    help="Which layer to extract from (default: 20)")
parser.add_argument("--sglang-url", type=str, default="http://localhost:30000",
                    help="SGLang server URL")
parser.add_argument("--av-repo", type=str, default="kitft/nla-qwen2.5-7b-L20-av")
parser.add_argument("--ar-repo", type=str, default="kitft/nla-qwen2.5-7b-L20-ar")
parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy / reproducible (matches examples/)")
args = parser.parse_args()

print("=" * 60)
print(" NLA Hello World")
print("=" * 60)
print(f"  Sentence    : {args.sentence}")
print(f"  Layer       : {args.layer}")
print(f"  Token index : {args.token_index} (last token if -1)")
print()

# ── Step 0: Download the NLA checkpoints to LOCAL dirs ────────────────────────
# NLAClient/NLACritic read nla_meta.yaml + safetensors from a local path,
# so a bare hub id ("kitft/...") won't work — snapshot_download first.
print("[Step 0] Downloading NLA checkpoints to local dirs...")
from huggingface_hub import snapshot_download

av_dir = snapshot_download(args.av_repo)
ar_dir = snapshot_download(args.ar_repo)
print(f"  AV local dir: {av_dir}")
print(f"  AR local dir: {ar_dir}")

# ── Step 1: Extract activation vector from Qwen ───────────────────────────────
print()
print("[Step 1] Loading Qwen2.5-7B-Instruct and extracting activation...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()

inputs = tokenizer(args.sentence, return_tensors="pt").to("cuda")
token_ids = inputs["input_ids"][0]
tokens = [tokenizer.decode([t]) for t in token_ids]
print(f"  Tokens ({len(tokens)}): {tokens}")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# hidden_states[0] = embeddings; hidden_states[i+1] = output of block i.
# So layer 20 output (what the NLA was trained on) is hidden_states[21].
activation = outputs.hidden_states[args.layer + 1][0, args.token_index, :]
print(f"  Extracted at token '{tokens[args.token_index]}' "
      f"(hidden_states[{args.layer + 1}])")
print(f"  Shape: {tuple(activation.shape)}   Norm: {activation.norm().item():.2f}")
print(f"  First 8 values: {activation[:8].float().cpu().numpy().round(3)}")

activation_np = activation.float().cpu().numpy()

# Free Qwen before loading the NLA models (not strictly needed on 80GB, but tidy)
del model
torch.cuda.empty_cache()
print("  Qwen freed from GPU memory.")

# Make nla_inference.py importable (it lives at the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nla_inference import NLAClient, NLACritic

# ── Step 2: Run AV — describe the activation in English ───────────────────────
print()
print("[Step 2] Running AV (Activation Verbalizer)...")
client = NLAClient(av_dir, sglang_url=args.sglang_url)  # loads config internally

print("  Sending activation to SGLang server...")
explanation = client.generate(activation_np, temperature=args.temperature)

print()
print("  AV Explanation:")
print("  " + "-" * 56)
for line in explanation.split("\n"):
    print(f"  {line}")
print("  " + "-" * 56)

# ── Step 3: Run AR — reconstruct + score ──────────────────────────────────────
print()
print("[Step 3] Running AR (Activation Reconstructor) and scoring...")
critic = NLACritic(ar_dir, device="cuda")

mse, cos = critic.score(explanation, activation_np)   # returns a TUPLE

print()
print("  Round-trip score:")
print(f"  MSE (direction)   : {mse:.4f}   (0=perfect, 2=orthogonal)")
print(f"  Cosine similarity : {cos:.4f}   (1=perfect, 0=orthogonal)")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(" Summary")
print("=" * 60)
print(f"  Input  : '{args.sentence}'")
print(f"  Token  : '{tokens[args.token_index]}' at layer {args.layer}")
print(f"  AV said: {explanation[:120]}...")
print(f"  MSE    : {mse:.4f}   |   Cosine: {cos:.4f}")
print()
print("  Reading the numbers (from nla_inference.py NLACritic docstring):")
print("    cos=0.9 → MSE=0.2   good decode (typical for clean positions)")
print("    cos=0.5 → MSE=1.0   mediocre")
print("    cos=0.0 → MSE=2.0   orthogonal")
print()
print("  If the AV output is Chinese/CJK characters → injection failed.")
print("  See docs/inference.md § 'Debugging: injection-failure smell'.")
