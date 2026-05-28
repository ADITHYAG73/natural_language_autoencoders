"""
Tutorial — Hello World
======================
The simplest possible end-to-end NLA run:

    sentence → extract activation vector (layer 20 of Qwen2.5-7B)
             → AV describes it in English
             → AR scores the description (MSE + cosine similarity)

Run after setup_pod.sh:
    python tutorials/hello_world.py

Or with your own sentence:
    python tutorials/hello_world.py --sentence "The cat sat on the mat."
"""

import argparse
import sys
import torch
import numpy as np

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
parser.add_argument("--av-checkpoint", type=str,
                    default="kitft/nla-qwen2.5-7b-L20-av")
parser.add_argument("--ar-checkpoint", type=str,
                    default="kitft/nla-qwen2.5-7b-L20-ar")
parser.add_argument("--base-model", type=str,
                    default="Qwen/Qwen2.5-7B-Instruct")
args = parser.parse_args()

print("=" * 60)
print(" NLA Hello World")
print("=" * 60)
print(f"  Sentence    : {args.sentence}")
print(f"  Layer       : {args.layer}")
print(f"  Token index : {args.token_index} (last token if -1)")
print()

# ── Step 1: Extract activation vector from Qwen ───────────────────────────────
print("[Step 1] Loading Qwen2.5-7B-Instruct and extracting activation...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()

inputs = tokenizer(args.sentence, return_tensors="pt").to("cuda")
token_ids = inputs["input_ids"][0]
tokens = [tokenizer.decode([t]) for t in token_ids]

print(f"  Tokens ({len(tokens)}): {tokens}")

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# hidden_states[i] = output of layer i-1 (index 0 = embedding layer)
# hidden_states[layer+1] = output of transformer block `layer`
hidden_states = outputs.hidden_states
activation = hidden_states[args.layer + 1][0, args.token_index, :]  # shape: [d_model]

print(f"  Extracted activation at token '{tokens[args.token_index]}'")
print(f"  Shape: {activation.shape}  (d_model = {activation.shape[0]})")
print(f"  Norm:  {activation.norm().item():.2f}")
print(f"  First 8 values: {activation[:8].float().cpu().numpy().round(3)}")

# Convert to numpy float32 for NLA client
activation_np = activation.float().cpu().numpy()

# Free Qwen from GPU memory before loading NLA models
del model
torch.cuda.empty_cache()
print("  Qwen freed from GPU memory.")

# ── Step 2: Run AV — describe the activation in English ───────────────────────
print()
print("[Step 2] Running AV (Activation Verbalizer)...")
print("         Loading NLA client...")

sys.path.insert(0, ".")  # make sure nla_inference.py is importable
from nla_inference import NLAClient, NLACritic, load_nla_config

config = load_nla_config(args.av_checkpoint, tokenizer_path=args.av_checkpoint)
print(f"  injection_scale : {config.injection_scale}")
print(f"  d_model         : {config.d_model}")

client = NLAClient(
    checkpoint=args.av_checkpoint,
    sglang_url=args.sglang_url,
    config=config,
)

print("  Sending activation to SGLang server...")
explanation = client.generate(activation_np)

print()
print("  ┌─────────────────────────────────────────────────┐")
print(f"  │ AV Explanation:                                 │")
for line in explanation.split(". "):
    if line.strip():
        print(f"  │   {line.strip()[:50]:<50} │")
print("  └─────────────────────────────────────────────────┘")

# ── Step 3: Run AR — reconstruct the vector from the explanation ──────────────
print()
print("[Step 3] Running AR (Activation Reconstructor)...")
print("         Scoring the explanation...")

critic = NLACritic(checkpoint=args.ar_checkpoint)
score = critic.score(explanation, activation_np)

print()
print("  Round-trip scoring:")
print(f"  MSE (normalized)  : {score['mse_nrm']:.4f}  (0=perfect, 2=orthogonal)")
print(f"  Cosine similarity : {score['cosine']:.4f}  (1=perfect, 0=orthogonal)")
print(f"  FVE               : {score['fve_nrm']:.4f}  (1=perfect, 0=no better than mean)")

print()
print("=" * 60)
print(" Summary")
print("=" * 60)
print(f"  Input   : '{args.sentence}'")
print(f"  Token   : '{tokens[args.token_index]}' at layer {args.layer}")
print(f"  AV said : {explanation[:100]}...")
print(f"  MSE     : {score['mse_nrm']:.4f}  |  Cosine: {score['cosine']:.4f}")
print()
print("  If MSE < 0.5 and cosine > 0.85 — the explanation is good.")
print("  If output looks like Chinese characters — injection failed.")
print("  See docs/inference.md for debugging.")
