"""Activation surgery / steering via the AR (text -> vector -> inject into base model).

The NLA "surgery on activations" case study: write the concept you want the model
to think in ENGLISH, run it through the AR to get a layer-32 activation vector,
overwrite the planning token's activation in base Gemma with it, and watch the
generation change.

  baseline:  prompt -> base Gemma generates normally
  steered :  hook layer 32; at the last prompt token, replace h(32) with
             AR(concept) rescaled to the ambient norm; regenerate

Usage:
  python tutorials/steer.py
  python tutorials/steer.py --prompt "Roses are red," --concept "the ocean, waves, and salt water" --strength 1.0
  python tutorials/steer.py --concept "cheese and dairy" --max-new 40 --steer-decode

Requires the AV server running (for the optional read of the ORIGINAL plan).
"""
import os, sys, gc, argparse, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla_inference import NLAClient, NLACritic

BASE, LAYER = "google/gemma-3-12b-it", 32

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", default="Write the next line of this poem:\n\nRoses are red,")
ap.add_argument("--concept", default="cheese, cheddar, and dairy products",
                help="English description; AR turns this into the injected vector")
ap.add_argument("--strength", type=float, default=1.0,
                help="0=no edit, 1=full replace of the planning activation")
ap.add_argument("--max-new", type=int, default=40)
ap.add_argument("--steer-decode", action="store_true",
                help="also overwrite each decoded token's activation (stronger, can derail)")
ap.add_argument("--sglang-url", default="http://localhost:30000")
ap.add_argument("--no-av", action="store_true", help="skip the AV read of the original plan")
args = ap.parse_args()

dev = "cuda"
print(f"\n=== PROMPT ===\n{args.prompt}\n=== CONCEPT TO INJECT ===\n{args.concept}\n")

# ---- load base + build chat input -----------------------------------------
tok = AutoTokenizer.from_pretrained(BASE)
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
enc = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                              add_generation_prompt=True, tokenize=True,
                              return_tensors="pt", return_dict=True).to(dev)
ids = enc["input_ids"]
n = ids.shape[1]
p = n - 1                                   # planning token = last prompt position

# ---- baseline generation (greedy) -----------------------------------------
with torch.no_grad():
    base_out = m.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
base_text = tok.decode(base_out[0][n:], skip_special_tokens=True)

# ---- capture the original planning activation (norm + optional AV read) ----
with torch.no_grad():
    hs = m(input_ids=ids, output_hidden_states=True).hidden_states[LAYER + 1][0]
h_orig = hs[p].float().cpu().numpy()
orig_norm = float(np.linalg.norm(h_orig))

orig_plan = None
if not args.no_av:
    AV_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-av")
    av = NLAClient(AV_DIR, sglang_url=args.sglang_url, device="cpu")
    orig_plan = av.generate(h_orig, temperature=0.7, max_new_tokens=120)

# ---- build the steering vector from the concept via the AR -----------------
AR_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-ar")
ar = NLACritic(AR_DIR, device=dev)
steer_raw = ar.reconstruct(args.concept).numpy()           # raw [d]
steer_vec = steer_raw / np.linalg.norm(steer_raw) * orig_norm   # match ambient norm
target = (1 - args.strength) * h_orig + args.strength * steer_vec
target = target / np.linalg.norm(target) * orig_norm        # renorm to ambient
target_t = torch.tensor(target, dtype=m.dtype, device=dev)
del ar
gc.collect()
torch.cuda.empty_cache()

# ---- hook: overwrite layer-32 OUTPUT at the planning token -----------------
def hook(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1:                       # prefill: overwrite position p
        h[:, p, :] = target_t.to(h.dtype)
    elif args.steer_decode:                  # decode: overwrite the new token
        h[:, -1, :] = target_t.to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

def _decoder_layers(model):
    # Gemma-3-it is multimodal: text layers nest under .language_model.
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers", "model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            print(f"[steer] decoder layers at: {path} ({len(obj)} layers)")
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers ModuleList")

handle = _decoder_layers(m)[LAYER].register_forward_hook(hook)
with torch.no_grad():
    steer_out = m.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
handle.remove()
steer_text = tok.decode(steer_out[0][n:], skip_special_tokens=True)

# ---- report ----------------------------------------------------------------
print("=" * 70)
print(f"planning token = {tok.decode([ids[0, p].item()])!r}  (position {p}, ||h||={orig_norm:.0f})")
if orig_plan:
    print(f"\nAV reads the ORIGINAL plan at that token:\n  {orig_plan}\n")
print("-" * 70)
print(f"BASELINE generation:\n  {base_text!r}\n")
print(f"STEERED  generation (injected {args.concept!r}, strength={args.strength}):\n  {steer_text!r}")
print("=" * 70)
