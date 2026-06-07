"""Full NLA round-trip on YOUR sentence:
   base Gemma -> h(layer 32, token k) -> AV -> text -> AR -> reconstructed h -> (mse, cos)

Usage:  python /workspace/round_trip.py "your passage here (60+ tokens is best)"
Reuses the repo's own classes (NLAClient = AV, NLACritic = AR) from nla_inference.py.
"""
import os, sys, gc, torch, numpy as np
# Make `import nla_inference` work no matter where this script is launched from:
# the repo root is the parent of this tutorials/ dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "google/gemma-3-12b-it"
LAYER = 32                      # NLA extraction layer -> hidden_states[LAYER+1]
MIN_P = 50                      # _MIN_POSITION: earlier tokens decode to noise

text = " ".join(sys.argv[1:]).strip() or (
    "The French Revolution began in 1789 amid a fiscal crisis and widespread "
    "anger at the monarchy. As bread prices soared and the Estates-General "
    "convened, ordinary citizens in Paris grew restless, and by July the city "
    "stood on the brink of open revolt against King Louis XVI.")

print("\n=== INPUT TEXT ===\n" + text + "\n")

# ---- 1. Extract real activations from the base model ----------------------
print("[1/3] loading base Gemma + extracting layer-32 activations ...")
tok = AutoTokenizer.from_pretrained(BASE)
ids = tok(text, return_tensors="pt").to("cuda")
n = ids["input_ids"].shape[1]
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
with torch.no_grad():
    hs = m(**ids, output_hidden_states=True).hidden_states[LAYER + 1][0]   # [seq, 3840]

# choose up to 4 positions >= 50 (incl. the last token); fall back if too short
toklist = ids["input_ids"][0].tolist()
cands = list(range(MIN_P, n))
if not cands:
    print("  !! only %d tokens; need >= %d for meaningful decode. "
          "Using last token %d anyway (expect noise)." % (n, MIN_P, n - 1))
    cands = [n - 1]
picks = sorted(set([cands[0], cands[len(cands) // 3], cands[2 * len(cands) // 3], n - 1]))
acts = []
for p in picks:
    v = hs[p].float().cpu().numpy()
    acts.append((p, tok.decode([toklist[p]]), v, float(np.linalg.norm(v))))
del m
gc.collect()
torch.cuda.empty_cache()
print("  extracted %d activations at positions %s (seq len %d)"
      % (len(acts), [a[0] for a in acts], n))

# ---- 2. AV: activation -> description (via the running sglang server) ------
print("[2/3] AV verbalizing each activation ...")
from nla_inference import NLAClient, NLACritic
AV_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-av")
av = NLAClient(AV_DIR, sglang_url="http://localhost:30000", device="cpu")
descs = [av.generate(v, temperature=0.7, max_new_tokens=200) for (_, _, v, _) in acts]

# ---- 3. AR: description -> reconstructed vector -> score vs original -------
print("[3/3] loading AR + scoring reconstructions ...")
AR_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-ar")
ar = NLACritic(AR_DIR, device="cuda")

print("\n" + "=" * 70)
print(" RESULTS  (cos~0.9 => mse~0.2 good ; cos~0 => mse~2 orthogonal)")
print("=" * 70)
for (p, tokstr, v, norm), d in zip(acts, descs):
    mse, cos = ar.score(d, v)
    print("\n--- position %d  token=%r  ||h||=%.1f ---" % (p, tokstr, norm))
    print("AV says: " + d)
    print("AR reconstruction:  mse_nrm=%.3f   cos=%.3f" % (mse, cos))
print("\n" + "=" * 70)
