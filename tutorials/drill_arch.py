"""Drill-down: print the ACTUAL distinguishing torch modules (meta device, no weights).

- E4B: top-level text-model children (find the PLE / AltUp / Laurel modules) + one decoder layer
- 26B-A4B: one decoder layer (the MoE block)
- 31B: one decoder layer (clean dense reference)
"""
import warnings; warnings.filterwarnings("ignore")
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

REPOS = {
    "E4B (PLE)":  "google/gemma-4-E4B",
    "26B-A4B (MoE)": "google/gemma-4-26B-A4B",
    "31B (dense)": "google/gemma-4-31B",
}

def text_model(repo):
    cfg = AutoConfig.from_pretrained(repo)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg)
    # navigate to the text decoder
    lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
    return lm

for label, repo in REPOS.items():
    print("\n" + "#" * 80 + f"\n#  {label}   ({repo})\n" + "#" * 80)
    lm = text_model(repo)
    print(">> text-model top-level children (name : type):")
    for name, child in lm.named_children():
        n = len(list(child.children())) if hasattr(child, "children") else 0
        extra = f"  [{len(child)} entries]" if hasattr(child, "__len__") else ""
        print(f"    {name:28s} {type(child).__name__}{extra}")
    print("\n>> decoder layer [0] (full):")
    print(lm.layers[0])
