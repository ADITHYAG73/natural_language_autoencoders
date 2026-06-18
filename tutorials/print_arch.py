"""Side-by-side architecture comparison: Gemma-3-12B vs Gemma-4 variants.

NO weights downloaded, NO GPU. Fetches only each model's config.json (KB), builds
the module tree on the META device (init_empty_weights → params are shape-only,
zero memory), prints the authentic torch tree, and renders a side-by-side PNG.

Run (in the uv env):  /tmp/archviz/bin/python tutorials/print_arch.py
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

OUT_PNG = "tutorials/gemma_arch_compare.png"

# Gemma-4 = Apache-2.0 (ungated). Gemma-3-12B is gated → hardcoded reference.
MODELS = [
    ("Gemma-3-12B\n(reference)", None),          # gated → use known values
    ("Gemma-4 E4B",   "google/gemma-4-E4B"),
    ("Gemma-4 31B",   "google/gemma-4-31B"),
    ("Gemma-4 26B-A4B", "google/gemma-4-26B-A4B"),
]

# Known Gemma-3-12B-it values (config gated; established in prior sessions).
G3_12B = dict(model_type="gemma3_text", num_hidden_layers=48, hidden_size=3840,
              num_attention_heads=16, num_key_value_heads=8, head_dim=256,
              vocab_size=262144, sliding_window=1024, ple=None, experts=None, top_k=None)


def _get(tc, *names, default=None):
    for n in names:
        v = getattr(tc, n, None)
        if v is not None:
            return v
    return default


def extract(repo):
    cfg = AutoConfig.from_pretrained(repo)
    tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    ple = _get(tc, "hidden_size_per_layer_input")
    experts = _get(tc, "num_experts", "num_local_experts")
    # some configs carry the field but disable it
    if not _get(tc, "enable_moe_block", default=True if experts else False):
        experts = experts  # keep; we report what's present
    return cfg, dict(
        model_type=_get(tc, "model_type"),
        num_hidden_layers=_get(tc, "num_hidden_layers"),
        hidden_size=_get(tc, "hidden_size"),
        num_attention_heads=_get(tc, "num_attention_heads"),
        num_key_value_heads=_get(tc, "num_key_value_heads"),
        head_dim=_get(tc, "head_dim"),
        vocab_size=_get(tc, "vocab_size"),
        sliding_window=_get(tc, "sliding_window"),
        ple=ple,
        experts=experts,
        top_k=_get(tc, "num_experts_per_tok"),
    )


def meta_tree(cfg, label):
    """Build the module tree on meta (zero memory) and return its repr."""
    try:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
        return str(model)
    except Exception as e:  # some multimodal autoclasses differ; fall back to text model
        return f"[could not instantiate {label}: {type(e).__name__}: {e}]"


rows = []
trees = {}
for label, repo in MODELS:
    if repo is None:
        rows.append((label, G3_12B)); continue
    try:
        cfg, fields = extract(repo)
        rows.append((label, fields))
        print(f"\n{'='*78}\n  {label}  ({repo})\n{'='*78}")
        tree = meta_tree(cfg, label)
        trees[label] = tree
        # print a compact slice of the tree (full tree is long for 60-layer models)
        print(tree[:2500] + ("\n  …(truncated)…" if len(tree) > 2500 else ""))
    except Exception as e:
        print(f"\n[skip {label} ({repo}): {type(e).__name__}: {e}]")
        rows.append((label, {"model_type": f"ERR: {type(e).__name__}"}))

# ---- render the side-by-side PNG -------------------------------------------
attrs = [
    ("model_type", "model_type"),
    ("num_hidden_layers", "# layers"),
    ("hidden_size", "d_model"),
    ("num_attention_heads", "attn heads"),
    ("num_key_value_heads", "KV heads (GQA)"),
    ("head_dim", "head_dim"),
    ("vocab_size", "vocab"),
    ("sliding_window", "sliding window"),
    ("ple", "Per-Layer Embeds (PLE)"),
    ("experts", "MoE experts"),
    ("top_k", "MoE top-k"),
]
ncol = len(rows)
fig, ax = plt.subplots(figsize=(2.3 + 2.2 * ncol, 0.55 * (len(attrs) + 2)))
ax.axis("off")
ax.set_title("Gemma-3 vs Gemma-4 — architecture (from config.json, meta-device)", fontsize=12, weight="bold", pad=14)

col_x = [0.30 + i * (0.70 / max(1, ncol - 1)) * (ncol - 1) / ncol for i in range(ncol)]
# simpler even spacing:
col_x = [0.30 + (i + 0.5) * (0.68 / ncol) for i in range(ncol)]
y0, dy = 0.92, 0.075

for i, (label, _f) in enumerate(rows):
    ax.text(col_x[i], y0, label, ha="center", va="center", fontsize=10, weight="bold")

for r, (key, disp) in enumerate(attrs):
    y = y0 - (r + 1) * dy
    ax.text(0.02, y, disp, ha="left", va="center", fontsize=9, weight="bold", color="#333")
    for i, (label, f) in enumerate(rows):
        v = f.get(key)
        txt = "—" if v in (None, 0) else f"{v:,}" if isinstance(v, int) and key in ("vocab_size",) else str(v)
        color, weight = "#111", "normal"
        if key == "ple" and v:
            txt, color, weight = f"YES ({v})", "#c0392b", "bold"
        if key == "ple" and not v:
            txt = "none"
        if key == "experts" and v:
            color, weight = "#d35400", "bold"
        if key == "experts" and not v:
            txt = "dense"
        ax.text(col_x[i], y, txt, ha="center", va="center", fontsize=9, color=color, weight=weight)

ax.text(0.02, y0 - (len(attrs) + 1.4) * dy,
        "Red = PLE (per-layer embeddings, threatens NLA injection).  Orange = MoE.  "
        "No weights downloaded — config-only, meta device.",
        ha="left", va="center", fontsize=7.5, color="#666")

plt.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nPNG written → {OUT_PNG}")
