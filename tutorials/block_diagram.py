"""Block diagram of a Gemma-4 decoder layer (dense canonical block) + the E4B (PLE)
and 26B-A4B (MoE) deltas. Boxes/labels are the REAL torch module names (from drill_arch).
No weights, no GPU — this just draws.
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 11))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("Gemma-3 & Gemma-4 — the SHARED decoder block (×N) + Gemma-4 deltas\n"
             "(real torch modules, meta device — no weights)",
             fontsize=13, weight="bold", pad=10)

CX, W = 0.50, 0.26   # center column
def box(x, y, w, h, text, fc="#eef3fb", ec="#3b6ea5", fs=9.5, weight="normal", tc="#111"):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.006",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight, color=tc)

def arrow(x1, y1, x2, y2, color="#444", lw=1.6, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}"))

# ---- center column: dense Gemma-4 decoder block (top → bottom) --------------
ys = dict(xin=0.93, iln=0.855, attn=0.745, paln=0.635, add1=0.565,
          pfln=0.495, mlp=0.385, pfln2=0.275, add2=0.205, xout=0.115)
box(CX, ys["xin"], W, 0.05, "hidden state  x  (residual stream, d_model)", fc="#fff7e6", ec="#c08a2d", weight="bold")
box(CX, ys["iln"], W, 0.05, "input_layernorm  (RMSNorm)")
box(CX, ys["attn"], W, 0.115,
    "Self-Attention  (GQA + RoPE)\nq_proj · k_proj · v_proj\nq_norm · k_norm  ( · v_norm ← Gemma-4 ONLY )\n→ o_proj", fc="#e9f3ec", ec="#3a7d44", fs=9)
box(CX, ys["paln"], W, 0.05, "post_attention_layernorm")
box(CX, ys["add1"], 0.05, 0.045, "⊕", fc="#fde9e9", ec="#b03a3a", fs=15, weight="bold")
box(CX, ys["pfln"], W, 0.05, "pre_feedforward_layernorm")
box(CX, ys["mlp"], W, 0.095,
    "MLP  (GeGLU)\ndown_proj( GELU(gate_proj(x)) ⊙ up_proj(x) )", fc="#e9f3ec", ec="#3a7d44")
box(CX, ys["pfln2"], W, 0.05, "post_feedforward_layernorm")
box(CX, ys["add2"], 0.05, 0.045, "⊕", fc="#fde9e9", ec="#b03a3a", fs=15, weight="bold")
box(CX, ys["xout"], W, 0.05, "x  →  next layer", fc="#fff7e6", ec="#c08a2d", weight="bold")

order = ["xin", "iln", "attn", "paln", "add1", "pfln", "mlp", "pfln2", "add2", "xout"]
for a, b in zip(order, order[1:]):
    arrow(CX, ys[a] - 0.026, CX, ys[b] + 0.026)

# residual bypasses (right side)
rx = CX + W/2 + 0.03
arrow(CX + W/2, ys["xin"], rx, ys["xin"], color="#b03a3a", lw=1.3, style="-")
arrow(rx, ys["xin"], rx, ys["add1"], color="#b03a3a", lw=1.3, style="-")
arrow(rx, ys["add1"], CX + 0.025, ys["add1"], color="#b03a3a", lw=1.3)
arrow(CX + W/2, ys["add1"], rx, ys["add1"], color="#b03a3a", lw=1.3, style="-")
arrow(rx, ys["add1"], rx, ys["add2"], color="#b03a3a", lw=1.3, style="-")
arrow(rx, ys["add2"], CX + 0.025, ys["add2"], color="#b03a3a", lw=1.3)
ax.text(rx + 0.005, (ys["xin"] + ys["add1"]) / 2, "residual", rotation=90, va="center", fontsize=7.5, color="#b03a3a")

# ---- LEFT: E4B PLE delta (red) ---------------------------------------------
lx = 0.135
ax.text(lx, 0.965, "E4B only — PLE", ha="center", fontsize=11, weight="bold", color="#c0392b")
box(lx, 0.80, 0.24, 0.085, "embed_tokens_per_layer\n(2nd table, keyed on TOKEN-ID)\n→ per_layer_model_projection",
    fc="#fdecea", ec="#c0392b", fs=8.5)
box(lx, 0.55, 0.24, 0.085, "per_layer_input_gate (d→256)\nper_layer_projection (256→d)\npost_per_layer_input_norm",
    fc="#fdecea", ec="#c0392b", fs=8.5)
arrow(lx, 0.80 - 0.045, lx, 0.55 + 0.045, color="#c0392b")
arrow(lx + 0.12, 0.55, CX - W/2, 0.50, color="#c0392b", rad=-0.2)
ax.text(lx, 0.44,
        "Injected into EVERY layer,\nkeyed on the token ID.\nA single-token activation\ninjection at the marker does\nNOT replace this → dilutes\nthe injected concept.",
        ha="center", va="top", fontsize=8, color="#c0392b")

# ---- RIGHT: 26B-A4B MoE delta (orange) -------------------------------------
mxr = 0.865
ax.text(mxr, 0.965, "26B-A4B only — MoE", ha="center", fontsize=11, weight="bold", color="#d35400")
box(mxr, 0.40, 0.24, 0.11,
    "router (RMSNorm → proj d→128)\n⇒ top-k of 128 experts\nGemma4TextExperts\n(+ shared dense mlp)",
    fc="#fdf0e3", ec="#d35400", fs=8.5)
arrow(CX + W/2, ys["mlp"], mxr - 0.12, 0.40, color="#d35400", rad=0.15)
ax.text(mxr, 0.305, "Replaces / augments the MLP\n(injection path unaffected —\nno PLE here).",
        ha="center", va="top", fontsize=8, color="#d35400")

ax.text(0.5, 0.045,
        "CENTER BLOCK = Gemma-3-12B (48L, d=3840)  ≡  Gemma-4-31B dense (60L, d=5376) — same block lineage;",
        ha="center", fontsize=8.5, color="#2c3e50", weight="bold")
ax.text(0.5, 0.02,
        "Gemma-4's ONLY structural change vs Gemma-3 is v_norm in attention.  RMSNorm 4-norm sandwich · GQA+RoPE · sliding/full · embed ×√d.",
        ha="center", fontsize=7.5, color="#555")

plt.savefig("tutorials/gemma4_block_diagram.png", dpi=160, bbox_inches="tight")
print("wrote tutorials/gemma4_block_diagram.png")
