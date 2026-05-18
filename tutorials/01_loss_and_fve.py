"""
Tutorial 01 — The NLA Loss L and FVE (Fraction of Variance Explained)

Paper equation (Method section):

    L = E_{h_l ~ H} [ E_{z ~ AV(·|h_l)} [ ||h_l - AR(z)||² ] ]

    FVE = 1 - L / E_{h_l ~ H} [ ||h_l - h̄_l||² ]

We simulate this without any real models — using toy vectors and toy
AV/AR functions. The math is identical to what the real codebase computes.

Run with:  python tutorials/01_loss_and_fve.py
"""

import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. The activation distribution H
#    In reality: extract hidden_states[layer] from a real LLM on a corpus.
#    Here: random vectors in R^d, simulating a distribution of activations.
# ---------------------------------------------------------------------------

D_MODEL = 8       # tiny d_model (Qwen uses 3584, we use 8 for readability)
N_SAMPLES = 1000  # number of activation vectors drawn from H

# Draw N activation vectors — this is our corpus of h_l values
activations = np.random.randn(N_SAMPLES, D_MODEL)  # shape: [N, D]

print("=" * 60)
print("STEP 1 — Activation distribution H")
print("=" * 60)
print(f"  d_model        = {D_MODEL}")
print(f"  corpus size    = {N_SAMPLES}")
print(f"  activations shape: {activations.shape}")
print(f"  one example h_l: {activations[0].round(3)}")


# ---------------------------------------------------------------------------
# 2. Toy AV — activation verbalizer (vector → text → encoded as vector)
#    In reality: inject h_l into an LLM, autoregress a text description z.
#    Here: we mock the AV as a noisy linear projection (simulates imperfect
#    compression through the language bottleneck).
# ---------------------------------------------------------------------------

def toy_av(h, noise_scale=0.5):
    """
    Toy AV: takes activation h, returns a 'description' z.
    We represent z as a vector here (in reality z is text tokens,
    but the AR maps it back to a vector anyway).
    Noise simulates the information lost through the language bottleneck.
    """
    z = h + np.random.randn(*h.shape) * noise_scale  # noisy copy
    return z


# ---------------------------------------------------------------------------
# 3. Toy AR — activation reconstructor (text → vector)
#    In reality: feed z (text) into truncated LM + linear head → h_hat.
#    Here: identity (AR just returns z as-is, as if it's a perfect inverter).
# ---------------------------------------------------------------------------

def toy_ar(z):
    """
    Toy AR: takes description z, returns reconstructed activation h_hat.
    In reality this is NLACriticModel.forward() extracting at tokens[-1].
    """
    return z  # identity — AR perfectly inverts the AV's output


# ---------------------------------------------------------------------------
# 4. Compute the loss L
#
#    L = (1/N) * sum_i [ ||h_i - AR(AV(h_i))||² ]
#
#    The double expectation in the paper:
#      - outer E: average over activations h_l drawn from corpus H
#      - inner E: average over explanations z sampled from AV(·|h_l)
#    Here AV is deterministic so inner E collapses to a single sample.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 2 — Computing loss L")
print("=" * 60)

reconstruction_errors = []

for i, h in enumerate(activations):
    z = toy_av(h)           # AV: h → z (with noise)
    h_hat = toy_ar(z)       # AR: z → h_hat
    error = np.sum((h - h_hat) ** 2)   # ||h - AR(z)||²  (squared L2 norm)
    reconstruction_errors.append(error)

    if i < 3:  # print first 3 samples so you can see what's happening
        print(f"\n  Sample {i}:")
        print(f"    h     = {h.round(3)}")
        print(f"    z     = {z.round(3)}")
        print(f"    h_hat = {h_hat.round(3)}")
        print(f"    ||h - h_hat||² = {error:.4f}")

L = np.mean(reconstruction_errors)   # outer expectation = average over corpus
print(f"\n  L (reconstruction loss) = {L:.4f}")
print(f"  Intuition: average squared distance between original and reconstructed vector")


# ---------------------------------------------------------------------------
# 5. Compute the FVE denominator
#
#    Denominator = E[ ||h_l - h̄_l||² ]
#
#    h̄_l is the MEAN activation vector across the corpus.
#    This is the variance of the activation distribution — how spread out
#    the activations are. A fixed constant, independent of the NLA.
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 — Computing FVE denominator (baseline variance)")
print("=" * 60)

h_bar = activations.mean(axis=0)   # mean activation vector, shape [D]
print(f"  Mean activation h̄ = {h_bar.round(3)}")

baseline_errors = []
for h in activations:
    baseline_error = np.sum((h - h_bar) ** 2)   # ||h - h̄||²
    baseline_errors.append(baseline_error)

denominator = np.mean(baseline_errors)
print(f"  Denominator (data variance) = {denominator:.4f}")
print(f"  Intuition: if you always predicted the mean, this is your average error")


# ---------------------------------------------------------------------------
# 6. Compute FVE
#
#    FVE = 1 - L / denominator
#
#    FVE = 1  → perfect reconstruction
#    FVE = 0  → no better than predicting the mean
#    FVE < 0  → worse than predicting the mean (terrible model)
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 — FVE (Fraction of Variance Explained)")
print("=" * 60)

FVE = 1 - (L / denominator)
print(f"  L           = {L:.4f}")
print(f"  denominator = {denominator:.4f}")
print(f"  FVE         = 1 - {L:.4f}/{denominator:.4f} = {FVE:.4f}")


# ---------------------------------------------------------------------------
# 7. Intuition check — vary noise_scale and watch FVE change
#    Perfect AV (no noise) → FVE should approach 1
#    Terrible AV (huge noise) → FVE should approach 0 or go negative
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 — Intuition check: how does noise affect FVE?")
print("=" * 60)
print(f"  {'noise_scale':>12} | {'L':>10} | {'FVE':>10}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}")

for noise in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
    errors = [np.sum((h - toy_ar(toy_av(h, noise))) ** 2) for h in activations]
    l = np.mean(errors)
    fve = 1 - l / denominator
    print(f"  {noise:>12.1f} | {l:>10.4f} | {fve:>10.4f}")

print("""
Observations:
  - noise=0.0  → FVE=1.0  (perfect reconstruction, AR recovers h exactly)
  - noise~1.0  → FVE~0.0  (AV loses as much info as the data variance)
  - noise>>1   → FVE<0    (AV is worse than predicting the mean every time)

This is exactly what the paper reports improving during RL training —
FVE goes from ~0.3 after SFT to ~0.75 after RL for Qwen2.5-7B.
""")


# ---------------------------------------------------------------------------
# 8. Connection to the real codebase
# ---------------------------------------------------------------------------

print("=" * 60)
print("WHERE THIS LIVES IN THE REAL CODEBASE")
print("=" * 60)
print("""
  This tutorial:          Real codebase:
  ─────────────────────────────────────────────────────────────
  activations array    →  nla/datagen/stage0_extract.py
                          (extracts hidden_states[layer] from LLM)

  toy_av()             →  nla/rollout/nla_generate.py generate()
                          (SGLang input_embeds inference)

  toy_ar()             →  nla/models.py NLACriticModel.forward()
                          (truncated LM + linear head)

  ||h - AR(z)||²       →  nla/loss.py nla_critic_loss()
                          (exact same computation, line by line)

  h_bar / denominator  →  nla/schema.py compute_predict_mean_baselines()
                          (precomputed from parquet, loaded at train time)

  FVE                  →  nla/loss.py (logged during training)
                          (also reported in paper Figure 2)
""")
