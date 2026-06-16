# Gemma-4 NLA onboarding notes

Working notes for the long-term goal: **train + contribute a Gemma-4 AV/AR** to this
repo. This is the TRAINING pipeline (data-gen → AV-SFT → AR-SFT → RL), not the
inference recipe (that's the `nla-runpod-inference` skill, which covers Gemma-3).

Status as of 2026-06-15: **substrate check done, variant not yet chosen.**

---

## 1. Substrate check — is Gemma-4 trainable today? YES (path is open)

- **Released** April 2, 2026. Variants: E2B, E4B, 12B Unified (June 3), 26B-A4B (MoE), 31B (dense).
- **License: Apache 2.0 — NOT gated.** No HF token / license-acceptance step (unlike Gemma-3). Big win.
- **transformers:** Gemma-4 supported since **v5.5.0** (`model_type: gemma4`). We used 5.3.0 for Gemma-3 → must bump to ≥5.5.0.
- **sglang:** supported (PR #21952, merged Apr 7 2026); cookbook docs for E2B/E4B/26B-A4B. Multimodal (vision+audio).
- **Version-stack implication:** Gemma-4 forces a bump ABOVE our proven Gemma-3 combo
  → newer sglang + transformers ≥5.5.0 + likely torch 2.11. Needs a fresh re-test on
  Blackwell, and the repo's sglang `input_embeds` patch re-checked against the newer sglang.

## 2. Architecture comparison (from config.json)

| Field | Gemma-3-12B (ref) | G4 E4B | G4 31B | G4 26B-A4B |
|---|---|---|---|---|
| model_type | gemma3 / gemma3_text | gemma4 / gemma4_text | gemma4 / gemma4_text | gemma4 / gemma4_text |
| architecture | Gemma3ForConditionalGen | Gemma4ForConditionalGen | Gemma4ForConditionalGen | Gemma4ForConditionalGen |
| num_hidden_layers | 48 | 42 | 60 | 30 |
| hidden_size (d_model) | 3840 | 2560 | 5376 | 2816 |
| heads / KV-heads | 16/8* | 8 / 2 | 32 / 16 | 16 / 8 |
| head_dim | 256 | 256 | 256 | 256 |
| vocab | 262144 | 262144 | 262144 | 262144 |
| context | 128K | 131K | 262K | 262K |
| tie_word_embeddings | true | true | true | true |
| **Per-Layer Embeddings (PLE)** | none | **YES** (hidden_size_per_layer_input:256) | none | none |
| **MoE** | dense | dense | dense | **128 experts, top-8** |
| sliding-window | 1024, 5:1 | 512, alt | 1024, 5:1 | 1024, alt |
| embed-scale field | absent (√d in code) | absent | absent | absent |

\*Gemma-3-12B heads from memory (its config is gated, not re-verified). d_model 3840,
48 layers, vocab 262144 confirmed from last session; AV/AR were trained at **layer 32**.

## 3. What each field means for NLA

- **model_type `gemma4_text`** → one-line edit: add `"gemma4"`/`"gemma4_text"` to
  `_EMBED_SCALES` in `nla/arch_adapters.py`. (Touches: embed-scale resolution.)
- **num_hidden_layers + hidden_size** → pick the extraction LAYER and set the activation
  vector LENGTH. Gemma-3 used 32/48 (~⅔ depth). Analogues: E4B ~28/42, 31B ~40/60,
  26B-A4B ~20/30. Bigger d_model = bigger vectors = more compute. (Touches: every stage.)
- **tie_word_embeddings: true** → same as Gemma-3; AR stripping lm_head→Identity already
  handled. No new work.
- **embed-scale "absent" in config** → Gemma applies ×√d IN MODEL CODE, not via config.
  `arch_adapters` keys it off model_type, so `gemma4 → sqrt_d_model` is the right default
  — BUT must be verified against Gemma-4's actual embedding class. (Touches: AV injection;
  the ~62× footgun.)
- **PLE (per-layer embeddings) — THE KEY RISK** → E4B has it; 31B & 26B-A4B don't. PLE looks
  up an EXTRA embedding from token IDs at each layer. NLA injection sends `input_embeds`
  (no `input_ids`) → PLE may have no token IDs to look up → injection path may break or be
  diluted. 31B / 26B-A4B sidestep this (standard single embed_tokens, like Gemma-3).
  (Touches: AV injection — potential dealbreaker for E2B/E4B.)
- **MoE (26B-A4B only)** → affects only the FFN; residual stream NLA extracts is still
  standard, AR just loads experts for its kept layers. Manageable; RL on an MoE policy adds
  load-balancing wrinkles the repo hasn't been tested on. (Touches: AR truncation, RL.)
- **sliding-window pattern** → alternating sliding/full like Gemma-3. For NLA's short
  prompts, sliding == full (window covers the prompt), so the triton-backend reasoning from
  the Gemma-3 skill still applies.

## 4. Key insight + variant tension (the decision still to make)

**None of the SMALL Gemma-4 models are a clean fit:** the cheap ones (E2B/E4B) use **PLE**,
which is exactly what could break NLA's `input_embeds` injection. The architecturally clean
choices are the **big 31B (dense)** or **26B-A4B (MoE)**.

- **E4B** — cheapest, but real **injection-path research risk** (PLE + input_embeds).
  Solving it would itself be a contribution, but risky for a first.
- **26B-A4B** — injection works (no PLE); 4B *active* params keep rollout cheap-ish; MoE is
  the only wrinkle. **Current lean for a first contribution.**
- **31B** — most likely to "just work" (closest to Gemma-3) but heaviest (60×5376).

## 5. Economics — Claude API + RL training (researched 2026-06-15)

### Claude API (data-gen stage 2) — what it does + cost
- **Where:** `nla/datagen/stage2_api_explain.py` (provider = `AnthropicProvider`). Runs ONLY on the
  SL subset (av_sft 0.25 + ar_sft 0.25 = **50% of vectors**); the RL 50% skips the API.
- **Role:** Claude is shown the **source TEXT snippet** (`detokenized_text_truncated`) — NOT the
  activation vector — and asked for the "2-3 features the LM would use to predict the next token …
  what it's thinking about where the text ends" (`<analysis>` tags, ~80-100 words). That output
  becomes the **AV-SFT target** (`response`) AND the **AR-SFT `<text>`**. It bootstraps the
  supervised stage; RL then refines the AV with no Claude in the loop.
- **Volumes (explicit in `configs/datagen/gemma3_12b_*.yaml`):** 1k smoke = ~10k vectors / **~5k API
  calls**; 100k full = ~1M vectors / **~500k API calls**. `positions_per_doc: 10`, `max_tokens: 300`.
- **Cost (pricing: Sonnet 4.6 $3/$15 per 1M; Haiku 4.5 $1/$5; ~600 in + ~250 out per call):**
  - 1k smoke on Haiku ≈ **~$10** (fits the $25 budget).
  - 100k full on Sonnet ≈ **~$2,500–3,000** (~$1,400 with the Batches API, −50%).
  - Levers: Haiku not Sonnet, Batches API, fewer docs, `cache_from` reuse (same tokenizer only).
- **Key fact:** Claude never sees the vector — it can't, it only reads the text prefix. So this is a
  text-derived SFT label, not a vector readout.

### RL loop — architecture, GPUs, cost (from `configs/rl.sh` + `configs/TRAINING_NOTES.md`)
- **Three simultaneous GPU consumers:** actor (AV, GRPO/FSDP) + critic (AR, MSE — the ONLINE reward
  model, `reward = −MSE(AR(expl), gold)` via Ray remote) + sglang rollout. Plus a reference model for
  the KL term (`KL_LOSS_COEF=0.01`; set 0 to drop ref-load on tiny runs). Separate Ray GPU pools.
- **Not single-GPU.** `rl.sh` defaults `ACTOR_GPUS=8 CRITIC_GPUS=4 ROLLOUT_GPUS=4`; the **released
  Qwen-7B run was 2 nodes × 8 × H100-80GB = 16 H100s**, ~4,200 rollouts. Env vars let you shrink the
  pools, but ~2-3 distinct GPUs is the realistic floor (untested territory — repo only tested 2×8).
- **On-policy only:** synchronous `train.py`, one optimizer step per rollout (128 prompts × 8 =
  1024 global batch). `train_async.py`/overlap is explicitly "not tested, may hurt." Don't deviate.
- **Cost estimates (~$2.5/H100-hr):**

  | Stage | Config (repo) | Wall-clock | GPU-h | ~Cost |
  |---|---|---|---|---|
  | AV-SFT | 2×H100, 1000 steps @4.97s | ~1.4h | ~2.8 | ~$7 |
  | AR-SFT | 2×H100, 1000 steps @3s | ~0.8h | ~1.7 | ~$4 |
  | **Full RL** | **16×H100, ~4200 rollouts @47s** | **~55h** | **~880** | **~$2,200–3,000** |

  90% of each RL step is sglang rollout wait; smaller models (E4B) are proportionally cheaper/faster.
- **Newcomer reality:** **SFT is the gentle on-ramp** (plain `train.py`, ~2 GPUs, hours, ~$10-20,
  no Ray-multi-group) — a newcomer CAN do AV-SFT + AR-SFT and see a working rough AV/AR.
  **Full RL is heavy**: Miles + Ray + sglang-from-source + patches (+ optional Megatron), repeatedly
  flagged "only config we tested." A **tiny RL smoke** (E4B, ~2-3 GPUs, ~30-100 steps like the repo's
  LR scans) to learn the dynamics is ~**$15-50** — after surviving the Miles install (a bigger setup
  gauntlet than the inference one in the skill).

### Bottom line for the Gemma-4 goal
- **Learnable now (~$25-65 total):** 1k smoke data-gen (~$10 API) → AV-SFT + AR-SFT on a small model
  (~$15-20 GPU) → optional tiny RL smoke (~$15-50). This teaches the WHOLE pipeline cheaply.
- **Full contributed checkpoint (12B-class):** ~$2.5-3k GPU + ~$2.5k API ≈ **~$5k and weeks**, on
  16-GPU infra. Do this only after the smoke stages work and there's real budget.
- **Staged path stands:** familiarize on the smoke scale first; never commit full-scale upfront.

## 6. SFT pipelines — complete path, file-by-file (verified 2026-06-16)

Both SFT stages run through Miles' `train.py` with `--debug-train-only` (NO sglang rollout — this is
the cheap ~2-GPU on-ramp), `--data-source-path nla.data_source.NLADataSource`, and
`--custom-actor-cls-path nla.train_actor.NLAFSDPActor`. They differ in prompt shape, what the
activation is *for*, and the loss.

### Data-gen prerequisite (shared)
`model_presets.py` → `stage0_extract.py` (raw layer-L activations, positions ≥50) →
`stage1_split.py` (doc-level: av_sft .25 / ar_sft .25 / rl .50) → `stage2_api_explain.py` (Claude
writes the explanation from the TEXT snippet) → `stage3_build.py` (writes the per-stage parquet +
`nla_meta.yaml` sidecar).

### AV-SFT (`configs/actor_sft.sh`)  — "vector → text"
- **Parquet (Stage 3a):** `prompt` = `list[dict]` messages with the `<INJECT>` marker in user content;
  `response` = the `<explanation>…</explanation>` (Claude's); `activation_vector` = raw vector.
- **Flags:** `--loss-type sft_loss` (Miles built-in), `--rollout-function-path nla.rollout.sft_actor.generate_rollout`,
  `--nla-injection-scale <INJ_SCALE>`, `n_samples_per_prompt 1`. Qwen defaults: batch 256, lr 2e-5 cosine→2e-6.
- **Runtime:**
  1. `NLADataSource` (`data_source.py:33`) → `Sample(prompt=messages, metadata={activation_vector, response})`; `<INJECT>`→ sidecar marker char.
  2. `sft_actor.generate_rollout` (`rollout/sft_actor.py`) → NO generation: appends `response` as assistant turn, tokenizes, builds a **response-only `loss_mask`** (`loss_mask[-response_length:]`), stashes the vector in `multimodal_train_inputs[MM_ACTIVATION_KEY]`.
  3. `NLAFSDPActor` (`train_actor.py:410` `injects = loss_type in (sft_loss, policy_loss)`) → during forward, **`inject_at_marked_positions`** overwrites the marker token's embedding with the `injection_scale`-normalized activation (input_embeds — *same injection as inference*).
  4. Loss = **Miles' `sft_loss`** (standard next-token CE, upstream `miles/.../loss.py`), **masked to the response tokens** via the mask in step 2. Backprop updates the **full AV**.
- **Objective:** given the injected layer-L activation at the marker, reproduce Claude's explanation. → DCP `iter_XXXX` ckpt (→ HF via `tools/convert_fsdp_to_hf.py` for inference; or fed to RL as `ACTOR_SFT_CKPT`).

### AR-SFT (`configs/critic_sft.sh`)  — "text → vector"
- **Prereq — `nla.scripts.prepare_critic_checkpoint`:** base model → keep blocks 0..K (K+1 layers),
  **strip lm_head + final LN**, **identity-init `value_head`** (`torch.eye(d)` — critical; random init
  starts pred_norm ~1/√3 off, per TRAINING_NOTES), `config.num_hidden_layers = K+1`, write
  `nla_meta.yaml` (token IDs/templates copied from the DATASET sidecar). = `CRITIC_INIT_CKPT`.
- **Parquet (Stage 3b):** `prompt` is a **complete formatted STRING**:
  `"Summary of the following text: <text>{explanation}</text> <summary>"`; `activation_vector` = the **gold** target.
- **Flags:** `--nla-model-is-critic`, `--loss-type custom_loss --custom-loss-function-path nla.loss.nla_critic_loss`,
  `--rollout-function-path nla.rollout.sft_critic.generate_rollout`.
- **Runtime:**
  1. `sft_critic.generate_rollout` (`rollout/sft_critic.py`) → tokenize the string prompt with
     `add_special_tokens=True` (BOS matters — matches stage0 extractor; without it layer-K means go OOD),
     **extraction = LAST token** (guaranteed by the fixed `</text> <summary>` suffix — no marker scan;
     `verify_critic_suffix` checks drift once), stash gold activation. `loss_mask=[]`, `response_length=0`.
  2. `NLACriticModel` forward → `value_head` output at every position; `nla_critic_loss` (`loss.py:44`)
     takes the **last-token** prediction and computes **MSE vs the gold activation**, with BOTH L2-normalized
     to `mse_scale` (direction-only MSE = 2(1−cos); `fve` logged per step).
- **Objective:** reconstruct the layer-L activation vector from the explanation text. → DCP ckpt (→ HF;
  this is the `-ar` checkpoint, and the RL `CRITIC_SL_CKPT`).

### AV vs AR at a glance
| | AV-SFT | AR-SFT |
|---|---|---|
| model | full LM | truncated K+1 layers + `value_head`, no lm_head/final-LN |
| prompt | `list[dict]` w/ marker | complete string `…<text>{expl}</text> <summary>` |
| activation is… | **injected input** (at marker) | **gold output target** |
| extraction | — | last token (suffix-anchored) |
| loss | Miles `sft_loss` (response-masked CE) | `nla_critic_loss` (MSE at last token, dir-only) |
| learns | vector → explanation text | explanation text → vector |

Both are ~2-GPU, ~hours, ~$10-20 each (TRAINING_NOTES). This is the recommended first hands-on stage
for Gemma-4 before any RL.

### Verification pass (2026-06-16) — train-time specifics + Gemma-4 watch-items
Re-verified the §6 claims against code; all hold. Four facts that matter for Gemma-4 training:

1. **Train-time injection rides the model's native embedding scaling — NO manual `embed_scale` needed**
   (unlike inference). `train_actor.py:525` registers a **forward hook on `get_input_embeddings()`**;
   it fires on the embedding layer's *output*, so for Gemma the normal tokens are already ×√d-scaled by
   `Gemma3TextScaledWordEmbedding.forward()`, and the activation (normalized to `injection_scale`) is
   slotted in after. So the √d footgun is an *inference-only* concern. ✓
2. **`use_cache=False` is forced at train time** (`train_actor.py:544+`) to kill two **Gemma3-specific**
   bugs: sliding-window attn picking a different mask when a DynamicCache exists (ref vs actor logprob
   divergence), and thd-packed cross-sequence contamination. Gemma-4 is also sliding-window → this fix
   is Gemma-relevant; **re-confirm it still applies for `gemma4`** (the code/comments target gemma3).
3. **PLE is a *train-time* risk too, not just inference.** The injection hook overwrites the **main
   `embed_tokens`** output at the marker; a PLE model (E4B) adds per-layer embeddings deeper, keyed on
   the marker's *token ID*, which the overwrite does NOT replace → injected concept diluted at every
   layer. Reinforces: **prefer a non-PLE Gemma-4 variant (31B dense / 26B-A4B) for a first AV/AR.**
4. **AR-SFT build is tokenizer-pinned:** `stage3_build.py` tokenizes every row and asserts it ends with
   the critic suffix (`</text> <summary>`). For Gemma-4, confirm that suffix tokenizes to a stable
   trailing sequence at the BPE boundary, or the template needs a delimiter (the assert will catch it).

Not yet traced file-by-file: the **RL loop internals** (`reward.py` `nla_rm`, `nla_generate.py`,
critic_fwd-via-Ray) — §5 has its cost/feasibility, not its mechanics. That's the next mapping if needed.

## 7. Open questions / next decisions (not yet done)

1. **Pick the variant** (E4B vs 26B-A4B vs 31B) — trades cost vs risk vs cleanliness.
2. **Verify embed-scale** = √d for Gemma-4 (check the Gemma4 embedding class in transformers 5.5+).
3. **If E4B/PLE:** investigate whether `input_embeds` injection survives PLE (likely the blocker).
4. **Code diff to write:** add gemma4 to `nla/arch_adapters.py` `_EMBED_SCALES` +
   register the model in `nla/datagen/model_presets.py` + pick an injection marker token.
5. **Re-establish the version stack** on a pod: sglang(gemma4) + transformers≥5.5.0 + torch 2.11,
   re-apply/re-check the sglang input_embeds patch.
6. **Then** the staged run: data-gen + AV-SFT first (cheap, learn the flow) → AR-SFT → RL.

Sources: HF gemma4 blog, ai.google.dev gemma4 model card, transformers gemma4 docs (v5.5.0),
sglang PR #21952; config.json of google/gemma-4-{E4B,31B,26B-A4B}.
