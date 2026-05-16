# Natural Language Autoencoders — Codebase Navigation Map

## Project Overview

NLA trains two small fine-tuned LMs to map LLM residual-stream activation vectors to natural language and back. The two models are:
- **AV (Activation Verbalizer)**: `vector → text` — injects an activation as a single token embedding, autoregressively generates a description
- **AR (Activation Reconstructor)**: `text → vector` — reads the description, reconstructs the original direction via a truncated transformer + linear head

The codebase is a thin extension on top of **Miles** (RL training, FSDP2/Megatron) and **SGLang** (inference serving). Four model families are released: Qwen2.5-7B, Gemma-3-12B, Gemma-3-27B, Llama-3.3-70B.

---

## Top-Level Files

### `nla_inference.py` (733 lines)
Self-contained standalone inference client — no training deps needed.

Key classes:
- **`NLAClient`** — SGLang input_embeds inference driver
  - `generate()` — one activation vector → explanation text
  - `generate_batch()` — sequential batch requests
- **`NLACritic`** — optional reconstruction scorer
  - `reconstruct()` — explanation text → predicted vector
  - `score()` — returns MSE + cosine similarity vs original

Key helpers:
- `load_nla_config()` — parse sidecar YAML, assert tokenizer consistency
- `load_embedding_only()` — lazy-load just embedding weights from safetensors (avoids full model load)
- `normalize_activation()` — L2-norm rescaling
- `inject_at_marked_positions()` — core injection logic with neighbor validation
- `resolve_embed_scale()` — handles Gemma's √d embedding scale quirk

### `pyproject.toml`
Standard deps: `torch`, `transformers`, `safetensors`, `httpx`, `orjson`, `pyyaml`, `pyarrow`, `datasets`, `anthropic`. Miles/SGLang/Megatron installed separately.

---

## `nla/` Package

### Core Schema & Config

#### `schema.py` (235 lines)
**Single source of truth for the sidecar contract.** Pure Python, no ML imports.

Key constants:
| Constant | Value | Purpose |
|---|---|---|
| `EXPLANATION_OPEN/CLOSE` | `<explanation>` / `</explanation>` | Response wrapping tags |
| `INJECT_PLACEHOLDER` | `<INJECT>` | Prompt placeholder for parquets |
| `ACTIVATION_COLUMN` | `"activation_vector"` | Parquet column name |
| `SCALE_SQRT_D` | `"sqrt_d_model"` | Sidecar scale sentinel string |

Key classes/functions:
- **`NLATokenMeta`** — dataclass holding injection token IDs + neighbor IDs
- `normalize_activation()` — L2-norm scaling (None → no-op)
- `resolve_target_scale()` — converts sidecar scale values (None / `"sqrt_d_model"` / float)
- `compute_canonical_neighbors()` — tokenize prompt, extract left/right neighbor IDs
- `compute_predict_mean_baselines()` — FVE baseline computation from parquet
- `wrap_explanation()` / `extract_explanation()` — tag handling for AV responses
- `sidecar_path_for()` — resolve `nla_meta.yaml` path from dataset or model dir

#### `config.py` (297 lines)
Runtime config loading. Depends on `schema.py`.

Key class:
- **`NLAConfig`** — dataclass with all runtime hyperparameters
  - Token IDs, prompt templates, layer counts
  - `injection_scale` (actor hyperparam) and `mse_scale` (critic loss constant), independently configurable (None / `"sqrt_d_model"` / float)

Key functions:
- `load_nla_config()` — read YAML, assert injection char tokenization, verify neighbors against live tokenizer
- `load_nla_config_from_args()` — resolve sidecar source (CLI > model checkpoint > dataset), apply CLI overrides
- `resolve_sidecar_source()` — precedence chain: explicit path > HF checkpoint > prompt_data dir
- `verify_critic_suffix()` — one-time suffix tokenization check at dataset load
- `write_model_sidecar()` — write checkpoint sidecar with resolved floats + metadata

### Model Architecture

#### `models.py` (272 lines)
AR (critic) model wrapper.

Key class:
- **`NLACriticModel`** — wraps a truncated transformer (first K+1 layers only)
  - Final LayerNorm replaced with Identity (exposes raw residual stream)
  - Linear(d, d) `value_head`, no bias
  - Custom `from_pretrained()` handles layer truncation via `config.num_hidden_layers`
  - `forward()` returns `NLACriticOutput(values, backbone_last_hidden)`

Helpers:
- `load_embedding_only()` — lazy safetensors load for just embeddings
- `embed_dump_path()` — path for actor's embedding dump (reloaded by rollout)
- `_find_embed_key()`, `_truncate_config_layers()`, `_inner_transformer()` — architecture utilities

#### `arch_adapters.py` (165 lines)
**Architecture dispatch** — handles multimodal wrapper differences (Gemma-3 wraps a text model inside a multimodal class).

- `resolve_text_config()` — unwrap to text config
- `resolve_text_model()` — unwrap to text-only CausalLM
- `resolve_decoder_layers()` — find `.model.layers` or `.transformer.h`
- `resolve_embed_scale()` — returns 1.0 (Qwen/Llama) or √d (Gemma/T5)
- `is_multimodal_wrapper()` — detection helper

#### `injection.py` (94 lines)
**Pure, unit-testable injection logic.** No training framework deps.

Key function:
- `inject_at_marked_positions()` — scans `input_ids` for injection token, validates neighbors, overwrites embedding rows with activation vectors, asserts count == expected
  - Supports Megatron sequence-parallel via `seq_slice` parameter
  - Destroys process group on failure if distributed (prevents NCCL hangs)

### Data Loading & Loss

#### `data_source.py` (226 lines)
Parquet → training sample conversion. Extends Miles' `RolloutDataSource`.

Key class:
- **`NLADataSource`**
  - Loads parquets, auto-fetches remote sources (GCS, S3, etc.)
  - Reads `activation_vector` as raw numpy (avoids Python list intermediates)
  - Substitutes `<INJECT>` → injection char at load time
  - Handles two prompt formats: list[dict] (AV/RL) and str (AR)
  - Carries provenance: `doc_id`, `layer`, token counts
  - Uses `gc.freeze()` post-load to prevent GC scanning large arrays

#### `loss.py` (112 lines)
Critic MSE loss — Miles-compatible signature.

Key function:
- `nla_critic_loss()` — extracts prediction at last-token position (suffix-anchored, no scanning), normalizes both pred and gold to `mse_scale`, returns per-sample sums, logs loss / pred_norm / gold_norm / FVE
- `_get_gold_activation()` — reads from batch or `multimodal_train_inputs`

### Training & Rollout

#### `train_actor.py` (~1000 lines)
**Main training actor** — FSDP backend. Extends Miles' `FSDPTrainRayActor`.

Key class:
- **`NLAFSDPActor`**
  - `init()` — loads NLA config, registers injection hook, asserts `cp_size==1`
  - `get_model_cls()` — returns `NLACriticModel` if critic, else `AutoModelForCausalLM`
  - `_get_model_inputs_args()` — pops `nla_activation` from multimodal dict, normalizes, stores in `self._nla_vectors` for the forward hook
  - `_train_step()` — critic branch for value-head loss
  - `save_model()` — saves HF checkpoint + writes sidecar

  Critic-specific methods:
  - `_swap_rollout_to_critic_tokens()` — convert actor token sequences → critic token sequences, filter failed extractions
  - `_train_critic_loop()` — minimal loop (no log_probs, no advantages needed)
  - `_assert_reward_train_paths_agree()` — step-0 preflight check (padded critic_fwd == thread-packed train)
  - `_truncate_to_cross_rank_min()` — all-reduce to ensure consistent lengths across ranks
  - `_repartition_for_critic()` — handle asymmetric DP (actor_dp ≠ critic_dp)

#### `reward.py` (200+ lines)
Async reward computation. Miles-compatible.

Key function:
- `nla_rm()` — accumulates samples into batches, timeout-flushes tail stragglers (5s default), extracts explanation text, tokenizes, forwards on live critic via Ray remote, returns -MSE (or -log(MSE) if `NLA_LOG_MSE_REWARD=1`)
- `FAILED_EXTRACTION_REWARD = -2.0` — penalty for samples where explanation parsing failed

#### `rollout/sft_actor.py` (56 lines)
Actor SFT rollout — no generation.
- `generate_rollout()` — tokenizes prompt+response, computes loss mask, stashes `activation_vector` in `multimodal_train_inputs`

#### `rollout/sft_critic.py`
Critic SFT rollout — tokenizes critic prompt, no response needed.

#### `rollout/nla_generate.py` (200+ lines)
**RL rollout** via SGLang `input_embeds`.
- `generate()` — lazy-inits tokenizer/config/embedding table, checks for fresh embedding dump post-weight-sync, tokenizes prompt, embeds + applies arch scale (Gemma √d), normalizes activation, injects at marked positions, builds orjson payload, sends to SGLang, extracts `<explanation>` tags, stashes critic tokens in multimodal dict for simultaneous AR training
- Env vars: `NLA_BF16_B64_EMBEDS` (bf16 transport), `NLA_BYPASS_ROUTER` (routing workaround)

### Storage & Utility

#### `storage.py`
Abstract storage backend.
- **`Storage`** ABC — `open_write/read()`, `write_text()`, `read_text()`, `exists()`, `ensure_parent()`
- **`LocalStorage`** — pathlib-backed (shipped default)
- `_load_storage()` — dynamic import by class path string (pluggable via `--storage-cls`)

#### `tis_metrics.py` (35 lines)
Optional importance-sampling metrics — logs TIS ratios + K3 KL without modifying the GRPO objective.

---

## Data Generation Pipeline (`nla/datagen/`)

Four sequential stages, orchestrated by a YAML config:

### `run_pipeline.py`
Config-driven orchestrator. Reads YAML → shells out to each stage CLI in order.

### `stage0_extract.py`
**Extract activations from corpus.**
- Base model forward pass, capture `hidden_states[layer_index]`
- Per-doc keyed RNG ensures same `(seed, doc_id)` → same sampled positions regardless of chunk or process count
- Enforces `_MIN_POSITION = 50` (skip early tokens, noise-dominated)
- Output: `base.parquet` with raw unnormalized vectors, column `activation_vector`

### `stage1_split.py`
**Document-level 3-way split.** Default 30:30:40 (av_sft / ar_sft / rl).
- Partitions by unique `doc_id` — all rows from the same doc go to the same bucket
- Never splits positions from one doc across buckets

### `stage2_api_explain.py`
**Generate explanations via Anthropic API** (or other provider).
- Uses `nla.datagen.providers.CompletionProvider` (pluggable via `--provider-cls`)
- Caches results to avoid re-querying
- Ships `AnthropicProvider` as default

### `stage3_build.py`
**Build training-ready parquets from raw activations + explanations.**
- **av_sft**: prompt with `<INJECT>` placeholder + response wrapped in `<explanation>` tags
- **ar_sft**: critic prompt ending with `<summary>`, extraction at `tokens[-1]`
- **rl**: prompt only (no response)
- Asserts input `norm == "none"` (normalization must not have happened yet)
- Writes `nla_meta.yaml` sidecars with token IDs, prompt templates, scale factors

### Supporting datagen files

| File | Purpose |
|---|---|
| `extractors.py` | `HFExtractor` (model forward pass), `vLLMExtractor` stub |
| `providers.py` | `CompletionProvider` ABC, `AnthropicProvider` implementation |
| `sidecar.py` | Write dataset `nla_meta.yaml` sidecars |
| `model_presets.py` | Pre-baked configs for Qwen/Gemma/Llama |
| `shuffle_activations.py` | Permute vectors only (random direction baseline) |
| `stage_shuffle.py` | Row-level shuffle (breaks document clustering) |
| `merge_base.py` | Multi-node shard merging |
| `stage2_join.py` | Join explanations back to activation rows |
| `recover_explained.py` | Resume partial stage-2 runs |
| `cast_to_fixed_size_list.py` | Parquet schema normalization helper |

### `nla/datagen/README.md`
Full datagen documentation: stage descriptions, examples, backend swapping, smoke test.

---

## Configs (`configs/`)

### Training shell scripts

| File | What it does |
|---|---|
| `actor_sft.sh` | AV SFT — Miles train.py with injection hook, no reward |
| `critic_sft.sh` | AR SFT — Miles train.py with MSE loss only |
| `rl.sh` | RL — Miles train.py with GRPO (actor) + simultaneous AR training, async reward |

All scripts pass:
- `--custom-actor-cls-path nla.train_actor.NLAFSDPActor`
- `--data-source-path nla.data_source.NLADataSource`
- `--custom-rm-path nla.reward.nla_rm` (RL only)
- `--custom-generate-function-path nla.rollout.nla_generate.generate` (RL only)

### `configs/datagen/*.yaml`
Model-specific datagen configs (qwen7b, gemma3_12b, gemma3_27b, llama70b). Each specifies corpus, layer index, positions, output dir, API settings.

### `configs/TRAINING_NOTES.md`
Qwen7B case study with profiling, LR scans, memory breakdown, grad checkpoint findings.

---

## Documentation (`docs/`)

### `design.md` (435 lines)
**Master architecture document.** Read this first to understand the full system.
- §0: Data-gen interface (parquet columns, sidecar schema, resolution precedence)
- §1: Miles extension points (rollout, custom-rm, loss, data-source)
- §2: Required upstream patches (`--custom-actor-cls-path`, `--force-use-critic`)
- §3: Data transport via `multimodal_train_inputs`
- §4: Three training modes (SFT actor, SFT critic, RL)
- §5: `NLAFSDPActor` deep dive (hooks, state management, cp_size constraint)
- §6: Package layout
- §7: Build order

### `setup.md` (142 lines)
Installation guide — inference-only vs full training stack, Miles/SGLang/Megatron installation, patch application.

### `inference.md` (442 lines)
Complete inference recipe:
- Sidecar contract
- 5-step inference recipe (tokenize → embed → scale → inject → send)
- Model-specific params (`injection_scale` 150 vs 80000, `embed_scale`, BOS handling)
- SGLang launch + throughput notes
- Gemma-3 multimodal wrapper workaround
- Debugging: injection failure signatures (CJK output = silent injection failure)
- AR scoring (reconstruction MSE + cosine similarity)

---

## Patches (`patches/`)

### `apply_sglang_patches.sh`
Applies source-level fixes to SGLang:
- bf16-base64 embedding transport
- Chunked-prefill correctness
- Gemma-3 multimodal `input_embeds` routing
- Retract-path KV cache fix

### `nla/miles_patches/`
Integration patches for Miles upstream:
- `UPSTREAM_PIN` — exact commit hash to apply patches against
- Adds `--custom-actor-cls-path`, `--force-use-critic`, NLA arg group

---

## Tools (`tools/`)
Checkpoint conversion utilities:
- FSDP-DCP → HF format
- Megatron distributed checkpoint → HF format

---

## Release (`release/`)
- `hf_stage_and_scrub.py` — checkpoint publication utility (sanitizes sidecars, stages to HF Hub)
- `model_cards/` — HF model card templates per base model

---

## Examples (`examples/`)
Worked inference transcripts for Qwen, Gemma, and Llama checkpoints showing real activation → explanation outputs.

## Scripts (`scripts/datagen/`)
- `stage0_multigpu.sh` — multi-GPU wrapper for stage 0 (data-parallel sharding across GPUs)

---

## Key Constants & Sentinel Values

| Name | Value | Location | Purpose |
|---|---|---|---|
| `EXPLANATION_OPEN/CLOSE` | `<explanation>` / `</explanation>` | `schema.py` | Response wrapping |
| `INJECT_PLACEHOLDER` | `<INJECT>` | `schema.py` | Prompt placeholder in parquets |
| `ACTIVATION_COLUMN` | `"activation_vector"` | `schema.py` | Parquet column name |
| `SCALE_SQRT_D` | `"sqrt_d_model"` | `schema.py` | Sidecar scale sentinel |
| `MM_ACTIVATION_KEY` | `"nla_activation"` | `train_actor.py` | Multimodal dict key (actor) |
| `MM_CRITIC_TOKENS_KEY` | `"nla_critic_tokens"` | `train_actor.py` | Multimodal dict key (critic, RL) |
| `MM_MSE_SCALE_KEY` | `"nla_mse_scale"` | `train_actor.py` | Multimodal dict key (loss) |
| `FAILED_EXTRACTION_REWARD` | `-2.0` | `reward.py` | Penalty for failed explanation parsing |
| `_MIN_POSITION` | `50` | `stage0_extract.py` | Skip early noisy positions |

---

## Data Flow

### SFT Training (Actor)
```
Parquet (av_sft.parquet)
  → NLADataSource            substitute <INJECT>→char, load activation_vector
  → sft_actor.generate_rollout  append response, compute loss_mask
  → NLAFSDPActor._get_model_inputs_args   normalize activation → _nla_vectors
  → Forward hook             scan input_ids, inject at marked position
  → Model forward            logits over response tokens
  → SFT loss                 masked cross-entropy
```

### RL Training (Actor + Critic simultaneous)
```
Parquet (rl.parquet) → NLADataSource
  → nla_generate             build embeds, inject, send to SGLang via input_embeds
  → AV output                <explanation>...</explanation>
  → extract_explanation      parse tags
  → multimodal_train_inputs  {activation, critic_tokens, mse_scale}
  → Actor branch             normalize, inject, compute log_probs, GRPO loss
  → Critic branch            swap to critic_tokens, forward, extract at tokens[-1], MSE loss
  → Reward (async)           batch critic_fwd on live weights via Ray → -mse_nrm
```

---

## Critical Invariants (never break)

1. **No `cp_size > 1`** — context-parallel splits samples across ranks, breaking the neighbor check. Asserted in `train_actor.init()`.
2. **Neighbor check mandatory** — injection char is rare but not unique; always validate left/right neighbors.
3. **`injection_scale` ≠ `mse_scale`** — independent hyperparameters. Injection is OOD magnitude control; MSE scale is loss numerical stability.
4. **Radix cache disabled** — SGLang must use `--disable-radix-cache`. `input_embeds` have no token IDs → silent cache collision.
5. **No double-BOS** — chat templates already include BOS (Gemma/Llama); use `add_special_tokens=False`.
6. **Gemma embed scale** — raw `nn.Embedding` output is √d too small; must multiply explicitly.
7. **Data-gen never normalizes** — all parquets store raw vectors (`norm="none"`). `stage3_build` asserts this. Normalization happens at injection time and loss time.
8. **Stage-1 split is document-level** — all rows from one doc go to the same bucket. Never split positions from one doc across `av_sft`/`ar_sft`/`rl`.
9. **Per-doc keyed RNG** — same `(seed, doc_id)` → same sampled positions across multi-GPU sharding.
10. **Injection hook scans inside the hook** — Miles reorders samples twice before forward; any precomputed injection index is wrong by construction.

---

## Inter-Module Dependencies

```
schema.py  (no ML imports — pure schema)
  ├── config.py
  ├── data_source.py
  ├── loss.py
  └── datagen/sidecar.py

arch_adapters.py  (pure arch dispatch, no NLA-specific logic)
  ├── models.py
  ├── train_actor.py
  └── rollout/nla_generate.py

models.py  (NLACriticModel)
  ├── train_actor.py
  └── nla_inference.py (reimplements load_embedding_only standalone)

injection.py  (pure, unit-testable)
  ├── train_actor.py  (register as forward hook)
  └── rollout/nla_generate.py  (inline pre-SGLang)

storage.py  (abstract I/O backend)
  ├── data_source.py
  └── datagen/* (all write via storage)

train_actor.py  (main training actor — everything converges here)
  uses: config.py, models.py, loss.py, injection.py, arch_adapters.py

reward.py  (async, runs on separate Ray actor)
  uses: config.py, schema.py

rollout/*.py  (generation + rollout helpers)
  nla_generate.py uses: arch_adapters.py, injection.py, config.py, schema.py
```
