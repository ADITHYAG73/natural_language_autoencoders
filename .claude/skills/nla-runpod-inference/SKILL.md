---
name: nla-runpod-inference
description: >
  End-to-end recipe for running NLA (Natural Language Autoencoder) inference on a
  RunPod GPU pod — AV (vector→text) and AR (text→vector). Captures every blocker we
  hit and beat: Blackwell/CUDA pinning, sglang version vs torch, transformers v4-vs-v5,
  flashinfer-on-sm120, hf-xet/hf_transfer download failures, and the small-disk model
  juggle. Covers BOTH Qwen2.5 (transformers<5) AND Gemma-3 (transformers 5.3.0 +
  return_dict patch + triton-not-fa3 + gated token) — see the Gemma-3 section.
  Use this whenever spinning up a pod to run kitft/nla-* checkpoints.
---

# NLA inference on RunPod — battle-tested recipe

This is the distilled result of a full debugging session (2026-05-31) getting
`kitft/nla-qwen2.5-7b-L20-av` + `-ar` to run on RunPod. Follow the "Fast path"
top to bottom. The "Why / troubleshooting" appendix explains every gotcha with the
exact error so you can recognize it instantly.

The single biggest lesson: **deploy with enough disk up front.** The whole session's
pain was a 30 GB container disk + 10 GB network volume vs ~40 GB of models.

---

## 0. Deploy the pod (get this right and the rest is smooth)

- **GPU:** anything ≥24 GB works for Qwen-7B inference. We used **RTX PRO 6000
  Blackwell (sm120, 96 GB)** — note it's a *new* arch with CUDA quirks (see appendix).
  An A100/H100 (Ampere/Hopper) avoids the Blackwell flashinfer/cuda issues entirely.
- **Container disk: set to 60–100 GB** (default 30 GB is too small). This is the fix
  for 90% of the pain. Models: base Qwen ~15 GB + AV ~15 GB + AR ~10 GB = ~40 GB.
- **Network volume: if you attach one, make it ≥100 GB.** The default tiny volume
  (we got 10 GB) hits `Disk quota exceeded (os error 122)` — a *hard quota*, not the
  shared backing store's free space (`df` lies and shows the 400 TB MFS pool).
- Template: a PyTorch image (we used `runpod-torch-v280`, shipped torch 2.8.0+cu128).
- **SSH key:** add your **passphrase-less** public key to RunPod → Settings → SSH
  Public Keys *before* creating the pod (keys are injected at boot). A passphrase
  key breaks non-interactive `ssh -o BatchMode=yes` with a misleading
  `Permission denied (publickey)`. Generate one:
  `ssh-keygen -t ed25519 -C you@example.com -f ~/.ssh/id_ed25519_runpod_new` (Enter
  twice for empty passphrase). Connect via **direct TCP** (`ssh root@<ip> -p <port>`),
  not just the proxy.

---

## 1. Keep `/` safe — redirect all caches (run first, every session)

```bash
export HF_HOME=/workspace/hf          # or wherever the BIG disk is
export HF_HUB_DISABLE_XET=1           # xet downloader fails on network FS + freezes
export HF_HUB_ENABLE_HF_TRANSFER=0    # base image sets =1 but hf_transfer isn't installed
export TORCH_CUDA_ARCH_LIST="12.0+PTX"  # Blackwell only; harmless elsewhere
```
Put models on the disk that actually has space. If the network volume is tiny, use
`HF_HOME=/root/hf` (the container disk) instead — whichever is bigger.

---

## 2. Install deps — DO NOT let sglang upgrade torch to a cu130 build

The base image already had a working **torch 2.8.0+cu128**. sglang needs newer torch
but its default pull is cu130 (CUDA 13) → breaks on a CUDA-12 driver. Pin the CUDA
build explicitly.

```bash
# small deps (PEP 668 → --break-system-packages; --no-cache-dir keeps / clean)
pip install --break-system-packages --no-cache-dir \
    transformers safetensors httpx orjson pyyaml numpy pyarrow huggingface_hub

# torch 2.9.1 built for cu128 (matches a CUDA 12.8 driver + Blackwell). cu130 = broken.
pip install --break-system-packages --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# newest sglang that still rides torch 2.9.1 (0.5.11+ demand torch 2.11)
pip install --break-system-packages --no-cache-dir "sglang[all]==0.5.10.post1"

# CRITICAL (QWEN ONLY): sglang pulls transformers 5.x, but nla_inference.py needs 4.x
# (transformers 5.x apply_chat_template(tokenize=True) returns a BatchEncoding,
#  not list[int] → injection marker scan finds 0 matches → AssertionError).
# sglang 0.5.10 STILL WORKS on 4.57.6 for plain Qwen2, so just downgrade:
pip install --break-system-packages --no-cache-dir "transformers<5"
```

> ⚠️ **GEMMA-3 IS THE OPPOSITE — DO NOT DOWNGRADE.** Gemma-3 in sglang
> 0.5.10.post1 needs `config.rope_parameters`, which only exists in
> **transformers 5.x** (4.x has `rope_scaling`). sglang 0.5.10.post1 in fact
> *hard-pins* `transformers==5.3.0`. So for Gemma you **keep 5.3.0** and instead
> patch `nla_inference.py` (one-liner, below). Qwen wants 4.x, Gemma wants 5.3.0
> — they are mutually exclusive in one env. See the **Gemma-3 section** for the
> full recipe; if you're doing Gemma, skip the `transformers<5` line entirely.

Verify torch survived and sees the GPU:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.9.1+cu128 True
```

**sglang↔torch version map** (from the resolver, useful to know):
`0.5.6–0.5.10.post1 → torch==2.9.1` · `0.5.11+ → torch==2.11.0`.

---

## 3. Launch the AV server (Blackwell needs the triton backend)

```bash
python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
  --port 30000 --disable-radix-cache --mem-fraction-static 0.8 --trust-remote-code \
  --attention-backend triton --sampling-backend pytorch
```
- `--attention-backend triton --sampling-backend pytorch` — **mandatory on sm120
  Blackwell.** Default flashinfer crashes with `FlashInfer requires GPUs with sm75 or
  higher` (misleading — it's the cu128 capability-probe failing, not your GPU).
  Confirmed maintainer workaround: sgl-project/sglang #24633, #15342.
- `--disable-radix-cache` — **mandatory for NLA always** (radix keys on token IDs;
  NLA injects raw activations → different activations would cache-collide).
- On Ampere/Hopper you can drop the triton flags (flashinfer works there).
- Ignore the repeated `Failed to get device capability: SM 12.x requires CUDA >= 12.9`
  — non-fatal warning; it falls back and runs.
- Success line: **`The server is fired up and ready to roll!`**

Detach tip: tmux sessions died on SSH disconnect on our pod. Either keep the terminal
open, or run inside the web terminal, or use a long-lived ssh command. If you must
detach, test that it survives before relying on it.

---

## 4. Level 1 — AV smoke test (the true hello world, AV only)

`nla_inference.py` reads `nla_meta.yaml` from a **local dir**, so resolve the cached
snapshot path (a bare hub id fails). Run as **two** commands (don't inline the
`VAR=$(...) python` form — the prefix-assignment expands `$VAR` empty):

```bash
AV_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")
python nla_inference.py "$AV_DIR" --sglang-url http://localhost:30000   # no parquet => random vector
```
English output (even if semantically random) = injection works. **All-Chinese/CJK =
injection failed** (the marker char `㈎` is CJK; failure verbalizes the marker itself).

---

## 5. Level 2 — real activations → meaningful descriptions

Extract a real layer-20 activation from base Qwen, feed it to the AV.
`extraction script` (no `accelerate` needed; `dtype=` not `torch_dtype=`):

```python
import torch, pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL, LAYER = "Qwen/Qwen2.5-7B-Instruct", 20   # hidden_states[LAYER+1] = output of block 20
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
text = "<a passage of 60+ tokens so positions are >=50; earlier tokens decode to noise>"
ids = tok(text, return_tensors="pt").to("cuda"); n = ids["input_ids"].shape[1]
with torch.no_grad():
    hs = m(**ids, output_hidden_states=True).hidden_states[LAYER+1][0]   # [seq, 3584]
rows = {"activation_vector": [], "doc_id": [], "position": []}
for p in [n-1, n-15, 55]:
    if 50 <= p < n:
        rows["activation_vector"].append(hs[p].float().cpu().numpy().tolist())
        rows["doc_id"].append("demo"); rows["position"].append(p)
pq.write_table(pa.table(rows), "/workspace/demo_activations.parquet")
```
Then (AV server running):
```bash
AV_DIR=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('kitft/nla-qwen2.5-7b-L20-av'))")
python nla_inference.py "$AV_DIR" --parquet /workspace/demo_activations.parquet --n 3
```
Real activations give topical, accurate descriptions — and at predictive tokens the AV
reads the model's *plan* (e.g. at ` King` it says "expecting Louis XVI"; at `topple` it
says "expecting the monarchy"). That's the paper's "planning" finding, reproduced.

### Small-disk juggle (only if disk < ~45 GB)
Can't hold base Qwen + AV at once on a 30 GB disk. Sequence:
1. Stop AV server (frees VRAM). 2. `rm -rf $HF_HOME/hub/models--kitft--*av*` /
   actually delete the model dir to free ~15 GB. 3. Download base Qwen, run extraction
   → tiny parquet on the volume. 4. Delete base Qwen. 5. Relaunch AV server
   (re-downloads AV). 6. Run the `--parquet` command. **Fix it properly by deploying
   with a bigger disk so you never juggle.**

---

## 6. AR scoring (the round-trip MSE / cosine)

The AR (`kitft/nla-qwen2.5-7b-L20-ar`, ~10 GB) is the reconstructor. In Python:
```python
from nla_inference import NLACritic
critic = NLACritic("<local AR dir>", device="cuda")   # NOTE: positional checkpoint_dir
mse, cos = critic.score(explanation_text, original_activation_np)   # returns a TUPLE
# cos≈0.9 → MSE≈0.2 good · cos=0 → MSE=2 orthogonal
```
Needs the AR model on disk too — so this is where the bigger disk really pays off.

---

## Gemma-3 (e.g. gemma3-12b-L32) — full recipe, verified 2026-06-07

Gemma differs from Qwen in **five** ways that each cost time the first run. The
working scripts are committed in the repo: `tutorials/launch_av.sh`,
`tutorials/round_trip.py`, `tutorials/steer.py`. Verified end-to-end on RTX PRO
6000 Blackwell (sm120): reproduced the planning finding (AV reads "expecting
'XVI'" at ` Louis`) with AR reconstruction **cos≈0.997**.

**Proven Gemma stack:** torch **2.9.1+cu128**, sglang **0.5.10.post1**,
**transformers 5.3.0** (NOT <5 — opposite of Qwen), triton attention backend,
CUDA 13.0 driver. Repos: `kitft/nla-gemma3-12b-L32-av` + `-ar`, base
`google/gemma-3-12b-it`.

### 1. transformers 5.3.0 + a 1-line patch (the headline gotcha)
- sglang 0.5.10.post1 **hard-pins `transformers==5.3.0`**; Gemma-3 needs its
  `rope_parameters` attr. Symptom if you downgraded: `'Gemma3TextConfig' object
  has no attribute 'rope_parameters'` at model load.
- But on 5.x, `tokenizer.apply_chat_template(tokenize=True)` returns a
  `BatchEncoding`, not `list[int]` → `nla_inference.py`'s marker scan finds 0
  matches → `injection token appears 0× ` assert. **Fix:** add
  `return_dict=False` to the two `apply_chat_template(..., tokenize=True, ...)`
  calls in `nla_inference.py` (lines ~189, ~405). Already committed to the fork;
  if on a fresh clone: `sed -i 's/tokenize=True, add_generation_prompt=True,/&  return_dict=False,/g' nla_inference.py` (then sanity-check it hit exactly 2 lines).

### 2. Attention backend: triton, NOT fa3 (on Blackwell)
- `docs/setup.md` says Gemma needs `--attention-backend fa3`. **fa3 only
  supports SM 80–90** (Ampere/Hopper) → on Blackwell sm120 it asserts
  `FlashAttention v3 Backend requires SM>=80 and SM<=90`.
- Use `--attention-backend triton --sampling-backend pytorch`. Correct for NLA
  because prompts are short — Gemma's sliding-window == full attention when the
  window covers the whole sequence (sglang also auto-disables hybrid-SWA).
- On a real Hopper/Ampere card, use `fa3` as the docs say.

### 3. Gated base model — token required
- `google/gemma-3-12b-it` is gated. Accept the license at its HF page, make a
  **Read** token, then on the pod (with `HF_HOME` set so it caches to the big
  disk): `export HF_HOME=/workspace/hf && hf auth login` (paste token, "add as
  git credential? n"). Caches to `/workspace/hf/token`; all later calls pick it
  up. The `kitft/*` AV/AR repos are public (no token needed).

### 4. VRAM: shrink the server so base+AR fit alongside
- Default `--mem-fraction-static 0.8` grabbed ~78 GB (mostly KV cache you don't
  need for tiny prompts), leaving too little for the 24 GB base model. Use
  **`--mem-fraction-static 0.35`** (~34 GB) — leaves ~60 GB for base (24) + AR
  (16) loaded in-process for extraction/scoring. (97 GB card.)

### 5. Sidecar values + misc
- Gemma sidecar: `d_model=3840`, `inj_scale=80000.0`, **`embed_scale=61.97`
  (=√d, Gemma multiplies embeddings by √d — Qwen is 1.0)**, `inj_char='㈜'`
  (id 246566). All auto-loaded; never hardcode.
- `model.norm.weight | MISSING → newly initialized` when loading the AR is
  **expected** — the critic replaces the final LayerNorm with Identity
  (`nla/models.py`), so that weight is intentionally absent.
- Multimodal nesting: base gemma-3-it decoder layers are at
  `m.model.language_model.layers`, NOT `m.model.layers` (matters for steering
  hooks; `tutorials/steer.py` resolves this).

### 6. Running the three tutorials (server up first)
```bash
bash tutorials/launch_av.sh                       # AV server (triton, mem 0.35)
# second terminal, env exported + hf auth login done:
python tutorials/round_trip.py "your 60+ token passage"   # AV→AR→(mse,cos)
python tutorials/steer.py --concept "cheese and dairy"     # AR-vector surgery
```

### Network flakiness (this pod had intermittent egress)
- TLS/read timeouts mid-download: re-run, `snapshot_download` resumes. Bump
  `export HF_HUB_DOWNLOAD_TIMEOUT=60`. For big repos, pre-fetch with
  `hf download <repo>` (resumes per-file) until clean, *then* run the script.

### Driving the pod from a laptop / agent
- The `ssh.runpod.io` **proxy only allows interactive sessions** (one-shot
  `ssh ... "cmd"` → "doesn't support PTY"). For non-interactive/automation use
  **direct TCP**: `ssh root@<ip> -p <port> -i <key>`. Backgrounding a process
  over a one-shot SSH (`&`) tears the channel down (exit 255) — instead write a
  launcher script and run the long-lived server in an interactive terminal.

---

## Key facts (don't rederive these)

| Thing | Value | Note |
|---|---|---|
| Qwen2.5-7B `d_model` | 3584 | layer 20, `hidden_states[21]` |
| `injection_scale` (Qwen) | 150 | from sidecar; mandatory, don't hardcode elsewhere |
| `embed_scale` (Qwen) | 1.0 | Gemma is √d ≈ 62 — different! |
| injection marker | `㈎` U+320E, id 149705 | CJK → failure smells like Chinese output |
| `_MIN_POSITION` | 50 | earlier tokens decode to noise |
| healthy Qwen L20 norm | ~100–170 | wildly off → bad/OOD activation |
| `NLACritic.score()` | `(mse, cos)` tuple | NOT a dict |
| `nla_inference.py` checkpoint arg | **local dir** | use snapshot_download, not hub id |

---

## Troubleshooting appendix (error → cause → fix)

- `ResolutionImpossible ... sglang depends on torch==2.9.1` → pin sglang to
  `==0.5.10.post1` (rides 2.9.1) instead of `>=0.5.6` which resolves to 0.5.12/torch 2.11.
- `torch ... cuda_ok False` + `NVIDIA driver ... too old (found 12080)` → you got a
  cu130 torch on a CUDA-12.8 driver. Reinstall `torch==2.9.1 --index-url .../cu128`.
- `FlashInfer requires GPUs with sm75 or higher` (on Blackwell) → add
  `--attention-backend triton --sampling-backend pytorch`.
- `Disk quota exceeded (os error 122)` even with `df` showing TB free → hard quota on
  the network volume. Use the container disk (`HF_HOME=/root/hf`) or grow the volume.
- `Fast download using 'hf_transfer' is enabled but ... not available` →
  `export HF_HUB_ENABLE_HF_TRANSFER=0`.
- xet download freezes / `os error 122` from `xet_get` → `export HF_HUB_DISABLE_XET=1`.
- `injection token appears 0× in canonical prompt` → transformers 5.x. Downgrade
  `pip install "transformers<5"`. (Long-term: a venv for the client if you must keep
  sglang on transformers 5.x, but 4.57.6 ran sglang 0.5.10 fine for us.)
- `requires accelerate` on `from_pretrained(device_map=...)` → drop `device_map`, use
  `.to("cuda")`; or `pip install accelerate`.
- `Unrecognized model in ..` → empty checkpoint path. You inlined `VAR=$(...) python`;
  run the assignment on its own line first.
- `externally-managed-environment` (PEP 668) → add `--break-system-packages`.
- `'Gemma3TextConfig' object has no attribute 'rope_parameters'` (Gemma load) →
  transformers too old. `pip install "transformers==5.3.0"` (what sglang
  0.5.10.post1 pins). Do NOT downgrade to <5 for Gemma.
- `injection token appears 0×` on Gemma with transformers 5.x → add
  `return_dict=False` to the two `apply_chat_template(tokenize=True,...)` calls
  in `nla_inference.py` (5.x returns BatchEncoding, not list[int]).
- `FlashAttention v3 Backend requires SM>=80 and SM<=90` (Blackwell + `--attention-backend fa3`)
  → use `--attention-backend triton --sampling-backend pytorch` instead.
- `'Gemma3Model' object has no attribute 'layers'` (forward hook on base gemma-it)
  → multimodal nesting; layers are at `model.language_model.layers`.
- `401/403` or "gated" on `google/gemma-3-12b-it` → accept the license on HF +
  `hf auth login` with `HF_HOME` exported first.
- `Your SSH client doesn't support PTY` / exit 255 on one-shot `ssh ... "cmd"`
  via `ssh.runpod.io` → proxy is interactive-only; use direct TCP `ssh root@<ip> -p <port>`.

---

## What we used (2026-05-31 session, for reference)
- Pod: RTX PRO 6000 Blackwell (sm120, 96 GB VRAM), 221 GB RAM, 28 vCPU, CA-MTL-3,
  $2.09/hr. Template `runpod-torch-v280`. Container disk 30 GB (too small),
  network volume 10 GB (too small — caused the quota fight).
- Final working stack: torch 2.9.1+cu128, sglang 0.5.10.post1 (triton backend),
  transformers 4.57.6, on a CUDA 12.8 driver.
- Verified against: HF env-vars docs, sglang install docs (CUDA-12 path),
  sgl-project/sglang issues #24633 #15342 (Blackwell triton), huggingface_hub #3266 /
  xet-core #483 (xet disable).

## What we used (2026-06-07 session — Gemma-3, for reference)
- Pod: RTX PRO 6000 Blackwell (sm120, 97 GB VRAM), 188 GB RAM, 16 vCPU, **100 GB
  container disk + 100 GB network volume** (the bigger disk made it painless —
  no juggling, base+AV+AR all resident). Driver 580.159.03 / CUDA 13.0.
- Final working stack: torch 2.9.1+cu128, sglang 0.5.10.post1 (**triton**
  backend), **transformers 5.3.0** (+ `return_dict=False` patch to
  nla_inference.py), CUDA 13.0 driver.
- Ran AV smoke (Level 1) → full AV→AR round-trip (`tutorials/round_trip.py`,
  cos≈0.997, reproduced planning) → AR-vector steering (`tutorials/steer.py`).
- Verified against: `docs/setup.md` (gated models §, Gemma fa3 note),
  sglang attention_registry (fa3 SM 80–90 assert), transformers 5.3.0
  apply_chat_template return-type change.
