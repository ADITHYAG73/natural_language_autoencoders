"""Completion provider backends for Stage 2 (API explanation generation).

Stage 2 calls an external LLM to produce natural-language explanations of
source text — these become the `response` column for AV-SFT and the `prompt`
content for AR-SFT. `CompletionProvider` is the pluggable interface: stage 2
code hands it a batch of fully-formed prompts and gets back a batch of
completions. Concurrency, retries, rate limits, and auth are all the
provider's problem.

Swap via `--provider-cls my.module.MyProvider` at stage2 invocation.
"""

import asyncio
import time
from abc import ABC, abstractmethod

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back.

    Stage 2 formats NLA-specific instruction prompts; the provider just maps
    `prompts[i] -> completion[i]` (or None for prompts that exhausted retries).
    A robust sampling engine can be plugged in by wrapping it in a subclass.

    None returns are per-prompt gave-up signals — stage2 drops those rows
    (same path as failed-extract-pattern). This means a chunk can survive
    losing a few prompts to sustained 429/500 storms instead of discarding
    511 good completions because one failed. Gaps ARE tracked: stage2 logs
    a drop count, and the parquet row count tells you exactly how many
    survived.
    """

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]: ...


class AnthropicProvider(CompletionProvider):
    """Default provider: Anthropic Messages API with bounded async concurrency.

    The SDK handles transport-level retries (408/429/5xx, exponential backoff
    with jitter, respects Retry-After). High `max_retries` extends the retry
    window for sustained rate-limit storms — at max_retries=100 the SDK will
    keep backing off for minutes before giving up on one prompt.

    Per-prompt failures after exhausting retries return None (caller drops
    the row). `gather(return_exceptions=True)` collects these without nuking
    the whole batch — otherwise one stubborn 429 in a chunk of 512 wastes
    the other 511 API calls. ONLY `RateLimitError` and server-side 5xx are
    tolerated; anything else (auth, bad request, unexpected content) still
    raises — those are code bugs, not transient.

    Calls `asyncio.run()` — do not invoke from inside a running event loop.
    Stage 2 is a standalone CLI, so this is fine in practice.
    """

    # Exceptions from which we degrade to None instead of killing the batch.
    # Anything NOT in this tuple is a code bug and should still blow up loud.
    _TOLERATED = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    )

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        concurrency: int = 32,
        max_retries: int = 10,
    ):
        self.client = anthropic.AsyncAnthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        async with sem:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        # refusal: source text tripped safety — no answer coming, drop this row.
        # content may be [] or the refusal message; either way, no explanation.
        if resp.stop_reason == "refusal":
            return None
        assert resp.stop_reason in ("end_turn", "max_tokens"), (
            f"unexpected stop_reason={resp.stop_reason!r} (want end_turn/max_tokens/refusal)"
        )
        assert len(resp.content) == 1 and resp.content[0].type == "text", (
            f"expected single text block, got {[b.type for b in resp.content]}"
        )
        text = resp.content[0].text.strip()
        assert text, "empty completion — refusing to emit blank explanation"
        return text

    def complete(self, prompts: list[str]) -> list[str | None]:
        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_failed = 0
        n_refused = 0
        for i, r in enumerate(raw):
            if isinstance(r, str):
                out.append(r)
            elif r is None:
                n_refused += 1
                out.append(None)
            elif isinstance(r, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(r, BaseException):
                # Not a transient — auth/schema/code bug. Blow up loud.
                raise r
            else:
                raise AssertionError(f"gather returned unexpected type at [{i}]: {type(r).__name__}")
        if n_failed or n_refused:
            print(f"  [AnthropicProvider] dropped {n_refused} refused + {n_failed} retry-exhausted of {len(prompts)}")
        return out


def _message_to_text(msg: anthropic.types.Message) -> str | None:
    """Shared: Messages response -> explanation text (or None). Mirrors
    AnthropicProvider._one's extraction so live + batch behave identically:
    refusal -> None; otherwise require a single non-empty text block."""
    if msg.stop_reason == "refusal":
        return None
    assert msg.stop_reason in ("end_turn", "max_tokens"), (
        f"unexpected stop_reason={msg.stop_reason!r} (want end_turn/max_tokens/refusal)"
    )
    assert len(msg.content) == 1 and msg.content[0].type == "text", (
        f"expected single text block, got {[b.type for b in msg.content]}"
    )
    text = msg.content[0].text.strip()
    assert text, "empty completion — refusing to emit blank explanation"
    return text


class AnthropicBatchProvider(CompletionProvider):
    """Batches API provider — ~50% cheaper than live calls; for non-latency-sensitive datagen.

    Same complete() contract as AnthropicProvider (prompts[i] -> completion[i] or None),
    but submits the whole list as ONE Message Batch, polls until it ends, and maps results
    back by custom_id. Trade-off: ~50% cost saving vs higher latency (minutes, up to 24h) —
    fine for stage 2, which is not latency-sensitive.

    Use:  --provider-cls nla.datagen.providers.AnthropicBatchProvider

    NOTE on chunking: stage 2 calls complete() once per chunk (stage2.chunk_size, default 512),
    so each chunk = one batch. A batch holds up to 100k requests, so for batch mode set a LARGE
    stage2.chunk_size (e.g. 8192+) → fewer batches, fewer polls. (Trade-off: larger chunks lose
    stage2's per-chunk crash-recovery granularity.)

    Errored / expired / canceled results degrade to None (row dropped), same as the live
    provider's gave-up signal; a breakdown is logged so systematic failures are visible.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 300,
        temperature: float = 1.0,
        poll_interval: float = 30.0,
        max_retries: int = 10,
    ):
        self.client = anthropic.Anthropic(max_retries=max_retries)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.poll_interval = poll_interval

    def complete(self, prompts: list[str]) -> list[str | None]:
        if not prompts:
            return []
        requests = [
            Request(
                custom_id=f"r{i}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": p}],
                ),
            )
            for i, p in enumerate(prompts)
        ]
        batch = self.client.messages.batches.create(requests=requests)
        print(f"  [AnthropicBatchProvider] submitted batch {batch.id} "
              f"({len(prompts)} requests, 50%-off); polling every {self.poll_interval:.0f}s…")

        while True:
            b = self.client.messages.batches.retrieve(batch.id)
            if b.processing_status == "ended":
                break
            time.sleep(self.poll_interval)

        out: list[str | None] = [None] * len(prompts)
        n_ok = n_refused = n_errored = n_other = 0
        err_types: dict[str, int] = {}
        for result in self.client.messages.batches.results(batch.id):
            idx = int(result.custom_id[1:])  # strip the "r" prefix
            rtype = result.result.type
            if rtype == "succeeded":
                text = _message_to_text(result.result.message)
                if text is None:
                    n_refused += 1
                else:
                    out[idx] = text
                    n_ok += 1
            elif rtype == "errored":
                # Official Batches API error shape is result.result.error.error.type
                # (double-nested; verified against platform.claude.com batch-processing docs).
                # getattr fallback stays robust if a category ever surfaces a flatter shape.
                err = result.result.error
                et = getattr(getattr(err, "error", None), "type", None) or getattr(err, "type", "unknown")
                err_types[et] = err_types.get(et, 0) + 1
                n_errored += 1          # left as None → row dropped (errored is NOT billed)
            else:                        # canceled / expired
                n_other += 1            # left as None → row dropped
        if n_refused or n_errored or n_other:
            print(f"  [AnthropicBatchProvider] {n_ok} ok | dropped {n_refused} refused, "
                  f"{n_errored} errored {err_types or ''}, {n_other} canceled/expired of {len(prompts)}")
        return out
