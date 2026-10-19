"""Token sampling. scores() builds full-vocab scores, pick() draws a token; both stay lazy."""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class SamplingParams:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 0.0
    max_tokens: int = 2048
    seed: int | None = None


class Sampler:
    def __init__(self, params: SamplingParams, vocab_size: int):
        self.p = params
        # presence mask is updated lazily inside the graph: no host sync per token
        self.seen = mx.zeros((vocab_size,), mx.float32) if params.presence_penalty else None
        if params.seed is not None:
            mx.random.seed(params.seed)

    @property
    def greedy(self) -> bool:
        return self.p.temperature <= 0

    def scores(self, logits: mx.array, extra_seen: mx.array | None = None) -> mx.array:
        """Full-vocab scores: penalized logits (greedy) or filtered log-probs (sampling)."""
        p = self.p
        logits = logits.astype(mx.float32)
        if self.seen is not None:
            seen = self.seen
            if extra_seen is not None:
                seen = mx.minimum(seen.at[extra_seen].add(1.0), 1.0)
            logits = logits - p.presence_penalty * seen
        if self.greedy:
            return logits
        logits = logits / p.temperature
        V = logits.shape[-1]
        # candidate set: top-k values (or the whole vocab), sorted descending
        if 0 < p.top_k < V:
            cand = logits[mx.argpartition(-logits, kth=p.top_k - 1)[: p.top_k]]
        else:
            cand = logits
        cand = mx.sort(cand)[::-1]
        cutoff = cand[-1]
        if 0 < p.top_p < 1:
            probs = mx.softmax(cand)
            keep = (mx.cumsum(probs) - probs) < p.top_p
            cutoff = mx.min(mx.where(keep, cand, mx.inf))
        logits = mx.where(logits >= cutoff, logits, -mx.inf)
        return logits - mx.logsumexp(logits)

    def pick(self, scores: mx.array) -> mx.array:
        return mx.argmax(scores) if self.greedy else mx.random.categorical(scores)

    def observe(self, tok):
        if self.seen is not None:
            self.seen = mx.minimum(self.seen.at[tok].add(1.0), 1.0)
