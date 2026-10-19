"""Decoding loop with optional speculative steps from the model's own MTP head.

An MTP cycle drafts one token with the MTP head, then verifies [last, draft] in a
single 2-token pass of the main model, emitting 1 or 2 tokens. Acceptance follows
standard speculative sampling (greedy draft => accept with probability p(draft),
else sample from the residual), so the output distribution is the main model's.

The loop can switch between plain and MTP steps at every iteration (`use_mtp`):
on this hardware a 2-token pass runs at higher GPU clocks, so MTP buys speed at
roughly +15% energy per token. Plain steps keep a list of (hidden, token) pairs
the MTP head has not seen yet and hand them over in one batch when MTP resumes,
so its KV cache stays complete without paying for it on every plain token.
"""

from typing import Callable

import mlx.core as mx

from .model import Qwen35
from .mtp import MTPHead
from .sampler import Sampler

MAX_PENDING = 1024  # beyond this, the MTP head just misses the oldest positions


def mtp_prefill(mtp: MTPHead, mtp_cache, hidden, ids, prev_hidden):
    """Feed the MTP head the (h_i, tok_{i+1}) pairs of a prefill chunk.

    hidden: [1, L, H] main hidden states for this chunk's tokens
    ids:    [1, L]    this chunk's tokens
    prev_hidden: [1, 1, H] hidden of the token just before the chunk, or None
    Returns the hidden of the chunk's last token (the next chunk's prev_hidden).
    """
    if prev_hidden is not None:
        h_in = mx.concatenate([prev_hidden, hidden[:, :-1]], axis=1)
        next_ids = ids
    else:
        h_in, next_ids = hidden[:, :-1], ids[:, 1:]
    if next_ids.shape[1] > 0:
        mx.eval(mtp(h_in, next_ids, mtp_cache), mtp_cache.state())
    return hidden[:, -1:]


def decode(model: Qwen35, sampler: Sampler, cache, first_logits, last_hidden=None,
           mtp: MTPHead | None = None, mtp_cache=None,
           use_mtp: Callable[[], bool] = lambda: True, stats: dict | None = None,
           forced: list[int] | None = None):
    """Generator of token ids. stats gets 'cycles' / 'accepted' (MTP) and 'plain' counters.

    forced: tokens the caller can append at any time; they are emitted next, in order, instead of
    sampled ones (plain steps only). Used to close the reasoning when its budget runs out."""
    forced = forced if forced is not None else []
    stats = stats if stats is not None else {}
    for k in ("cycles", "accepted", "plain"):
        stats.setdefault(k, 0)

    feed = sampler.pick(sampler.scores(first_logits))
    mx.eval(feed)
    sampler.observe(feed)
    # (hidden, token) pairs not yet given to the MTP head
    pend_h = [last_hidden] if mtp is not None else []
    pend_ids = [feed.reshape(1, 1)] if mtp is not None else []
    yield feed.item()

    vocab = mx.arange(model.cfg.vocab_size)
    while True:
        if mtp is not None and pend_h and not forced and use_mtp():
            h_in = pend_h[0] if len(pend_h) == 1 else mx.concatenate(pend_h, axis=1)
            id_in = pend_ids[0] if len(pend_ids) == 1 else mx.concatenate(pend_ids, axis=1)
            draft = mx.argmax(mtp(h_in, id_in, mtp_cache)[0, -1])
            ids = mx.stack([feed.reshape(()), draft]).reshape(1, 2)
            logits, hidden = model(ids, cache, last_only=False, return_hidden=True, keep_prev=True)

            s0 = sampler.scores(logits[0, 0])
            if sampler.greedy:
                resample = mx.argmax(s0)
                accept = resample == draft
            else:
                accept = mx.random.uniform() < mx.exp(s0[draft])
                resample = mx.random.categorical(mx.where(vocab == draft, -mx.inf, s0))
            t1 = sampler.pick(sampler.scores(logits[0, 1], extra_seen=draft))
            mx.eval(accept, resample, t1, draft)
            stats["cycles"] += 1

            if accept.item():
                stats["accepted"] += 1
                sampler.observe(draft)
                sampler.observe(t1)
                for c in cache:
                    if hasattr(c, "ssm_prev"):
                        c.conv_prev = c.ssm_prev = None
                pend_h = [hidden]
                pend_ids = [mx.concatenate([draft.reshape(1, 1), t1.reshape(1, 1)], axis=1)]
                feed = t1
                yield draft.item()
                yield t1.item()
            else:
                for c in cache:
                    c.rollback(1)
                sampler.observe(resample)
                pend_h, pend_ids, feed = [hidden[:, :1]], [resample.reshape(1, 1)], resample
                yield resample.item()
        else:
            if mtp is not None:
                logits, hidden = model(feed.reshape(1, 1), cache, return_hidden=True)
            else:
                logits, hidden = model(feed.reshape(1, 1), cache), None
            t = mx.array(forced.pop(0)) if forced else sampler.pick(sampler.scores(logits[0, -1]))
            mx.async_eval(t)
            sampler.observe(t)
            stats["plain"] += 1
            if mtp is not None:
                pend_h.append(hidden)
                pend_ids.append(t.reshape(1, 1))
                if len(pend_h) > MAX_PENDING:
                    pend_h, pend_ids = pend_h[-MAX_PENDING:], pend_ids[-MAX_PENDING:]
            feed = t
            yield t.item()
