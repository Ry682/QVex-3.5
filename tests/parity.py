"""Compare our engine against mlx_lm on the same weights.

    python tests/parity.py ref     # run mlx_lm, save logits + greedy tokens
    python tests/parity.py ours    # run engine, compare with saved reference

The two runs are separate processes so only one model copy sits in memory.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
REF = ROOT / "results" / "parity_ref.npz"
PROMPT = ("<|im_start|>user\nSpiega in tre frasi perché un MacBook Air senza ventola "
          "rallenta quando si scalda.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
N_GEN = 64


def greedy(forward, prompt_ids, n):
    """forward(ids[1, L]) -> logits[1, L, V] using an internal cache."""
    logits = forward(mx.array([prompt_ids]))
    first = logits[0].astype(mx.float32)
    tok = mx.argmax(logits[0, -1]).item()
    out = [tok]
    t0 = time.perf_counter()
    for _ in range(n - 1):
        logits = forward(mx.array([[tok]]))
        tok = mx.argmax(logits[0, -1]).item()
        out.append(tok)
    dt = time.perf_counter() - t0
    return first, out, (n - 1) / dt


def run_ref():
    from mlx_lm import load
    model, tok = load(str(MODEL))
    ids = tok.encode(PROMPT, add_special_tokens=False)
    cache = model.make_cache()
    first, out, tps = greedy(lambda x: model(x, cache=cache), ids, N_GEN)
    REF.parent.mkdir(exist_ok=True)
    np.savez(REF, ids=np.array(ids), logits=np.array(first), tokens=np.array(out))
    print(f"ref: {tps:.1f} tok/s\n{tok.decode(out)!r}")


def run_ours():
    from engine.model import load_model
    from tokenizers import Tokenizer
    ref = np.load(REF)
    ids = ref["ids"].tolist()
    t0 = time.perf_counter()
    model = load_model(MODEL)
    print(f"load {time.perf_counter() - t0:.1f}s, active mem {mx.get_active_memory() / 1e9:.2f} GB")
    cache = model.make_cache()
    first, out, tps = greedy(lambda x: model(x, cache, last_only=False), ids, N_GEN)

    a, b = np.array(first), ref["logits"]
    diff = np.abs(a - b)
    agree_top1 = (a.argmax(-1) == b.argmax(-1)).mean()
    same = 0
    for x, y in zip(out, ref["tokens"].tolist()):
        if x != y:
            break
        same += 1
    print(f"prefill logits: max|diff| {diff.max():.4f}  mean|diff| {diff.mean():.5f}  "
          f"top-1 agreement {agree_top1 * 100:.1f}% over {len(ids)} positions")
    print(f"greedy: {same}/{N_GEN} identical tokens   ours {tps:.1f} tok/s")
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    print(repr(tok.decode(out)))


if __name__ == "__main__":
    {"ref": run_ref, "ours": run_ours}[sys.argv[1]]()
