"""Rollback correctness + numerics of 2-token vs 1-token passes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.model import load_model

MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
m = load_model(MODEL)
prompt = mx.array([[9707, 11, 1246, 525, 498, 30, 358, 1079, 264, 1273]])
A, B, C = 2585, 4009, 1879

def f32(x): return x.astype(mx.float32)

# reference: prompt, A, C one token at a time
c1 = m.make_cache(); m(prompt, c1)
la = m(mx.array([[A]]), c1); lc_ref = m(mx.array([[C]]), c1)

# speculative path: prompt, [A, B] with keep_prev, rollback, then C
c2 = m.make_cache(); m(prompt, c2)
l2 = m(mx.array([[A, B]]), c2, last_only=False, keep_prev=True)
for c in c2: c.rollback(1)
lc = m(mx.array([[C]]), c2)

# exact-state check: same 2-token pass vs. two 1-token passes on a 1-token batch
c3 = m.make_cache(); m(prompt, c3); m(mx.array([[A, B]]), c3, keep_prev=True)
for c in c3: c.rollback(1)
c4 = m.make_cache(); m(prompt, c4); m(mx.array([[A, B]]), c4)  # no rollback, offset +2

d_first = mx.abs(f32(l2[0, 0]) - f32(la[0, -1])).max().item()
d_after = mx.abs(f32(lc[0, -1]) - f32(lc_ref[0, -1])).max().item()
print(f"logits pos0 of 2-token pass vs 1-token pass: max|diff| {d_first:.4f}")
print(f"after rollback + C vs sequential A,C:        max|diff| {d_after:.4f}")
print("argmax match:", mx.argmax(lc[0, -1]).item() == mx.argmax(lc_ref[0, -1]).item())
offs = [(a.offset, b.offset) for a, b in zip(c3, c4)][:4]
print("offsets after rollback vs not:", offs)
