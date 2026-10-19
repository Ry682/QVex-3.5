"""Teacher-forced check: is every greedy MTP token the main model's argmax (up to near-ties)?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.model import load_model
from engine.mtp import load_mtp
from engine.sampler import Sampler, SamplingParams
from engine.speculative import decode, mtp_prefill
from engine.tokenizer import ChatTokenizer

MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
model = load_model(MODEL)
mtp = load_mtp(model, Path(__file__).resolve().parent.parent / "models/qwen3.5-9b-mtp/mtp-4bit.safetensors")
tok = ChatTokenizer(MODEL)
for text in ["Racconta in 300 parole la storia di un faro abbandonato sulla costa ligure.",
             "Scrivi una classe Python LRUCache con get e put in O(1), con docstring e test."]:
    ids = tok.encode(tok.render([{"role": "user", "content": text}], True, False))
    cache, mc = model.make_cache(), mtp.make_cache()
    arr = mx.array([ids]); logits, h = model(arr, cache, return_hidden=True)
    lh = mtp_prefill(mtp, mc, h, arr, None)
    out = []
    for t in decode(model, Sampler(SamplingParams(temperature=0), model.cfg.vocab_size), cache,
                    logits[0, -1], lh, mtp, mc):
        out.append(t)
        if len(out) >= 256: break
    # one-token-at-a-time teacher forcing (the plain decode numerics)
    c = model.make_cache(); l = model(arr, c)
    mism, margins = 0, []
    for i, t in enumerate(out):
        lf = l[0, -1].astype(mx.float32)
        top2 = mx.topk(lf, 2)
        am = mx.argmax(lf).item()
        if am != t:
            mism += 1
            margins.append(round(abs(top2[1].item() - top2[0].item()) if False else (lf[am] - lf[t]).item(), 3))
        l = model(mx.array([[t]]), c)
    print(f"{len(out)} token MTP, non-argmax: {mism}, scarto logit in quei punti: {margins}")
