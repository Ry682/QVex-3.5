"""Normal vs MTP-speculative decoding: correctness (greedy) + tok/s + J/token.

    .venv/bin/python bench/mtp_bench.py [--bits 4] [--hidden pre|post] [--tokens 256] [--temp 0]
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mlx.core as mx
from engine.model import load_model
from engine.mtp import load_mtp
from engine.sampler import Sampler, SamplingParams
from engine.speculative import decode, mtp_prefill
from engine.telemetry import PowerMonitor
from engine.tokenizer import ChatTokenizer

MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
ROOT = Path(__file__).resolve().parent.parent
PROMPTS = {
    "codice": "Scrivi una classe Python LRUCache con get e put in O(1), con docstring e test.",
    "prosa": "Racconta in 300 parole la storia di un faro abbandonato sulla costa ligure.",
    "spiegazione": "Spiega come funziona la memoria unificata dei chip Apple Silicon e perché conta per gli LLM.",
}

ap = argparse.ArgumentParser()
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--hidden", default="pre", choices=["pre", "post"])
ap.add_argument("--tokens", type=int, default=256)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--only", default=None)
args = ap.parse_args()

model = load_model(MODEL)
mtp = load_mtp(model, ROOT / f"models/qwen3.5-9b-mtp/mtp-{args.bits}bit.safetensors", args.hidden)
tok = ChatTokenizer(MODEL)
mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])


def params():
    return SamplingParams(temperature=args.temp, top_p=0.8, top_k=20, seed=0)


def run(ids, use_mtp):
    cache, mcache = model.make_cache(), mtp.make_cache()
    arr = mx.array([ids])
    logits, hidden = model(arr, cache, return_hidden=True)
    last_h = mtp_prefill(mtp, mcache, hidden, arr, None)
    s = Sampler(params(), model.cfg.vocab_size)
    stats, out = {}, []
    t0 = time.perf_counter()
    gen = decode(model, s, cache, logits[0, -1], last_h, mtp if use_mtp else None, mcache, stats=stats)
    for t in gen:
        out.append(t)
        if len(out) >= args.tokens:
            break
    return out, time.perf_counter() - t0, stats if use_mtp else {}


run_normal = lambda ids: run(ids, False)
run_mtp = lambda ids: run(ids, True)


with PowerMonitor(250) as mon:
    for name, text in PROMPTS.items():
        if args.only and name != args.only:
            continue
        ids = tok.encode(tok.render([{"role": "user", "content": text}], True, False))
        res = {}
        for label, fn in (("normale", run_normal), ("MTP", run_mtp)):
            time.sleep(3)
            t_start = time.perf_counter()
            out, dt, st = fn(ids)
            pw = mon.window(t_start + (time.perf_counter() - t_start - dt), time.perf_counter())
            tps = len(out) / dt
            res[label] = out
            extra = ""
            if st:
                extra = f"  accettati {st['accepted']}/{st['cycles']} ({100 * st['accepted'] / max(1, st['cycles']):.0f}%), {len(out) / st['cycles']:.2f} tok/ciclo"
            print(f"[{name:11s}] {label:7s} {tps:5.1f} tok/s  {pw.get('soc_w', 0):4.1f} W  "
                  f"{pw.get('soc_w', 0) / tps:.3f} J/tok  max {pw.get('max_gpu_c', 0):.0f}°C{extra}", flush=True)
        if args.temp == 0:
            a, b = res["normale"], res["MTP"]
            same = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
            print(f"[{name:11s}] greedy identici: {same}/{len(a)}")
