"""Decode speed and energy through the engine's own decode loop (the path the server uses).

    .venv/bin/python -m quant.speed lmstudio qvex-q4 qvex-mix40 --rounds 2 --tokens 512

Models are alternated round by round and every run starts below COOL_C, so clock boosts and
heat soak hit all of them alike. The model is the only thing that changes between runs.
"""

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from engine.model import load_model
from engine.sampler import Sampler, SamplingParams
from engine.speculative import decode
from engine.telemetry import PowerMonitor
from engine.tokenizer import ChatTokenizer

from .common import MLX_DIR, ROOT
from .evaluate import RESULTS, SPEED_PROMPT, model_dir

COOL_C = 50.0


def run(model, ids, n):
    sampler = Sampler(SamplingParams(temperature=0.0, max_tokens=n), model.cfg.vocab_size)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache)
    mx.eval(logits)
    count = 0
    t0 = time.perf_counter()
    for _ in decode(model, sampler, cache, logits[0, -1], use_mtp=lambda: False):
        count += 1
        if count >= n:
            break
    return t0, time.perf_counter(), count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=512)
    args = ap.parse_args()
    tok = ChatTokenizer(MLX_DIR)
    ids = tok.encode(tok.render([{"role": "user", "content": SPEED_PROMPT}], True, False))
    monitor = PowerMonitor(interval_ms=250).start()
    runs = {m: [] for m in args.models}
    for r in range(args.rounds):
        for name in args.models:
            model = load_model(model_dir(name))
            run(model, ids, 16)  # compile kernels
            while monitor.current_temp() > COOL_C:
                time.sleep(2)
            t0, t1, n = run(model, ids, args.tokens)
            w = monitor.window(t0, t1)
            tps = n / (t1 - t0)
            runs[name].append({"tps": tps, "watts": w["soc_w"], "j_per_token": w["soc_w"] / tps,
                               "max_temp_c": max(w["max_gpu_c"], w["max_cpu_c"]), "gpu_mhz": w["gpu_mhz"]})
            x = runs[name][-1]
            print(f"giro {r + 1} {name:11s} {tps:5.1f} tok/s  {x['watts']:4.1f} W  {x['j_per_token']:.3f} J/token  "
                  f"{x['gpu_mhz']:4.0f} MHz  max {x['max_temp_c']:.0f} °C", flush=True)
            del model
            mx.clear_cache()
    monitor.stop()

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    print(f"\n{'modello':11s} {'tok/s':>6s} {'W':>5s} {'J/tok':>6s}")
    for name, rs in runs.items():
        s = {k: float(np.median([x[k] for x in rs])) for k in rs[0]}
        results.setdefault(name, {})["speed"] = {**s, "method": "engine decode", "tokens": args.tokens, "runs": len(rs)}
        print(f"{name:11s} {s['tps']:6.1f} {s['watts']:5.1f} {s['j_per_token']:6.3f}")
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
