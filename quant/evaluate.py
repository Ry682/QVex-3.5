"""Compare models against the bf16 original: quality, capability checks, speed and energy.

    .venv/bin/python -m quant.evaluate lmstudio qvex-q4 qvex-mix ...   (names under models/, or 'lmstudio')

Quality, on the held-out evaluation windows (never used for calibration):
    KL       mean KL divergence from the bf16 model's next-token distribution (lower is better;
             0 = identical predictions). The standard way llama.cpp / ds4 compare quantizations.
    top-1    how often the most likely next token matches bf16
    ppl      perplexity on the real text (bf16 is printed as the floor)
Capability: quant/tasks.py, greedy, thinking off.
Speed: 256 greedy tokens after the chip has cooled below 55 °C, with average SoC watts.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import numpy as np

from engine.model import load_model
from engine.telemetry import PowerMonitor
from engine.tokenizer import ChatTokenizer

from .common import DATA, MLX_DIR, ROOT, kl_vs_ref, load_tokens, log_softmax
from .tasks import TASKS

RESULTS = ROOT / "results" / "quant.json"
SPEED_PROMPT = "Spiega in dettaglio come funziona un motore a combustione interna a quattro tempi."


def model_dir(name: str) -> Path:
    return MLX_DIR if name == "lmstudio" else ROOT / "models" / name


def greedy(model, prompt_ids, max_new, stop=()):
    cache = model.make_cache()
    y = mx.argmax(model(mx.array([prompt_ids]), cache)[0, -1])
    mx.async_eval(y)
    out = []
    for _ in range(max_new):
        nxt = mx.argmax(model(y.reshape(1, 1), cache)[0, -1])
        mx.async_eval(nxt)
        t = y.item()
        if t in stop:
            break
        out.append(t)
        y = nxt
    return out


def quality(model, evals, eval_src, ref, n_cal):
    per = {}
    for j, ids in enumerate(evals):
        lp = log_softmax(model(mx.array(ids)[None], model.make_cache(), last_only=False)[0])
        r_idx = mx.array(ref["idx"][n_cal + j])
        r_lp = mx.array(ref["lp"][n_cal + j].astype(np.float32))
        kl = kl_vs_ref(lp, r_idx, r_lp).mean()
        top_ref = mx.take_along_axis(r_idx, mx.argmax(r_lp, -1, keepdims=True), -1)[:, 0]
        agree = (mx.argmax(lp, -1) == top_ref).mean()
        nll = -mx.take_along_axis(lp[:-1], mx.array(ids[1:, None]), -1).mean()
        mx.eval(kl, agree, nll)
        per.setdefault(str(eval_src[j]), []).append((kl.item(), agree.item(), nll.item()))
    allv = np.array([v for vs in per.values() for v in vs])
    out = {"kl": float(allv[:, 0].mean()), "top1": float(allv[:, 1].mean()), "ppl": float(np.exp(allv[:, 2].mean()))}
    out["by_source"] = {s: {"kl": float(np.mean([v[0] for v in vs])), "top1": float(np.mean([v[1] for v in vs]))}
                        for s, vs in per.items()}
    return out


def tasks(model, tok):
    passed, detail = 0, []
    for prompt, check in TASKS:
        ids = tok.encode(tok.render([{"role": "user", "content": prompt}], True, False))
        ans = tok.decode(greedy(model, ids, 400, tok.stop_ids))
        ok = bool(check(ans))
        passed += ok
        detail.append({"prompt": prompt[:60], "ok": ok, "answer": ans[:300]})
    return {"passed": passed, "total": len(TASKS), "detail": detail}


def speed(model, tok, monitor, n=256):
    while monitor.current_temp() > 55:
        time.sleep(2)
    ids = tok.encode(tok.render([{"role": "user", "content": SPEED_PROMPT}], True, False))
    greedy(model, ids, 8)  # warm the kernels
    t0 = time.perf_counter()
    greedy(model, ids, n)
    t1 = time.perf_counter()
    w = monitor.window(t0, t1)
    tps = n / (t1 - t0)
    return {"tps": tps, "watts": w.get("soc_w"), "j_per_token": (w.get("soc_w") or 0) / tps,
            "max_temp_c": max(w.get("max_gpu_c", 0), w.get("max_cpu_c", 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--skip", nargs="*", default=[], choices=["quality", "tasks", "speed"])
    args = ap.parse_args()

    _, evals, _, eval_src = load_tokens()
    ref = np.load(DATA / "reference.npz")
    n_cal = int(ref["n_calib"])
    tok = ChatTokenizer(MLX_DIR)
    monitor = PowerMonitor(interval_ms=250).start()
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results["bf16"] = {"ppl": float(np.exp(-ref["tok_lp"][n_cal:].mean())), "kl": 0.0, "top1": 1.0}

    for name in args.models:
        d = model_dir(name)
        model = load_model(d)
        size = sum(f.stat().st_size for f in d.glob("*.safetensors"))
        r = {"bytes": size, "date": datetime.now().isoformat(timespec="seconds")}
        meta = d / "ours.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            r.update(bpw_linears=m["bpw_linears"], frac_3bit=m["frac_3bit"])
        if "quality" not in args.skip:
            r.update(quality(model, evals, eval_src, ref, n_cal))
            print(f"{name}: KL {r['kl']:.4f}  top-1 {100 * r['top1']:.1f}%  ppl {r['ppl']:.3f}", flush=True)
        if "tasks" not in args.skip:
            r["tasks"] = tasks(model, tok)
            print(f"{name}: compiti {r['tasks']['passed']}/{r['tasks']['total']}", flush=True)
        if "speed" not in args.skip:
            r["speed"] = speed(model, tok, monitor)
            s = r["speed"]
            print(f"{name}: {s['tps']:.1f} tok/s  {s['watts']:.1f} W  {s['j_per_token']:.3f} J/token", flush=True)
        results[name] = {**results.get(name, {}), **r}
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        del model
        mx.clear_cache()
    monitor.stop()

    print(f"\n{'modello':14s} {'GB':>5s} {'KL':>7s} {'top-1':>6s} {'ppl':>6s} {'compiti':>8s} {'tok/s':>6s} {'J/tok':>6s}")
    for name, r in results.items():
        t, s = r.get("tasks"), r.get("speed", {})
        print(f"{name:14s} {r.get('bytes', 0) / 1e9:5.2f} {r.get('kl', float('nan')):7.4f} "
              f"{100 * r.get('top1', float('nan')):5.1f}% {r.get('ppl', float('nan')):6.3f} "
              f"{(str(t['passed']) + '/' + str(t['total'])) if t else '-':>8s} "
              f"{s.get('tps', float('nan')):6.1f} {s.get('j_per_token', float('nan')):6.3f}")


if __name__ == "__main__":
    main()
