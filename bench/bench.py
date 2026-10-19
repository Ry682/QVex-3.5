"""Speed + energy + temperature benchmark for MLX models on Apple Silicon.

Usage:
    .venv/bin/python bench/bench.py --sustain 90
"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.telemetry import PowerMonitor

DEFAULT_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

FILLER = (
    "The history of computing spans centuries, from mechanical calculators to "
    "modern processors with billions of transistors. Engineers balance speed, "
    "power, and heat; a laptop without a fan must spread its thermal load over "
    "time. Memory bandwidth often limits language model inference more than raw "
    "arithmetic, because every generated token reads all of the weights once. "
)


def make_prompt(tokenizer, n_tokens: int) -> list[int]:
    ids = tokenizer.encode(FILLER)
    reps = math.ceil(n_tokens / len(ids)) + 1
    return (ids * reps)[:n_tokens]


def no_eos_processor(eos_ids):
    mask = None

    def proc(tokens, logits):
        nonlocal mask
        if mask is None:
            m = [0.0] * logits.shape[-1]
            for e in eos_ids:
                if e < len(m):
                    m[e] = -math.inf
            mask = mx.array(m, dtype=logits.dtype)
        return logits + mask

    return proc


def cool_down(mon: PowerMonitor, target_c: float, max_wait: float):
    t0 = time.perf_counter()
    while mon.current_temp() > target_c and time.perf_counter() - t0 < max_wait:
        time.sleep(1)


def run_case(model, tok, mon, procs, ctx: int, gen: int, prefill_step: int) -> dict:
    prompt = make_prompt(tok, ctx)
    t_start = time.perf_counter()
    t_first = None
    last = None
    for r in stream_generate(
        model, tok, prompt, max_tokens=gen,
        logits_processors=procs, prefill_step_size=prefill_step,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        last = r
    t_end = time.perf_counter()
    time.sleep(0.3)  # let the sampler catch the tail
    pw_pre = mon.window(t_start, t_first)
    pw_dec = mon.window(t_first, t_end)
    return {
        "ctx": ctx,
        "gen": last.generation_tokens,
        "prefill_tps": last.prompt_tps,
        "decode_tps": last.generation_tps,
        "peak_mem_gb": last.peak_memory,
        "prefill_power": pw_pre,
        "decode_power": pw_dec,
        "prefill_j_per_tok": pw_pre.get("soc_w", 0) / last.prompt_tps,
        "decode_j_per_tok": pw_dec.get("soc_w", 0) / last.generation_tps,
    }


def run_sustained(model, tok, mon, procs, seconds: float, bucket: float = 10.0) -> list[dict]:
    """Decode continuously and report throughput/power/temp per time bucket."""
    prompt = make_prompt(tok, 512)
    buckets = []
    t0 = time.perf_counter()
    b_start, b_tokens = None, 0
    for r in stream_generate(model, tok, prompt, max_tokens=10**6, logits_processors=procs):
        now = time.perf_counter()
        if b_start is None:
            b_start = now
            continue
        b_tokens += 1
        if now - b_start >= bucket:
            pw = mon.window(b_start, now)
            buckets.append({
                "t": round(now - t0),
                "tps": b_tokens / (now - b_start),
                "soc_w": pw.get("soc_w", 0),
                "gpu_mhz": pw.get("gpu_mhz", 0),
                "gpu_c": pw.get("max_gpu_c", 0),
            })
            b = buckets[-1]
            print(f"  t={b['t']:>4}s  {b['tps']:5.1f} tok/s  {b['soc_w']:5.1f} W  "
                  f"{b['gpu_mhz']:5.0f} MHz  GPU {b['gpu_c']:5.1f}°C", flush=True)
            b_start, b_tokens = now, 0
        if now - t0 >= seconds:
            break
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--ctx", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--prefill-step", type=int, default=2048)
    ap.add_argument("--sustain", type=float, default=0, help="seconds of sustained decode (0 = skip)")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tok = load(args.model)
    procs = [no_eos_processor(tok.eos_token_ids)]

    with PowerMonitor() as mon:
        idle = mon.window(time.perf_counter() - 0.5, time.perf_counter())
        print(f"Idle: SoC {idle.get('soc_w', 0):.1f} W, GPU {idle.get('max_gpu_c', 0):.1f}°C")

        # warmup (compiles kernels, faults in weights)
        run_case(model, tok, mon, procs, 64, 16, args.prefill_step)

        cases = []
        for ctx in args.ctx:
            cool_down(mon, idle.get("max_gpu_c", 40) + 8, 30)
            c = run_case(model, tok, mon, procs, ctx, args.gen, args.prefill_step)
            cases.append(c)
            print(
                f"ctx={ctx:>5}  prefill {c['prefill_tps']:7.1f} tok/s @ {c['prefill_power'].get('soc_w', 0):4.1f} W "
                f"({c['prefill_j_per_tok']*1000:5.1f} mJ/tok)  |  "
                f"decode {c['decode_tps']:5.1f} tok/s @ {c['decode_power'].get('soc_w', 0):4.1f} W "
                f"({c['decode_j_per_tok']:.3f} J/tok)  |  "
                f"max GPU {max(c['prefill_power'].get('max_gpu_c', 0), c['decode_power'].get('max_gpu_c', 0)):.0f}°C  "
                f"mem {c['peak_mem_gb']:.2f} GB"
            )

        sustained = []
        if args.sustain > 0:
            cool_down(mon, idle.get("max_gpu_c", 40) + 8, 60)
            print(f"Sustained decode for {args.sustain:.0f}s:")
            sustained = run_sustained(model, tok, mon, procs, args.sustain)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{args.tag}.json"
    out.write_text(json.dumps({
        "tag": args.tag, "model": args.model, "mlx": mx.__version__,
        "idle": idle, "cases": cases, "sustained": sustained,
    }, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
