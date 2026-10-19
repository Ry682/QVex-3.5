"""Reasoning budget vs. quality and energy on the task suite (quant/tasks.py).

    .venv/bin/python -m bench.thinking_budget 0 512 2048 free

Runs through the Engine exactly like the server (thinking sampling defaults, thermal governor),
MTP off so energy compares like with like. Results go to results/thinking_budget.json after
every task, so an interrupted run keeps what it measured.
"""

import argparse
import json
import threading
from pathlib import Path

from engine.engine import Engine
from engine.server import DEFAULT_MODEL, _params
from quant.tasks import TASKS

OUT = Path(__file__).resolve().parent.parent / "results/thinking_budget.json"


def ask(engine, prompt, budget):
    thinking = budget != 0
    params = _params({"max_tokens": 6144, "seed": 0}, thinking)
    content, done, result = [], threading.Event(), {}

    def emit(ev):
        if ev["type"] == "content":
            content.append(ev["text"])
        elif ev["type"] == "done":
            result.update(ev)
            done.set()

    engine.submit([{"role": "user", "content": prompt}], params, thinking, emit,
                  budget if budget and budget > 0 else None)
    done.wait()
    return "".join(content), result["finish_reason"], result["stats"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("budgets", nargs="+", help="0 = thinking off, N = token budget, free = unlimited")
    args = ap.parse_args()
    engine = Engine(DEFAULT_MODEL, mtp_mode="off")
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for b in args.budgets:
        budget = -1 if b == "free" else int(b)
        rows = []
        for i, (prompt, check) in enumerate(TASKS):
            answer, finish, s = ask(engine, prompt, budget)
            ok = bool(check(answer))
            rows.append({"ok": ok, "finish": finish, "tokens": s["completion_tokens"],
                         "reasoning_tokens": s.get("reasoning_tokens", 0), "cut": s.get("reasoning_cut", False),
                         "energy_j": s["energy_j"], "seconds": s["completion_tokens"] / (s["decode_tps"] or 1),
                         "max_temp_c": s["max_temp_c"]})
            r = rows[-1]
            print(f"[{b:>4s}] {i + 1:2d}/{len(TASKS)} {'ok ' if ok else 'NO '} {r['reasoning_tokens']:5d} ragion. "
                  f"{r['tokens']:5d} tot  {r['energy_j']:6.0f} J{'  (tagliato)' if r['cut'] else ''}", flush=True)
            results[b] = {"passed": sum(x["ok"] for x in rows), "rows": rows}
            OUT.write_text(json.dumps(results, indent=1))

    print(f"\n{'budget':>7s} {'compiti':>8s} {'token':>7s} {'ragion.':>8s} {'J':>6s} {'s':>5s} {'max °C':>7s}")
    for b, r in results.items():
        rows = r["rows"]
        n = len(rows)
        print(f"{b:>7s} {r['passed']:>4d}/{n:<3d} {sum(x['tokens'] for x in rows) / n:7.0f} "
              f"{sum(x['reasoning_tokens'] for x in rows) / n:8.0f} {sum(x['energy_j'] for x in rows) / n:6.0f} "
              f"{sum(x['seconds'] for x in rows) / n:5.0f} {max(x['max_temp_c'] or 0 for x in rows):7.0f}")


if __name__ == "__main__":
    main()
