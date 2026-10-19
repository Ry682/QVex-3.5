"""Two-turn chat through the Engine: checks streaming, prompt cache and stats."""
import sys, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.engine import Engine
from engine.sampler import SamplingParams

MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
eng = Engine(MODEL)

def ask(messages, thinking):
    done = threading.Event(); out = {"content": "", "reasoning": ""}
    def emit(ev):
        if ev["type"] in ("content", "reasoning"):
            out[ev["type"]] += ev["text"]
        elif ev["type"] == "done":
            out["stats"] = ev["stats"]; out["finish"] = ev["finish_reason"]; done.set()
        elif ev["type"] == "error":
            print("ERROR", ev); done.set()
    eng.submit(messages, SamplingParams(temperature=0.7, top_p=0.8, max_tokens=200), thinking, emit)
    done.wait()
    return out

msgs = [{"role": "user", "content": "Ciao! In una frase: cos'è la memoria unificata?"}]
r = ask(msgs, False)
print("T1:", r["content"].strip()[:300]); print("   ", r["finish"], {k: round(v, 3) if isinstance(v, float) else v for k, v in r["stats"].items()})
msgs += [{"role": "assistant", "content": r["content"]}, {"role": "user", "content": "E perché aiuta i modelli linguistici?"}]
r = ask(msgs, True)
print("T2 reasoning chars:", len(r["reasoning"]), "| content:", r["content"].strip()[:200])
print("   ", r["finish"], {k: round(v, 3) if isinstance(v, float) else v for k, v in r["stats"].items()})
