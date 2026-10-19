"""Build the calibration and evaluation texts for our own quantization.

    .venv/bin/python -m quant.data

Four sources, mirroring how the engine is actually used:
    it    Italian prose, public-domain books from Project Gutenberg
    en    English prose, public-domain books from Project Gutenberg
    code  Python standard library sources
    chat  conversations answered by the current model (rendered in the chat template)

Calibration and evaluation never share a document. Output: data/texts.json (raw, reproducible)
and data/tokens.npz with calib/eval arrays of SEQ-token windows plus their source labels.
"""

import argparse
import json
import random
import sysconfig
import time
import urllib.request
from pathlib import Path

import numpy as np

from engine.tokenizer import ChatTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
SEQ = 512
# windows per source: (calibration, evaluation)
WINDOWS = {"it": (16, 12), "en": (8, 6), "code": (16, 10), "chat": (24, 12)}
UA = {"User-Agent": "local-engine-quant/0.1 (personal research)"}

CHAT_PROMPTS = [
    # (prompt, thinking)
    ("Spiegami come funziona la fotosintesi a un ragazzo di 14 anni.", False),
    ("Scrivi una funzione Python che trova i numeri primi fino a N con il crivello di Eratostene, con test.", False),
    ("Scrivi una email formale al mio condominio per segnalare un'infiltrazione d'acqua dal tetto.", False),
    ("Quali sono le differenze tra TCP e UDP? Fai degli esempi pratici.", False),
    ("Un treno parte alle 14:35 e arriva alle 18:10 percorrendo 290 km. Qual è la velocità media?", True),
    ("Riassumi le cause della prima guerra mondiale in 10 punti.", False),
    ("Come posso organizzare un viaggio di 5 giorni in Sicilia con un budget di 800 euro?", False),
    ("Scrivi un componente React con un contatore e un pulsante di reset, spiegando gli hook usati.", False),
    ("Qual è la differenza tra un mutuo a tasso fisso e uno a tasso variabile?", False),
    ("Traduci in inglese: 'Domani mattina passo in ufficio a ritirare i documenti, poi ti chiamo.'", False),
    ("Dimostra che la radice quadrata di 2 è irrazionale.", True),
    ("Scrivi una poesia in endecasillabi sull'autunno a Milano.", False),
    ("Cos'è il machine learning? Spiega la differenza tra apprendimento supervisionato e non.", False),
    ("Correggi questo codice: def media(l): return sum(l)/len(l) -- cosa succede con una lista vuota?", False),
    ("Dammi una ricetta della carbonara autentica, con le dosi per 4 persone.", False),
    ("Come funziona la memoria unificata dei Mac con Apple Silicon?", False),
    ("Se ho 3 camicie, 4 pantaloni e 2 paia di scarpe, quanti abbinamenti diversi posso fare? E se una camicia non va con due pantaloni?", True),
    ("Scrivi uno script bash che fa il backup di una cartella in un archivio con la data nel nome.", False),
    ("Quali sono i pro e i contro del lavoro da remoto?", False),
    ("Spiega la teoria della relatività ristretta senza formule complicate.", False),
    ("Write a SQL query that returns the top 5 customers by total order value, with a short explanation.", False),
    ("Raccontami la storia dell'Impero Romano d'Occidente fino alla sua caduta.", False),
    ("Come si calcola l'IRPEF in Italia? Fai un esempio con un reddito di 35.000 euro.", True),
    ("Scrivi una classe Python per una coda con priorità usando heapq, con docstring.", False),
    ("Che consigli daresti a chi vuole iniziare a correre?", False),
    ("Spiega cosa sono le API REST e come si progetta un buon endpoint.", False),
    ("Scrivi un breve racconto di fantascienza ambientato a Napoli nel 2150.", False),
    ("Qual è la differenza tra virus e batteri? Perché gli antibiotici non funzionano sui virus?", False),
    ("Risolvi: 2x^2 - 7x + 3 = 0, spiegando i passaggi.", True),
    ("Scrivi in JavaScript una funzione debounce e spiega quando usarla.", False),
    ("Come funziona il sistema elettorale italiano?", False),
    ("Explain the difference between a process and a thread, with examples.", False),
    ("Prepara una scaletta per una presentazione di 10 minuti sull'energia solare.", False),
    ("Perché il cielo è blu e il tramonto è rosso?", False),
    ("Scrivi un programma C che inverte una stringa in place, gestendo i casi limite.", False),
    ("Quanti giorni ci sono tra il 3 marzo 2024 e il 17 agosto 2024?", True),
    ("Spiega il funzionamento di git rebase rispetto a git merge.", False),
    ("Scrivi la lettera di presentazione per un lavoro da sviluppatore junior.", False),
    ("Cosa sono i buchi neri e come si formano?", False),
    ("Dimmi cinque libri italiani del Novecento da leggere e perché.", False),
    ("Implementa una ricerca binaria in Rust con test.", False),
    ("Come posso ridurre i consumi elettrici di casa?", False),
    ("Spiega cos'è l'inflazione e come la BCE prova a controllarla.", False),
    ("Fammi un piano di studio di 4 settimane per imparare Python da zero.", False),
    ("Una vasca si riempie in 6 ore con un rubinetto e in 4 con un altro. Insieme quanto ci mettono?", True),
    ("Descrivi le differenze tra stile gotico e romanico nell'architettura.", False),
    ("Write a short product description for a fanless laptop aimed at students.", False),
    ("Spiega come funziona una rete neurale transformer, in modo intuitivo.", False),
]


GUTENBERG = {
    # prose only (no medieval verse): calibration takes the first books of each list, evaluation the rest
    "it": [52484, 45334, 25179, 25178, 26453, 22020, 38628, 38720, 61885, 22507, 65391, 26452, 25587, 22501],
    "en": [1342, 84, 2701, 1661, 345, 145, 2641, 98, 43, 174, 1260, 11],
}


def gutenberg_books(lang, chars=24_000):
    """A slice from the middle of each book (skips the licence header and front matter)."""
    out = []
    for book in GUTENBERG[lang]:
        req = urllib.request.Request(f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt", headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            text = r.read().decode("utf-8", errors="ignore").replace("\r\n", "\n")
        start = text.find("*** START OF")
        end = text.find("*** END OF")
        body = text[start:end] if start >= 0 and end > start else text
        mid = max(0, len(body) // 3)
        piece = body[mid: mid + chars]
        piece = piece[piece.find("\n\n") + 2:]  # start on a paragraph
        out.append({"title": f"gutenberg {book}", "text": piece})
        print(f"  {lang}: gutenberg {book} ({len(piece)} chars)", flush=True)
        time.sleep(1)
    return out


def code_files(n, rng):
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(p for p in stdlib.glob("*.py") if 8_000 < p.stat().st_size < 60_000)
    rng.shuffle(files)
    return [{"title": p.name, "text": p.read_text(errors="ignore")} for p in files[:n]]


def chat_samples(cache_path):
    """Answers from the current model, generated in-process (no server); resumable via cache_path."""
    import queue

    from engine.engine import Engine
    from engine.server import _params

    done = json.loads(cache_path.read_text()) if cache_path.exists() else []
    if len(done) >= len(CHAT_PROMPTS):
        return done
    engine = Engine(MODEL, mode="balanced", mtp_mode="off")
    for i, (prompt, thinking) in enumerate(CHAT_PROMPTS[len(done):], start=len(done)):
        events = queue.Queue()
        params = _params({"max_tokens": 700}, thinking)
        engine.submit([{"role": "user", "content": prompt}], params, thinking, events.put)
        reasoning, answer = [], []
        while (ev := events.get())["type"] not in ("done", "error"):
            (reasoning if ev["type"] == "reasoning" else answer if ev["type"] == "content" else []).append(ev.get("text", ""))
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
        done.append({"title": prompt[:60], "prompt": prompt, "thinking": thinking,
                     "reasoning": "".join(reasoning), "answer": "".join(answer)})
        cache_path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
        print(f"  chat {i + 1}/{len(CHAT_PROMPTS)}: {prompt[:50]}", flush=True)
    engine.monitor.stop()
    return done


def chat_text(tok, s):
    """Render a sample exactly as the engine sees it: template prompt + (thinking) + answer."""
    text = tok.render([{"role": "user", "content": s["prompt"]}], True, s["thinking"])
    if s["thinking"]:
        text += s["reasoning"].strip("\n") + "\n</think>\n\n"
    return text + s["answer"] + "<|im_end|>\n"


def windows(tok, docs, n, render=lambda d: d["text"]):
    """Concatenate documents and cut SEQ-token windows; returns up to n windows."""
    ids = []
    for d in docs:
        ids += tok.encode(render(d))
        if len(ids) >= n * SEQ:
            break
    ids = ids[: (len(ids) // SEQ) * SEQ]
    w = np.array(ids, dtype=np.int32).reshape(-1, SEQ)[:n]
    if len(w) < n:
        raise RuntimeError(f"only {len(w)} windows, wanted {n}: add documents")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="re-tokenize data/texts.json without fetching")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    rng = random.Random(0)

    texts_path = DATA / "texts.json"
    if args.reuse:
        texts = json.loads(texts_path.read_text())
    else:
        texts = {
            "it": gutenberg_books("it"),
            "en": gutenberg_books("en"),
            "code": code_files(40, rng),
            "chat": chat_samples(DATA / "chat_cache.json"),
        }
        texts_path.write_text(json.dumps(texts, ensure_ascii=False, indent=1))

    tok = ChatTokenizer(MODEL)
    calib, evals, calib_src, eval_src = [], [], [], []
    for src, (nc, ne) in WINDOWS.items():
        docs = texts[src]
        render = (lambda d: chat_text(tok, d)) if src == "chat" else (lambda d: d["text"])
        # split documents first so no text is shared between the two sets
        cut = len(docs) * nc // (nc + ne)
        c, e = windows(tok, docs[:cut], nc, render), windows(tok, docs[cut:], ne, render)
        calib.append(c); evals.append(e)
        calib_src += [src] * len(c); eval_src += [src] * len(e)
        print(f"{src:5s} calib {len(c):3d}  eval {len(e):3d} windows of {SEQ}")
    np.savez(DATA / "tokens.npz", calib=np.concatenate(calib), eval=np.concatenate(evals),
             calib_src=np.array(calib_src), eval_src=np.array(eval_src))
    print(f"wrote {DATA / 'tokens.npz'}")


if __name__ == "__main__":
    main()
