"""QVex engine: owns the model on a single worker thread and runs generation jobs.

MLX work stays on one thread (the one that loaded the weights); callers submit
jobs and receive events through a callback.
"""

import gc
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mlx.core as mx

from .model import KVCache, RecurrentCache, load_model
from .mtp import load_mtp
from .sampler import Sampler, SamplingParams
from .speculative import decode, mtp_prefill
from .telemetry import PowerMonitor
from .thermal import ThermalGovernor
from .tokenizer import ChatTokenizer, StreamDecoder

PREFILL_CHUNK = 512
THINK_END = "</think>"
# Qwen's own wording for cutting the reasoning short; the model was trained to answer after it.
THINK_CLOSE = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now.\n</think>\n\n"
THINK_GRACE = 64  # past the budget, wait up to this many tokens for a line break before closing
DEFAULT_MTP = Path(__file__).resolve().parent.parent / "models/qwen3.5-9b-mtp/mtp-4bit.safetensors"
MTP_MODES = ("off", "auto", "on")
AUTO_MTP_MARGIN_C = 5.0  # auto: use MTP only while this far below the thermal target


@dataclass
class Job:
    messages: list[dict]
    params: SamplingParams
    enable_thinking: bool
    emit: Callable[[dict], None]
    thinking_budget: int | None = None  # max reasoning tokens; None = unlimited
    tools: list[dict] | None = None     # function schemas, rendered by the chat template
    cancelled: threading.Event = field(default_factory=threading.Event)


@dataclass
class Control:
    """Load/unload request handled by the worker between jobs."""
    action: str
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class Snapshot:
    """Model state after a conversation prefix (everything up to the last user turn)."""
    tokens: list[int]
    layers: list[dict]
    mtp: dict | None = None
    last_hidden: mx.array | None = None


class Engine:
    def __init__(self, model_dir: str | Path, mode: str = "balanced",
                 mtp_path: str | Path | None = DEFAULT_MTP, mtp_mode: str = "auto", preload: bool = True):
        self.model_dir = Path(model_dir)
        self.mtp_path = Path(mtp_path) if mtp_path and Path(mtp_path).exists() else None
        self.mtp = None
        self.model = None
        self.preload = preload
        self.state = "loading" if preload else "unloaded"  # loading | loaded | unloading | unloaded
        self.mtp_mode = mtp_mode if self.mtp_path else "off"
        self.monitor = PowerMonitor(interval_ms=500).start()
        self.governor = ThermalGovernor(self.monitor, mode)
        self.snapshot: Snapshot | None = None
        self.busy = False
        self._jobs: queue.Queue[Job | Control] = queue.Queue()
        self._ready = threading.Event()
        self._error: Exception | None = None
        threading.Thread(target=self._worker, daemon=True, name="engine").start()
        self._ready.wait()
        if self._error:
            raise self._error

    # ------------------------------------------------------------ public API

    def submit(self, messages, params: SamplingParams, enable_thinking: bool, emit,
               thinking_budget: int | None = None, tools: list[dict] | None = None) -> Job:
        job = Job(messages, params, enable_thinking, emit, thinking_budget, tools)
        self._jobs.put(job)
        return job

    def unload(self):
        """Free the weights and caches; the next chat request loads them back."""
        c = Control("unload")
        self._jobs.put(c)
        c.done.wait()

    def load(self):
        c = Control("load")
        self._jobs.put(c)
        c.done.wait()
        if self._error:
            raise self._error

    def set_mtp_mode(self, mode: str):
        if mode not in MTP_MODES:
            raise ValueError(f"unknown MTP mode {mode!r}")
        if mode != "off" and self.mtp_path is None:
            raise ValueError("MTP head not loaded (run tools/fetch_mtp.py)")
        self.mtp_mode = mode

    def _use_mtp(self) -> bool:
        if self.mtp_mode == "on":
            return True
        if self.mtp_mode == "off":
            return False
        g = self.governor
        return g.tps_cap is None and self.monitor.current_temp() < g.target_c - AUTO_MTP_MARGIN_C

    @property
    def model_name(self) -> str:
        return self.model_dir.name

    # ------------------------------------------------------------ worker

    def _worker(self):
        if self.preload:
            self._load()
        self._ready.set()
        if self._error:  # surfaced to the constructor
            return
        while True:
            job = self._jobs.get()
            if isinstance(job, Control):
                if job.action == "unload":
                    self._unload()
                elif self.model is None:
                    self._load()
                job.done.set()
                continue
            self.busy = True
            try:
                if self.model is None:
                    self._load()
                    if self._error:
                        raise self._error
                self._run(job)
            except Exception as e:
                job.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                self.busy = False
                mx.clear_cache()

    def _load(self):
        self.state, self._error = "loading", None
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
            mx.set_wired_limit(info["max_recommended_working_set_size"])
            self.model = load_model(self.model_dir)
            self.tok = ChatTokenizer(self.model_dir)
            if self.mtp_path:
                self.mtp = load_mtp(self.model, self.mtp_path)
            self._warmup()
            self.state = "loaded"
        except Exception as e:
            self._error = e
            self._unload()

    def _unload(self):
        self.state = "unloading"
        self.model = self.mtp = self.snapshot = None
        gc.collect()
        mx.clear_cache()
        mx.set_wired_limit(0)  # stop pinning memory the weights no longer use
        self.state = "unloaded"

    def _warmup(self):
        cache = self.model.make_cache()
        mx.eval(self.model(mx.array([[1, 2, 3, 4]]), cache))
        mx.eval(self.model(mx.array([[5]]), cache))
        if self.mtp is not None:  # compile the rollback kernel and the MTP path too
            logits, h = self.model(mx.array([[6, 7]]), cache, last_only=False,
                                   return_hidden=True, keep_prev=True)
            mx.eval(logits, self.mtp(h, mx.array([[7, 8]]), self.mtp.make_cache()))

    # ------------------------------------------------------------ prompt cache

    @staticmethod
    def _kv_snap(c: KVCache) -> dict:
        # sliced copies: later in-place KV writes must not leak into the snapshot
        if c.k is None:
            return {"k": None, "v": None, "offset": 0}
        return {"k": c.k[..., : c.offset, :], "v": c.v[..., : c.offset, :], "offset": c.offset}

    @staticmethod
    def _kv_restore(c: KVCache, s: dict):
        c.k, c.v, c.offset = s["k"], s["v"], s["offset"]

    def _take_snapshot(self, tokens, cache, mtp_cache, last_hidden) -> Snapshot:
        layers = []
        for c in cache:
            if isinstance(c, KVCache):
                layers.append(self._kv_snap(c))
            else:
                layers.append({"conv": c.conv, "ssm": c.ssm, "offset": c.offset})
        snap = Snapshot(list(tokens), layers,
                        self._kv_snap(mtp_cache) if mtp_cache is not None else None, last_hidden)
        arrays = [x for l in layers for x in l.values() if isinstance(x, mx.array)]
        if snap.mtp:
            arrays += [x for x in snap.mtp.values() if isinstance(x, mx.array)]
        if last_hidden is not None:
            arrays.append(last_hidden)
        mx.eval(arrays)
        return snap

    def _restore(self, snap: Snapshot):
        cache = self.model.make_cache()
        for c, s in zip(cache, snap.layers):
            if isinstance(c, KVCache):
                self._kv_restore(c, s)
            else:
                c.conv, c.ssm, c.offset = s["conv"], s["ssm"], s["offset"]
        mtp_cache = None
        if self.mtp is not None:
            mtp_cache = self.mtp.make_cache()
            if snap.mtp:
                self._kv_restore(mtp_cache, snap.mtp)
        return cache, mtp_cache, snap.last_hidden

    # ------------------------------------------------------------ generation

    def _prefill(self, ids, cache, mtp_cache, last_hidden, job):
        """Returns (logits of the last token, its hidden state) or (None, None) if cancelled."""
        logits = None
        for i in range(0, len(ids), PREFILL_CHUNK):
            if job.cancelled.is_set():
                return None, None
            t0 = time.perf_counter()
            chunk = mx.array([ids[i : i + PREFILL_CHUNK]])
            if mtp_cache is not None:
                logits, hidden = self.model(chunk, cache, return_hidden=True)
                mx.eval(logits, [x for c in cache for x in c.state()])
                last_hidden = mtp_prefill(self.mtp, mtp_cache, hidden, chunk, last_hidden)
            else:
                logits = self.model(chunk, cache)
                mx.eval(logits, [x for c in cache for x in c.state()])
            self.governor.pace_prefill(time.perf_counter() - t0)
        return logits, last_hidden

    def _run(self, job: Job):
        tok, model, gov = self.tok, self.model, self.governor
        prompt_ids = tok.encode(tok.render(job.messages, True, job.enable_thinking, job.tools))
        base_ids = tok.encode(tok.render(job.messages, False, job.enable_thinking, job.tools))
        snap_at = len(base_ids) if prompt_ids[: len(base_ids)] == base_ids else 0

        # reuse the state of the previous turn when this prompt extends it
        start, cache, mtp_cache, last_hidden = 0, None, None, None
        s = self.snapshot
        if s and len(s.tokens) < len(prompt_ids) and prompt_ids[: len(s.tokens)] == s.tokens:
            cache, mtp_cache, last_hidden = self._restore(s)
            start = len(s.tokens)
        if cache is None:
            cache = model.make_cache()
            mtp_cache = self.mtp.make_cache() if self.mtp is not None else None
        cached = start

        gov.begin()
        t_start = time.perf_counter()
        job.emit({"type": "start", "prompt_tokens": len(prompt_ids), "cached_tokens": cached})

        if snap_at > start:
            logits, last_hidden = self._prefill(prompt_ids[start:snap_at], cache, mtp_cache, last_hidden, job)
            if logits is None:
                return job.emit({"type": "done", "finish_reason": "cancelled", "stats": {}})
            self.snapshot = self._take_snapshot(prompt_ids[:snap_at], cache, mtp_cache, last_hidden)
            start = snap_at
        logits, last_hidden = self._prefill(prompt_ids[start:], cache, mtp_cache, last_hidden, job)
        if logits is None:
            return job.emit({"type": "done", "finish_reason": "cancelled", "stats": {}})
        t_first = time.perf_counter()

        sampler = Sampler(job.params, model.cfg.vocab_size)
        decoder = StreamDecoder(tok)
        in_reasoning = job.enable_thinking
        max_tokens = job.params.max_tokens

        spec, forced = {}, []
        tokens = decode(model, sampler, cache, logits[0, -1], last_hidden,
                        self.mtp, mtp_cache, self._use_mtp, spec, forced)
        budget = job.thinking_budget if job.enable_thinking else None
        n, n_reasoning, closing, answered, finish = 0, 0, False, False, "length"
        t_last_stats = t_first
        for t in tokens:
            n += 1
            if t in tok.stop_ids:
                finish = "stop"
                break

            text = decoder.push(t)
            if in_reasoning:
                n_reasoning += 1
                if budget is not None and not closing and n_reasoning >= budget and \
                        (n_reasoning >= budget + THINK_GRACE or "\n" in (text or "")):
                    closing = True
                    forced.extend(tok.encode(THINK_CLOSE))
            if text:
                if in_reasoning and THINK_END in text:
                    before, text = text.split(THINK_END, 1)
                    if before:
                        job.emit({"type": "reasoning", "text": before})
                    in_reasoning = False
                if in_reasoning:
                    job.emit({"type": "reasoning", "text": text})
                else:
                    if not answered:  # the line breaks after </think> can arrive as separate tokens
                        text = text.lstrip("\n")
                        answered = bool(text)
                    if text:
                        job.emit({"type": "content", "text": text})

            now = time.perf_counter()
            tps = n / (now - t_first) if now > t_first else None
            if now - t_last_stats > 0.5:
                t_last_stats = now
                job.emit({"type": "stats", "tokens": n, **self._live(tps)})
            if n >= max_tokens:
                break
            if job.cancelled.is_set():
                finish = "cancelled"
                break
            gov.pace_token(tps)
        tokens.close()

        tail = decoder.flush()
        if tail:
            job.emit({"type": "reasoning" if in_reasoning else "content", "text": tail})

        t_end = time.perf_counter()
        pre = self.monitor.window(t_start, t_first)
        dec = self.monitor.window(t_first, t_end)
        whole = self.monitor.window(t_start, t_end)
        decode_s = t_end - t_first
        stats = {
            "prompt_tokens": len(prompt_ids),
            "cached_tokens": cached,
            "completion_tokens": n,
            "ttft_s": t_first - t_start,
            "prefill_tps": None,
            "decode_tps": n / decode_s if decode_s > 0 else None,
            "energy_j": whole.get("soc_w", 0) * (t_end - t_start),
            "avg_w": whole.get("soc_w"),
            "decode_w": dec.get("soc_w"),
            "prefill_w": pre.get("soc_w"),
            "max_temp_c": max(whole.get("max_gpu_c", 0), whole.get("max_cpu_c", 0)) or None,
            "throttled_s": gov.throttled_s,
            "mtp_cycles": spec.get("cycles", 0),
            "mtp_accepted": spec.get("accepted", 0),
            "reasoning_tokens": n_reasoning,
            "reasoning_cut": closing,
        }
        processed = len(prompt_ids) - stats["cached_tokens"]
        if t_first > t_start:
            stats["prefill_tps"] = processed / (t_first - t_start)
        if n:
            stats["j_per_token"] = stats["energy_j"] / n
        job.emit({"type": "done", "finish_reason": finish, "stats": stats})

    def _live(self, tps):
        x = self.monitor.latest()
        g = self.governor
        return {
            "tps": tps,
            "watts": x.get("soc_w"),
            "temp_c": max(x.get("gpu_c", 0), x.get("cpu_c", 0)) or None,
            "tps_cap": g.tps_cap,
            "mode": g.mode,
            "mtp_active": self.mtp_mode != "off" and self._use_mtp(),
        }

    def telemetry(self) -> dict:
        x = self.monitor.latest()
        g = self.governor
        return {
            "watts": x.get("soc_w"),
            "gpu_w": x.get("gpu_w"),
            "cpu_w": x.get("cpu_w"),
            "gpu_c": x.get("gpu_c"),
            "cpu_c": x.get("cpu_c"),
            "gpu_mhz": x.get("gpu_mhz"),
            "gpu_pct": x.get("gpu_pct"),
            "cpu_pct": x.get("cpu_pct"),
            "ram_used_gb": x.get("ram_used_gb"),
            "ram_total_gb": x.get("ram_total_gb"),
            "mode": g.mode,
            "target_c": g.target_c,
            "tps_cap": g.tps_cap,
            "busy": self.busy,
            "mtp_mode": self.mtp_mode,
            "mtp_available": self.mtp_path is not None,
            "model_state": self.state,
            "memory_gb": mx.get_active_memory() / 1e9,
            "context_cached": len(self.snapshot.tokens) if self.snapshot else 0,  # tokens kept for the next turn
        }
