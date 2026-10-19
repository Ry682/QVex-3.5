"""HTTP server: OpenAI-compatible chat API + telemetry + web UI.

    .venv/bin/python -m engine.server [--model DIR] [--port 8080] [--mode balanced]
"""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import SYSTEM, TOOLS, Agent, ToolError, check_root
from .engine import DEFAULT_MTP, MTP_MODES, Engine
from .sampler import SamplingParams
from .thermal import MODES
from .tokenizer import ChatTokenizer

LMSTUDIO_MODEL = Path.home() / ".lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
OUR_MODEL = Path(__file__).resolve().parent.parent / "models/qvex-q4"  # built by quant/ (imatrix 4-bit)
DEFAULT_MODEL = OUR_MODEL if (OUR_MODEL / "model.safetensors").exists() else LMSTUDIO_MODEL
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="QVex")


@app.middleware("http")
async def no_stale_ui(request: Request, call_next):
    # the UI is edited in place: make the browser revalidate instead of showing an old copy
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response
engine: Engine | None = None


def _params(body: dict, thinking: bool) -> SamplingParams:
    # Qwen3.5 recommended defaults differ between thinking and direct answers
    d = SamplingParams(temperature=0.6, top_p=0.95, top_k=20) if thinking else \
        SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5)
    return SamplingParams(
        temperature=float(body.get("temperature", d.temperature)),
        top_p=float(body.get("top_p", d.top_p)),
        top_k=int(body.get("top_k", d.top_k)),
        presence_penalty=float(body.get("presence_penalty", d.presence_penalty)),
        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
        seed=body.get("seed"),
    )


def _messages(body: dict) -> list[dict]:
    msgs = body.get("messages")
    if not msgs:
        raise HTTPException(400, "messages is required")
    out = []
    for m in msgs:
        content = m.get("content") or ""
        if isinstance(content, list):  # OpenAI content parts: keep the text ones
            content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
        out.append({"role": m["role"], "content": content})
    return out


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = _messages(body)
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = bool(body.get("enable_thinking", kwargs.get("enable_thinking", True)))
    params = _params(body, thinking)
    budget = body.get("thinking_budget")
    budget = int(budget) if budget else None

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    job = engine.submit(messages, params, thinking,
                        lambda ev: loop.call_soon_threadsafe(events.put_nowait, ev), budget)

    rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = engine.model_name

    def chunk(delta: dict, finish=None, **extra) -> str:
        payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                   "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                   **extra}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    if body.get("stream"):
        async def stream():
            try:
                yield chunk({"role": "assistant"})
                while True:
                    ev = await events.get()
                    t = ev["type"]
                    if t == "content":
                        yield chunk({"content": ev["text"]})
                    elif t == "reasoning":
                        yield chunk({"reasoning_content": ev["text"]})
                    elif t in ("stats", "start"):
                        yield f"event: {t}\ndata: {json.dumps(ev)}\n\n"
                    elif t == "error":
                        yield f"data: {json.dumps({'error': {'message': ev['message']}})}\n\n"
                        break
                    elif t == "done":
                        s = ev["stats"]
                        usage = {"prompt_tokens": s.get("prompt_tokens", 0),
                                 "completion_tokens": s.get("completion_tokens", 0),
                                 "total_tokens": s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)}
                        yield chunk({}, ev["finish_reason"], usage=usage, x_stats=s)
                        break
                yield "data: [DONE]\n\n"
            finally:
                job.cancelled.set()  # client went away or stream ended

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    content, reasoning = [], []
    try:
        while True:
            ev = await events.get()
            if ev["type"] == "content":
                content.append(ev["text"])
            elif ev["type"] == "reasoning":
                reasoning.append(ev["text"])
            elif ev["type"] == "error":
                raise HTTPException(500, ev["message"])
            elif ev["type"] == "done":
                break
    finally:
        job.cancelled.set()
    s = ev["stats"]
    msg = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    return {
        "id": rid, "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": ev["finish_reason"]}],
        "usage": {"prompt_tokens": s.get("prompt_tokens", 0),
                  "completion_tokens": s.get("completion_tokens", 0),
                  "total_tokens": s.get("prompt_tokens", 0) + s.get("completion_tokens", 0)},
        "x_stats": s,
    }


DEFAULT_WORKSPACE = Path.home() / "QVex"


@app.get("/api/agent/workspace")
async def agent_workspace():
    DEFAULT_WORKSPACE.mkdir(exist_ok=True)
    return {"default": str(DEFAULT_WORKSPACE)}


PICK_SCRIPT = """on run argv
  activate
  try
    set d to POSIX file (item 1 of argv) as alias
    return POSIX path of (choose folder with prompt "Folder for the agent to work in" default location d)
  on error number -128
    return ""
  end try
end run"""


@app.post("/api/agent/pick")
async def agent_pick(request: Request):
    """Native macOS folder dialog (the server runs on this Mac, so it can open one in front)."""
    body = await request.json()
    start = Path(body.get("current") or DEFAULT_WORKSPACE).expanduser()
    if not start.is_dir():
        start = DEFAULT_WORKSPACE
    DEFAULT_WORKSPACE.mkdir(exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "osascript", "-e", PICK_SCRIPT, str(start),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(500, err.decode().strip() or "folder dialog not available")
    path = out.decode().strip()
    if not path:
        return {"path": None}  # cancelled
    try:
        return {"path": str(check_root(path))}
    except ToolError as e:
        raise HTTPException(400, str(e))


@app.post("/api/agent")
async def agent_run(request: Request):
    """Agent mode as server-sent events: step, reasoning, content, stats, tool_call, tool_result, done."""
    body = await request.json()
    try:
        root = check_root(body.get("workspace") or str(DEFAULT_WORKSPACE))
    except ToolError as e:
        raise HTTPException(400, str(e))
    budget = body.get("thinking_budget")
    agent = Agent(engine, root, _messages(body), int(budget) if budget else 0,
                  int(body.get("max_steps") or 10))

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'workspace', 'path': str(root)})}\n\n"
            async for ev in agent.run():
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"
        finally:
            agent.cancel()  # client went away or run ended

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/agent/confirm")
async def agent_confirm(request: Request):
    body = await request.json()
    fut = Agent.pending.get(body.get("id"))
    if fut is None or fut.done():
        raise HTTPException(404, "no pending action with this id")
    fut.set_result(bool(body.get("approve")))
    return {"ok": True}


KV_BYTES_PER_TOKEN = 8 * 2 * 4 * 256 * 2  # 8 attention layers x K,V x 4 kv heads x 256 dims x bf16
_ctx_tok: ChatTokenizer | None = None


@app.post("/api/context")
async def context(request: Request):
    """Token count of a conversation as the model would see it, message by message.
    Works with the model unloaded (only the tokenizer is read)."""
    global _ctx_tok
    body = await request.json()
    msgs = [m for m in (body.get("messages") or []) if m.get("content")]
    thinking = bool(body.get("enable_thinking", False))
    tools = None
    if body.get("agent"):
        msgs = [{"role": "system", "content": SYSTEM.format(root=body.get("workspace") or DEFAULT_WORKSPACE)}] + msgs
        tools = TOOLS
    if _ctx_tok is None:
        _ctx_tok = await asyncio.to_thread(ChatTokenizer, engine.model_dir)

    def size(ms, t=tools):
        return len(_ctx_tok.encode(_ctx_tok.render(ms, False, thinking, t)))

    def count():
        parts, prev, first = [], 0, 0
        if tools:
            # the template refuses a system prompt with no user query: measure it as the difference
            q = msgs[1:2] or [{"role": "user", "content": ""}]
            prev = size(msgs[:1] + q) - size(q, None)
            parts.append({"role": "system+tools", "tokens": prev})
            first = 1
            if len(msgs) == 1:
                return parts, prev
        for i in range(first, len(msgs)):
            n = size(msgs[: i + 1])
            parts.append({"role": msgs[i]["role"], "tokens": n - prev})
            prev = n
        total = len(_ctx_tok.encode(_ctx_tok.render(msgs, True, thinking, tools))) if msgs else 0
        return parts, total

    parts, total = await asyncio.to_thread(count)
    cfg = json.loads((Path(engine.model_dir) / "config.json").read_text())
    limit = cfg.get("text_config", cfg).get("max_position_embeddings", 262144)
    return {"total": total, "limit": limit, "parts": parts,
            "kv_mb": total * KV_BYTES_PER_TOKEN / 1e6, "kv_bytes_per_token": KV_BYTES_PER_TOKEN}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": engine.model_name, "object": "model", "owned_by": "local"}]}


@app.get("/api/telemetry")
async def telemetry():
    return engine.telemetry()


@app.get("/api/modes")
async def modes():
    return {"current": engine.governor.mode,
            "modes": [{"id": k, "target_c": v[0], "label": v[1]} for k, v in MODES.items()]}


@app.post("/api/mode")
async def set_mode(request: Request):
    body = await request.json()
    try:
        engine.governor.set_mode(body.get("mode"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"mode": engine.governor.mode, "target_c": engine.governor.target_c}


@app.post("/api/mtp")
async def set_mtp(request: Request):
    body = await request.json()
    try:
        engine.set_mtp_mode(body.get("mode"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"mtp_mode": engine.mtp_mode}


@app.post("/api/unload")
async def unload():
    if engine.busy:
        raise HTTPException(409, "generating: stop it before ejecting the model")
    await asyncio.to_thread(engine.unload)
    return {"model_state": engine.state}


@app.post("/api/load")
async def load():
    try:
        await asyncio.to_thread(engine.load)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return {"model_state": engine.state}


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--mode", default="balanced", choices=list(MODES))
    ap.add_argument("--mtp", default=str(DEFAULT_MTP), help="MTP head weights ('' to disable)")
    ap.add_argument("--mtp-mode", default="auto", choices=MTP_MODES)
    ap.add_argument("--preload", action="store_true",
                    help="load the model at startup (default: wait for 'Load' or the first chat request)")
    args = ap.parse_args()
    t0 = time.perf_counter()
    if args.preload:
        print(f"Loading {args.model} ...", flush=True)
    engine = Engine(args.model, args.mode, args.mtp or None, args.mtp_mode, preload=args.preload)
    if args.preload:
        print(f"MTP: {'loaded, mode ' + engine.mtp_mode if engine.mtp else 'not available'}", flush=True)
    else:
        print(f"Model not loaded: press 'Load' in the web UI ({args.model})", flush=True)
    print(f"Ready in {time.perf_counter() - t0:.1f}s -> http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
