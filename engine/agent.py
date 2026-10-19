"""Agent mode: the model works inside one folder the user picked, through a few small tools.

Reading tools run straight away; anything that changes the disk or runs code waits for the user
to approve it in the UI. Every step is a normal engine job, so the thermal governor, the reasoning
budget and the conversation snapshot all apply: each step only prefills its own new tokens
(the last tool call and its result), not the whole history again.
"""

import asyncio
import re
import subprocess
import sys
import uuid
from pathlib import Path

from .sampler import SamplingParams

MAX_STEPS = 10
RESULT_CHARS = 6000   # tool output given back to the model: every char is prefill, i.e. heat
READ_LINES = 200
TOOL_CALL = "<tool_call>"

SYSTEM = """You are QVex, an assistant running locally on the user's Mac. You work inside the \
folder {root} through the tools below; paths are relative to it. Look before you act: list and \
read files instead of guessing their content. Keep tool calls few and targeted, since every step \
costs time and battery. When you have what you need, answer directly in the user's language, \
briefly, without mentioning the tools. Writing files and running Python need the user's approval; \
if they refuse, do not insist."""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files and folders (two levels deep) with their size.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Folder to list, default '.'"}}}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"Read a text file, {READ_LINES} lines at a time. Lines are numbered.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "description": "First line to read, default 1"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_text",
        "description": "Search text files for a word or regular expression (case-insensitive). "
                       "Returns file:line: text for each match.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Folder or file to search, default '.'"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content (needs approval).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Run a Python 3 script with the folder as working directory and return "
                       "its output (needs approval, 30 s limit).",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}},
            "required": ["code"]}}},
]
NEEDS_APPROVAL = {"write_file", "run_python"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".DS_Store"}


class ToolError(Exception):
    pass


def check_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ToolError(f"folder {root} does not exist")
    if root == Path("/") or root == Path.home():
        raise ToolError("choose a specific working folder, not / or the whole home folder")
    return root


class Tools:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, rel: str | None) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{rel}: outside the working folder")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def list_files(self, path="."):
        base = self._path(path)
        if not base.is_dir():
            raise ToolError(f"{path}: not a folder")
        out = []
        for p in sorted(base.iterdir()):
            if p.name in SKIP_DIRS:
                continue
            if p.is_dir():
                kids = [k for k in sorted(p.iterdir()) if k.name not in SKIP_DIRS]
                out.append(f"{self._rel(p)}/")
                out += [f"  {self._rel(k)}{'/' if k.is_dir() else f'  ({k.stat().st_size} B)'}" for k in kids[:40]]
                if len(kids) > 40:
                    out.append(f"  ... altri {len(kids) - 40}")
            else:
                out.append(f"{self._rel(p)}  ({p.stat().st_size} B)")
            if len(out) > 300:
                out.append("...")
                break
        return "\n".join(out) or "(empty folder)"

    def read_file(self, path, start_line=1):
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"{path}: file not found")
        raw = p.read_bytes()
        if b"\0" in raw[:4096]:
            raise ToolError(f"{path}: binary file")
        lines = raw.decode("utf-8", errors="replace").splitlines()
        start = max(int(start_line or 1), 1)
        chunk = lines[start - 1 : start - 1 + READ_LINES]
        body = "\n".join(f"{start + i:5d}  {l}" for i, l in enumerate(chunk))
        end = start + len(chunk) - 1
        more = f"\n[lines {start}-{end} of {len(lines)}: continue with start_line={end + 1}]" if end < len(lines) else ""
        return (body or "(empty file)") + more

    def search_text(self, pattern, path="."):
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        base = self._path(path)
        files = [base] if base.is_file() else (
            f for f in sorted(base.rglob("*")) if f.is_file() and not SKIP_DIRS & set(f.relative_to(base).parts))
        hits = []
        for f in files:
            if f.stat().st_size > 2_000_000:
                continue
            try:
                text = f.read_text("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self._rel(f)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= 60:
                        return "\n".join(hits) + "\n[prime 60 corrispondenze]"
        return "\n".join(hits) or "no matches"

    def write_file(self, path, content):
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        p.write_text(content, "utf-8")
        return f"{'sovrascritto' if existed else 'creato'} {self._rel(p)} ({len(content.encode())} B)"

    def run_python(self, code):
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=self.root,
                               capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise ToolError("timed out (30 s)")
        out = r.stdout + (f"\n[stderr]\n{r.stderr}" if r.stderr else "")
        return f"[exit code {r.returncode}]\n{out.strip() or '(no output)'}"

    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, name, None) if name in {t["function"]["name"] for t in TOOLS} else None
        if fn is None:
            raise ToolError(f"unknown tool: {name}")
        try:
            return fn(**args)
        except TypeError as e:
            raise ToolError(f"invalid arguments: {e}")


def parse_tool_calls(text: str) -> list[dict]:
    """Qwen3.5 XML format: <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>.
    The closing tags may be missing when generation stopped right after them."""
    calls = []
    for m in re.finditer(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", text.split(TOOL_CALL, 1)[-1] if TOOL_CALL in text else "", re.S):
        args = {}
        for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", m.group(2), re.S):
            v = v.strip("\n") if k != "content" else v
            if k.strip() == "start_line":
                try:
                    v = int(v.strip())
                except ValueError:
                    pass
            args[k.strip()] = v
        calls.append({"name": m.group(1).strip(), "arguments": args})
    return calls


class Agent:
    """One agent run. `pending` holds approval futures the /api/agent/confirm endpoint resolves."""

    pending: dict[str, asyncio.Future] = {}

    def __init__(self, engine, root: Path, messages: list[dict], thinking_budget: int | None,
                 max_steps: int = MAX_STEPS):
        self.engine, self.tools = engine, Tools(root)
        self.messages = [{"role": "system", "content": SYSTEM.format(root=root)}] + \
                        [m for m in messages if m["role"] != "system"]
        self.budget = thinking_budget
        self.max_steps = max_steps
        self.job = None

    def cancel(self):
        if self.job is not None:
            self.job.cancelled.set()
        for fut in Agent.pending.values():
            if not fut.done():
                fut.set_result(False)

    async def _step(self):
        """Run one model turn; yields UI events, returns (content, reasoning, stats, finish) via the last item."""
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        thinking = self.budget != 0
        params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096) if thinking else \
            SamplingParams(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=4096)
        self.job = self.engine.submit(self.messages, params, thinking,
                                      lambda ev: loop.call_soon_threadsafe(events.put_nowait, ev),
                                      self.budget if thinking else None, TOOLS)
        content, reasoning, shown = "", "", 0
        while True:
            ev = await events.get()
            t = ev["type"]
            if t == "reasoning":
                reasoning += ev["text"]
                yield ev
            elif t == "content":
                content += ev["text"]
                # stream the prose, hold back the raw tool call markup (and any prefix of its tag)
                cut = content.find(TOOL_CALL)
                safe = len(content) if cut < 0 else cut
                if cut < 0:
                    for k in range(len(TOOL_CALL) - 1, 0, -1):
                        if content.endswith(TOOL_CALL[:k]):
                            safe = len(content) - k
                            break
                if safe > shown:
                    yield {"type": "content", "text": content[shown:safe]}
                    shown = safe
            elif t == "stats":
                yield ev
            elif t == "error":
                raise RuntimeError(ev["message"])
            elif t == "done":
                yield {"type": "_step_done", "content": content, "reasoning": reasoning,
                       "stats": ev["stats"], "finish": ev["finish_reason"]}
                return

    async def run(self):
        total = {"completion_tokens": 0, "energy_j": 0.0, "prompt_tokens": 0, "cached_tokens": 0,
                 "max_temp_c": 0.0, "reasoning_tokens": 0}
        finish = "max_steps"
        for step in range(1, self.max_steps + 1):
            yield {"type": "step", "n": step}
            done = None
            async for ev in self._step():
                if ev["type"] == "_step_done":
                    done = ev
                else:
                    yield ev
            s = done["stats"]
            for k in ("completion_tokens", "energy_j", "prompt_tokens", "cached_tokens", "reasoning_tokens"):
                total[k] += s.get(k) or 0
            total["max_temp_c"] = max(total["max_temp_c"], s.get("max_temp_c") or 0)
            total["decode_tps"] = s.get("decode_tps")
            if done["finish"] == "cancelled":
                finish = "cancelled"
                break

            content = done["content"]
            calls = parse_tool_calls(content) if done["finish"] == "stop" else []
            prose = content.split(TOOL_CALL, 1)[0].rstrip()
            if not calls:
                finish = done["finish"]
                break

            msg = {"role": "assistant", "content": prose, "tool_calls": [{"function": c} for c in calls]}
            if done["reasoning"]:
                msg["reasoning_content"] = done["reasoning"]
            self.messages.append(msg)
            for c in calls:
                cid = uuid.uuid4().hex[:12]
                needs = c["name"] in NEEDS_APPROVAL
                yield {"type": "tool_call", "id": cid, "name": c["name"], "args": c["arguments"],
                       "needs_approval": needs}
                approved = True
                if needs:
                    fut = asyncio.get_running_loop().create_future()
                    Agent.pending[cid] = fut
                    try:
                        approved = await fut
                    finally:
                        Agent.pending.pop(cid, None)
                if not approved:
                    ok, out = False, "The user denied this action."
                else:
                    try:
                        ok, out = True, await asyncio.to_thread(self.tools.call, c["name"], c["arguments"])
                    except ToolError as e:
                        ok, out = False, f"Errore: {e}"
                    except Exception as e:  # a tool bug must not kill the run
                        ok, out = False, f"Errore: {type(e).__name__}: {e}"
                if len(out) > RESULT_CHARS:
                    out = out[:RESULT_CHARS] + f"\n[... truncated: {len(out) - RESULT_CHARS} more characters]"
                yield {"type": "tool_result", "id": cid, "ok": ok, "approved": approved, "output": out}
                self.messages.append({"role": "tool", "content": out})
            if self.job.cancelled.is_set():
                finish = "cancelled"
                break
        if total["completion_tokens"]:
            total["j_per_token"] = total["energy_j"] / total["completion_tokens"]
        yield {"type": "done", "finish_reason": finish, "steps": step, "stats": total}

