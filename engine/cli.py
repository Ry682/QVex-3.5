"""QVex in the terminal: a client of the QVex server (chat and agent mode).

    ./qvex                          full-screen interface: chat, context and live stats (engine/tui.py)
    ./qvex --plain                  simple line-by-line chat
    ./qvex "question"               one answer, then exit (stdin is attached: cat f.py | ./qvex "explain")
    ./qvex --agent ~/QVex/test      agent mode on a folder
    ./qvex --think                  short reasoning (512 tokens) instead of none

If no server is running on the port, one is started in the background and stopped on exit,
so the model is loaded once and the web UI keeps working at the same address meanwhile.
"""

import argparse
import atexit
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THINK_BUDGET = 512  # same "Short" budget as the web UI (bench/thinking_budget.py)
HISTORY_FILE = Path.home() / ".qvex_history"
TOOL_LABEL = {"list_files": "List files", "read_file": "Read", "search_text": "Search",
              "write_file": "Write", "run_python": "Run Python"}

HELP = """commands:
  /new              start a new conversation
  /agent [folder]   agent mode (works on the files in the folder)
  /chat             back to plain chat
  /think on|off     short reasoning on or off
  /mode eco|balanced|max   thermal mode
  /load  /eject     load or free the model (~5 GB)
  /status           model, temperature, power
  /quit             exit (also Ctrl+D)
Ctrl+C stops the answer. Shift+Enter is not available here: end a line with \\ to continue it."""


# ------------------------------------------------------------------ output

class Style:
    def __init__(self, on: bool):
        c = (lambda code: f"\033[{code}m") if on else (lambda code: "")
        self.reset, self.dim, self.bold = c("0"), c("2"), c("1")
        self.accent = c("38;2;3;165;252")  # #03a5fc, the UI accent
        self.red, self.green = c("31"), c("32")


OUT = Style(sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
ERR = Style(sys.stderr.isatty() and not os.environ.get("NO_COLOR"))


def info(text: str):
    print(f"{ERR.dim}{text}{ERR.reset}", file=sys.stderr, flush=True)


def stats_line(s: dict, extra: list[str] = ()) -> str:
    parts = list(extra)
    if s.get("decode_tps"):
        parts.append(f"{s['decode_tps']:.1f} tok/s")
    if s.get("completion_tokens"):
        parts.append(f"{s['completion_tokens']} tokens")
    if s.get("cached_tokens"):
        parts.append(f"{s['cached_tokens']}/{s['prompt_tokens']} cached")
    if s.get("energy_j"):
        parts.append(f"{s['energy_j']:.0f} J")
    if s.get("max_temp_c"):
        parts.append(f"max {s['max_temp_c']:.0f} °C")
    if (s.get("throttled_s") or 0) > 0.05:
        parts.append(f"throttled {s['throttled_s']:.1f} s")
    return " · ".join(parts)


# ------------------------------------------------------------------ server

class Server:
    def __init__(self, host: str, port: int):
        self.host, self.port, self.proc = host, port, None

    def conn(self, timeout=None):
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def request(self, method: str, path: str, body=None, timeout=30):
        c = self.conn(timeout)
        try:
            c.request(method, path, json.dumps(body) if body is not None else None,
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            data = json.loads(r.read() or b"null")
            if r.status >= 400:
                raise RuntimeError((data or {}).get("detail") or f"HTTP {r.status}")
            return data
        finally:
            c.close()

    def alive(self) -> bool:
        try:
            self.request("GET", "/v1/models", timeout=2)
            return True
        except (OSError, RuntimeError):
            return False

    def ensure(self):
        if self.alive():
            return
        info(f"starting the QVex server on {self.host}:{self.port} …")
        log = open(ROOT / "results/cli-server.log", "w") if (ROOT / "results").is_dir() else subprocess.DEVNULL
        self.proc = subprocess.Popen([sys.executable, "-m", "engine.server", "--host", self.host,
                                      "--port", str(self.port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.stop)
        for _ in range(120):
            if self.proc.poll() is not None:
                sys.exit(f"the server did not start (see results/cli-server.log)")
            if self.alive():
                info(f"web UI meanwhile at http://{self.host}:{self.port} (stops when you exit)")
                return
            time.sleep(0.25)
        sys.exit("the server did not answer in 30 s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def stream(self, path: str, body: dict, on_conn=None):
        """POST and yield (event, data) pairs of a server-sent event stream.
        on_conn gets the connection, so another thread can cut the stream short."""
        c = self.conn()
        if on_conn:
            on_conn(c)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        try:
            if r.status >= 400:
                detail = r.read().decode()
                try:
                    detail = json.loads(detail).get("detail", detail)
                except ValueError:
                    pass
                raise RuntimeError(detail)
            event, data = "message", ""
            for raw in r:
                line = raw.decode().rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()
                elif not line and data:
                    yield event, data
                    event, data = "message", ""
        finally:
            c.close()  # closing early (Ctrl+C) makes the server cancel the generation


# ------------------------------------------------------------------ session

class Session:
    def __init__(self, server: Server, args):
        self.server = server
        self.think = args.think
        self.max_tokens = args.max_tokens
        self.agent_dir = str(Path(args.agent).expanduser().resolve()) if args.agent else None
        self.history: list[dict] = []

    # one streamed answer, shared by chat and agent: reasoning dim, answer plain
    def _printer(self):
        """Streamed output: reasoning dim on stderr, the answer on stdout (so `> file` keeps only the answer)."""
        state = {"mode": None, "fresh": True}

        def emit(kind: str, text: str):
            if kind != state["mode"]:
                if state["mode"] == "reasoning":
                    sys.stderr.write(f"{ERR.reset}\n\n")
                    sys.stderr.flush()
                if kind == "reasoning":
                    sys.stderr.write(ERR.dim)
                state["mode"] = kind
            if kind == "reasoning":
                sys.stderr.write(text)
                sys.stderr.flush()
                return
            if state["fresh"]:
                text = text.lstrip("\n")
                state["fresh"] = not text
            sys.stdout.write(text)
            sys.stdout.flush()

        def end():
            if state["mode"] == "reasoning":
                sys.stderr.write(f"{ERR.reset}\n")
                sys.stderr.flush()
            elif state["mode"]:
                sys.stdout.write("\n")
                sys.stdout.flush()
            state.update(mode=None, fresh=True)
        return emit, end

    def chat(self, text: str):
        self.history.append({"role": "user", "content": text})
        body = {"messages": self.history, "stream": True, "enable_thinking": self.think,
                "thinking_budget": THINK_BUDGET if self.think else None, "max_tokens": self.max_tokens}
        emit, end = self._printer()
        content, stats, finish = "", {}, None
        try:
            for event, data in self.server.stream("/v1/chat/completions", body):
                if event != "message" or data == "[DONE]":
                    continue
                obj = json.loads(data)
                if "error" in obj:
                    raise RuntimeError(obj["error"]["message"])
                ch = obj["choices"][0]
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    emit("reasoning", d["reasoning_content"])
                if d.get("content"):
                    content += d["content"]
                    emit("content", d["content"])
                if ch.get("finish_reason"):
                    finish, stats = ch["finish_reason"], obj.get("x_stats") or {}
        except KeyboardInterrupt:
            finish = "stopped"
        finally:
            end()
        if content.strip():
            self.history.append({"role": "assistant", "content": content.strip()})
        else:
            self.history.pop()
        extra = ["stopped"] if finish == "stopped" else ["cut at max tokens"] if finish == "length" else []
        line = stats_line(stats, extra)
        if line:
            info(line)

    def agent(self, text: str):
        self.history.append({"role": "user", "content": text})
        body = {"messages": self.history, "workspace": self.agent_dir,
                "thinking_budget": THINK_BUDGET if self.think else 0}
        emit, end = self._printer()
        answer, stats, steps, finish = [], {}, 0, None
        try:
            for _, data in self.server.stream("/api/agent", body):
                ev = json.loads(data)
                t = ev["type"]
                if t == "reasoning":
                    emit("reasoning", ev["text"])
                elif t == "content":
                    answer.append(ev["text"])
                    emit("content", ev["text"])
                elif t == "step" and ev["n"] > 1:
                    answer.clear()  # only the last step's prose is the answer kept in history
                elif t == "tool_call":
                    end()
                    self._tool_call(ev)
                elif t == "tool_result":
                    self._tool_result(ev)
                elif t == "error":
                    raise RuntimeError(ev["message"])
                elif t == "done":
                    stats, steps, finish = ev["stats"], ev["steps"], ev["finish_reason"]
        except KeyboardInterrupt:
            finish = "stopped"
        finally:
            end()
        reply = "".join(answer).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        elif finish in (None, "stopped"):
            self.history.pop()
        extra = [f"{steps} step{'s' if steps != 1 else ''}"] if steps else []
        if finish == "stopped":
            extra.append("stopped")
        elif finish == "max_steps":
            extra.append("step limit reached")
        line = stats_line(stats, extra)
        if line:
            info(line)

    def _tool_call(self, ev: dict):
        a, name = ev["args"], ev["name"]
        arg = a.get("pattern") if name == "search_text" else \
            next((l.strip() for l in (a.get("code") or "").splitlines() if l.strip()), "") if name == "run_python" else \
            a.get("path") or "."
        print(f"{OUT.accent}● {TOOL_LABEL.get(name, name)}{OUT.reset} {arg}", flush=True)
        if not ev["needs_approval"]:
            return
        body = a.get("code") if name == "run_python" else a.get("content", "")
        lines = (body or "").splitlines()
        shown = lines[:40]
        for l in shown:
            print(f"  {OUT.dim}│{OUT.reset} {l}")
        if len(lines) > len(shown):
            print(f"  {OUT.dim}│ … {len(lines) - len(shown)} more lines{OUT.reset}")
        try:
            ok = input(f"  {OUT.bold}Allow? [y/N]{OUT.reset} ").strip().lower() in ("y", "yes", "s", "si", "sì")
        except EOFError:
            ok = False
        forget_last_input()  # a y/n answer is not worth keeping in the arrow-up history
        self.server.request("POST", "/api/agent/confirm", {"id": ev["id"], "approve": ok})

    def _tool_result(self, ev: dict):
        if not ev["approved"]:
            print(f"  {OUT.dim}denied{OUT.reset}")
        elif not ev["ok"]:
            print(f"  {OUT.red}{ev['output'].splitlines()[0] if ev['output'] else 'error'}{OUT.reset}")
        else:
            n = len(ev["output"].splitlines())
            print(f"  {OUT.dim}done · {n} line{'s' if n != 1 else ''}{OUT.reset}")
        sys.stdout.flush()

    def ask(self, text: str):
        (self.agent if self.agent_dir else self.chat)(text)


# ------------------------------------------------------------------ interactive

def forget_last_input():
    try:
        import readline
        n = readline.get_current_history_length()
        if n:
            readline.remove_history_item(n - 1)
    except (ImportError, ValueError):
        pass


def command(s: Session, line: str) -> bool:
    """Handle a /command; False means quit."""
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()
    try:
        if cmd in ("quit", "exit", "q"):
            return False
        elif cmd == "help":
            print(HELP)
        elif cmd == "new":
            s.history.clear()
            info("new conversation")
        elif cmd == "agent":
            path = arg or s.agent_dir or s.server.request("GET", "/api/agent/workspace")["default"]
            s.agent_dir = str(Path(path).expanduser().resolve())
            info(f"agent mode · folder {s.agent_dir}")
        elif cmd == "chat":
            s.agent_dir = None
            info("chat mode")
        elif cmd == "think":
            s.think = arg != "off" if arg else not s.think
            info(f"reasoning {'short' if s.think else 'off'}")
        elif cmd == "mode":
            r = s.server.request("POST", "/api/mode", {"mode": arg})
            info(f"thermal mode {r['mode']} · target {r['target_c']:.0f} °C")
        elif cmd in ("load", "eject"):
            info("loading …" if cmd == "load" else "ejecting …")
            r = s.server.request("POST", "/api/load" if cmd == "load" else "/api/unload", {}, timeout=300)
            info(f"model {r['model_state']}")
        elif cmd == "status":
            t = s.server.request("GET", "/api/telemetry")
            temp = max(t.get("gpu_c") or 0, t.get("cpu_c") or 0)
            info(f"{t.get('model_state')} · {t.get('mode')} (target {t.get('target_c', 0):.0f} °C) · "
                 f"{temp:.0f} °C · {t.get('watts') or 0:.1f} W · "
                 f"{'agent ' + s.agent_dir if s.agent_dir else 'chat'} · reasoning {'short' if s.think else 'off'}")
        else:
            info(f"unknown command /{cmd} · /help for the list")
    except RuntimeError as e:
        print(f"{ERR.red}{e}{ERR.reset}", file=sys.stderr)
    return True


def read_prompt(s: Session) -> str:
    where = f"agent:{Path(s.agent_dir).name}" if s.agent_dir else "chat"
    text = input(f"{OUT.accent}{where} ›{OUT.reset} ")
    while text.endswith("\\"):  # line continuation
        text = text[:-1] + "\n" + input(f"{OUT.dim}  …{OUT.reset} ")
    return text.strip()


def interactive(s: Session):
    try:
        import readline
        try:
            readline.read_history_file(HISTORY_FILE)
        except OSError:
            pass
        readline.set_history_length(1000)
        atexit.register(lambda: readline.write_history_file(HISTORY_FILE))
    except ImportError:
        pass
    model = s.server.request("GET", "/v1/models")["data"][0]["id"]
    print(f"{OUT.accent}{OUT.bold}QVex{OUT.reset} · {model} · {OUT.dim}/help for commands, Ctrl+D to exit{OUT.reset}")
    while True:
        try:
            text = read_prompt(s)
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            return
        if not text:
            continue
        if text.startswith("/"):
            if not command(s, text):
                return
            continue
        try:
            s.ask(text)
        except (RuntimeError, OSError) as e:
            print(f"{ERR.red}{e}{ERR.reset}", file=sys.stderr)
        print()


def main():
    ap = argparse.ArgumentParser(prog="qvex", description="QVex in the terminal (chat and agent mode).")
    ap.add_argument("prompt", nargs="*", help="ask once and exit; without it, interactive chat")
    ap.add_argument("--agent", nargs="?", const="", metavar="FOLDER",
                    help="agent mode on FOLDER (default ~/QVex)")
    ap.add_argument("--think", action="store_true", help=f"short reasoning ({THINK_BUDGET} tokens)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--plain", action="store_true", help="simple line mode instead of the full-screen interface")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = Server(args.host, args.port)
    server.ensure()
    if args.agent == "":
        args.agent = server.request("GET", "/api/agent/workspace")["default"]
    s = Session(server, args)

    prompt = " ".join(args.prompt).strip()
    piped = "" if sys.stdin.isatty() else sys.stdin.read()
    if prompt or piped:
        if piped:
            prompt = f"```\n{piped.rstrip()}\n```\n\n{prompt}" if prompt else piped
            if args.agent:  # approvals read the keyboard, stdin is taken by the pipe
                try:
                    sys.stdin = open("/dev/tty")
                except OSError:
                    pass
        try:
            s.ask(prompt)
        except (RuntimeError, OSError) as e:
            sys.exit(f"{ERR.red}{e}{ERR.reset}")
    elif args.plain or not (sys.stdin.isatty() and sys.stdout.isatty()):
        interactive(s)
    else:
        from .tui import run
        run(server, args)


if __name__ == "__main__":
    main()
