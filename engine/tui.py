"""QVex full-screen terminal UI: chat and agent mode, with the context and the machine in view.

    ./qvex                 (interactive terminal; ./qvex --plain for the simple line mode)

Left: the conversation (reasoning folds away when the answer starts, tool calls as cards).
Right: context use message by message, the last answer's numbers, live temperature/power, session totals.
"""

import json
import socket
import threading
import time
from pathlib import Path

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Footer, Input, Markdown, Static

from .cli import THINK_BUDGET, TOOL_LABEL, Server

ACCENT = "#03a5fc"
HOT = "#d2493c"
TEMP_SCALE, WATT_SCALE = 105, 25  # full width of the temperature and power bars
BAR = 37
MODES = ["eco", "balanced", "max"]
CTX_ROWS = 8  # context rows listed before "… N earlier"
HELP = ("/new  /agent [folder]  /chat  /think on|off  /mode eco|balanced|max  /load  /eject  /quit\n"
        "Esc stops the answer. Keys are listed at the bottom.")


def n_(x) -> str:
    return f"{int(x or 0):,}"


def bar(frac: float, width: int = BAR, hot: bool = False, mark: float | None = None) -> Text:
    frac = max(0.0, min(1.0, frac or 0.0))
    full = round(frac * width)
    cells = ["█"] * full + ["░"] * (width - full)
    t = Text()
    m = round(mark * width) if mark is not None else None
    for i, c in enumerate(cells):
        if m is not None and i == min(m, width - 1):
            t.append("│", style="bold")
        else:
            t.append(c, style=(HOT if hot else ACCENT) if c == "█" else "grey35")
    return t


def kv(rows) -> Table:
    t = Table.grid(expand=True)
    t.add_column(style="grey62", no_wrap=True)
    t.add_column(justify="right", no_wrap=True)
    for r in rows:
        t.add_row(*r)
    return t


def section(title: str, *parts) -> Table:
    t = Table.grid(expand=True)
    t.add_row(Text(title.upper(), style=f"bold {ACCENT}"))
    for p in parts:
        t.add_row(p)
    return t


# ------------------------------------------------------------------ conversation widgets

class Turn(Vertical):
    """One assistant turn: reasoning / answer / tool segments in arrival order."""

    def __init__(self):
        super().__init__(classes="turn")
        self.kind, self.seg, self.body, self.text = None, None, None, ""
        self.dirty = False
        self.answer: list[str] = []  # prose of the current step (the agent's last step is the answer)

    def add(self, kind: str, chunk: str):
        if kind != self.kind:
            self.flush()
            if kind == "reasoning":
                self.body = Static("", classes="rtext")
                self.seg = Collapsible(self.body, title="Reasoning", collapsed=False, classes="reasoning")
            else:
                for c in self.query(Collapsible):
                    c.collapsed = True  # answer started: fold the thinking
                self.seg = self.body = Markdown("", classes="answer")
            self.mount(self.seg)
            self.kind, self.text = kind, ""
        if kind == "content":
            if not self.text:
                chunk = chunk.lstrip("\n")
            self.answer.append(chunk)
        self.text += chunk
        self.dirty = True

    def flush(self):
        if not self.dirty or self.body is None:
            return
        self.dirty = False
        if self.kind == "reasoning":
            self.body.update(Text(self.text))
        else:
            self.body.update(self.text)

    def tool(self, ev: dict) -> Static:
        self.flush()
        self.kind = self.seg = self.body = None
        a, name = ev["args"], ev["name"]
        arg = a.get("pattern") if name == "search_text" else \
            next((l.strip() for l in (a.get("code") or "").splitlines() if l.strip()), "") if name == "run_python" else \
            a.get("path") or "."
        card = Static(classes="tool")
        card.asked = ev["needs_approval"]
        card.label = Text.assemble(("● ", ACCENT), (TOOL_LABEL.get(name, name), f"bold {ACCENT}"), "  ", arg or "")
        card.update(Text.assemble(card.label, ("   waiting for approval…" if ev["needs_approval"] else "   …", "grey50")))
        self.mount(card)
        return card

    def meta(self, text: str, error: bool = False):
        self.flush()
        self.mount(Static(Text(text), classes="error" if error else "meta"))


class Approve(ModalScreen[bool]):
    BINDINGS = [Binding("y", "allow", "allow"), Binding("n", "deny", "deny"), Binding("escape", "deny", "deny")]

    def __init__(self, name: str, args: dict):
        super().__init__()
        self.tool_name, self.args = name, args

    def compose(self) -> ComposeResult:
        py = self.tool_name == "run_python"
        body = self.args.get("code" if py else "content") or ""
        with Vertical(id="dialog"):
            yield Static(Text.assemble(("Allow  ", "bold"), (TOOL_LABEL.get(self.tool_name, self.tool_name), f"bold {ACCENT}"),
                                       "  ", "" if py else self.args.get("path", "")))
            with VerticalScroll(id="preview"):
                yield Static(Syntax(body, "python", theme="ansi_dark", line_numbers=True, word_wrap=True)
                             if py else Text(body or "(empty)"))
            with Horizontal(id="buttons"):
                yield Button("Allow  y", variant="primary", id="allow")
                yield Button("Deny  n", id="deny")

    def on_button_pressed(self, e: Button.Pressed):
        self.dismiss(e.button.id == "allow")

    def action_allow(self):
        self.dismiss(True)

    def action_deny(self):
        self.dismiss(False)


# ------------------------------------------------------------------ app

class QVexApp(App):
    TITLE = "QVex"
    CSS = f"""
    Screen {{ layout: vertical; }}
    #top {{ height: 1; padding: 0 1; background: $panel; }}
    #main {{ height: 1fr; }}
    #log {{ width: 1fr; padding: 0 1 1 1; }}
    #side {{ width: 40; border-left: solid $panel-lighten-2; padding: 0 1; }}
    #side > Static {{ margin-bottom: 1; }}
    #input {{ margin: 0 1; border: tall $panel-lighten-2; }}
    #input:focus {{ border: tall {ACCENT}; }}
    .user {{ background: $boost; padding: 0 1; margin: 1 0 0 6; }}
    .turn {{ height: auto; margin-top: 1; }}
    .reasoning {{ border: none; padding: 0; background: transparent; }}
    .reasoning CollapsibleTitle {{ color: $text-muted; padding: 0; }}
    .rtext {{ color: $text-muted; padding-left: 2; }}
    .answer {{ margin: 0; padding: 0; }}
    .tool {{ margin: 0 0 1 0; }}
    .meta {{ color: $text-muted; }}
    .note {{ color: $text-muted; margin-top: 1; }}
    .error {{ color: {HOT}; }}
    Approve {{ align: center middle; }}
    #dialog {{ width: 90%; max-width: 110; height: auto; max-height: 85%; border: thick {ACCENT}; background: $surface; padding: 1 2; }}
    #preview {{ height: auto; max-height: 24; margin: 1 0; background: $boost; }}
    #buttons {{ height: auto; align-horizontal: right; }}
    #buttons Button {{ margin-left: 2; }}
    """
    BINDINGS = [
        Binding("escape", "stop", "stop"),
        Binding("ctrl+n", "new_chat", "new chat", priority=True),
        Binding("ctrl+t", "toggle_think", "reasoning", priority=True),
        Binding("ctrl+g", "toggle_agent", "agent", priority=True),
        Binding("ctrl+o", "cycle_mode", "heat", priority=True),
        Binding("ctrl+l", "load_eject", "load/eject", priority=True),
        Binding("ctrl+q", "quit", "quit", priority=True),
    ]

    def __init__(self, server: Server, args):
        super().__init__()
        self.server = server
        self.think = args.think
        self.max_tokens = args.max_tokens
        self.agent_dir = str(Path(args.agent).expanduser().resolve()) if args.agent else None
        self.default_dir = None
        self.history: list[dict] = []
        self.busy, self.cancelled, self.conn, self.turn = False, False, None, None
        self.model = "—"
        self.tele: dict = {}
        self.ctx: dict = {}
        self.live: dict = {}
        self.last: dict = {}
        self.totals = {"answers": 0, "tokens": 0, "energy_j": 0.0}
        self.cards: dict[str, Static] = {}  # tool call id -> its card, until the result arrives

    def compose(self) -> ComposeResult:
        yield Static(id="top")
        with Horizontal(id="main"):
            yield VerticalScroll(id="log")
            with VerticalScroll(id="side"):
                yield Static(id="ctx")
                yield Static(id="last")
                yield Static(id="machine")
                yield Static(id="session")
        yield Input(id="input")
        yield Footer()

    def on_mount(self):
        self.log_view = self.query_one("#log", VerticalScroll)
        self.note(f"Everything runs on this Mac. {HELP}")
        self.set_interval(1.0, self.poll)
        self.set_interval(0.12, self.tick)
        self.start()
        self.show_mode()

    @work(thread=True)
    def start(self):
        try:
            self.model = self.server.request("GET", "/v1/models")["data"][0]["id"]
            self.default_dir = self.server.request("GET", "/api/agent/workspace")["default"]
        except (OSError, RuntimeError):
            pass
        self.call_from_thread(self.show_mode)
        self.refresh_context()

    # ---------------------------------------------------------- rendering

    def note(self, text: str, error: bool = False):
        self.log_view.mount(Static(Text(text), classes="error" if error else "note"))
        self.log_view.scroll_end(animate=False)

    def show_mode(self):
        inp = self.query_one("#input", Input)
        inp.placeholder = ("Ask the agent to work on the files in the folder. Enter to send."
                           if self.agent_dir else "Type here. Enter to send, /help for commands.")
        self.render_top()
        self.render_ctx()

    def render_top(self):
        st = self.tele.get("model_state")
        state = {"loaded": ("● loaded", "green"), "loading": ("● loading…", ACCENT), "unloading": ("● ejecting…", ACCENT),
                 "unloaded": ("○ not loaded · the first message loads it", "grey50")}.get(st, ("○ server unreachable", HOT))
        if st == "loaded" and self.tele.get("busy"):
            state = ("● generating", ACCENT)
        where = f"agent · {Path(self.agent_dir).name}" if self.agent_dir else "chat"
        left = Text.assemble((" QVex ", f"bold black on {ACCENT}"), "  ", (self.model, "bold"), "  ", state)
        right = Text.assemble((where, "bold"), ("  ·  reasoning ", "grey62"), ("short" if self.think else "off", "bold"))
        t = Table.grid(expand=True)
        t.add_column()
        t.add_column(justify="right")
        t.add_row(left, right)
        self.query_one("#top", Static).update(t)

    def render_ctx(self):
        c, lv = self.ctx, self.live
        limit = c.get("limit") or 262144
        parts = c.get("parts") or []
        base = c.get("total") or 0
        rows = []
        live_total = None
        if self.busy and lv.get("prompt"):
            live_total = lv["prompt"] + (lv.get("tokens") or 0)
        total = live_total or base
        shown = parts[-CTX_ROWS:]
        if len(parts) > len(shown):
            hidden = sum(p["tokens"] for p in parts[: len(parts) - len(shown)])
            rows.append((f"… {len(parts) - len(shown)} earlier", n_(hidden)))
        names = {"user": "you", "assistant": "qvex", "system": "system", "tool": "tool"}
        for p in shown:
            rows.append((names.get(p["role"], p["role"]), n_(p["tokens"])))
        if live_total:
            rows.append(("this turn (live)", n_(live_total - base)))
        cached = self.tele.get("context_cached") or 0
        per = c.get("kv_bytes_per_token") or 32768
        extra = kv([("kept in engine memory", f"{n_(cached)} tok"),
                    ("KV cache", f"≈ {total * per / 1e6:,.0f} MB"),
                    ("agent folder" if self.agent_dir else "mode", Path(self.agent_dir).name if self.agent_dir else "chat")])
        head = Text.assemble((n_(total), "bold"), (f" / {n_(limit)} tokens", "grey62"),
                             (f"   {100 * total / limit:.1f}%", "grey62"))
        body = [head, bar(total / limit), kv(rows) if rows else Text("empty conversation", style="grey50"), extra]
        self.query_one("#ctx", Static).update(section("Context", *body))

    def render_last(self):
        lv, s = self.live, self.last
        if self.busy:
            el = time.perf_counter() - lv.get("t0", time.perf_counter())
            ttft = (lv["tfirst"] - lv["t0"]) if lv.get("tfirst") else None
            rows = [("elapsed", f"{el:.1f} s"),
                    ("first token", f"{ttft:.2f} s" if ttft is not None else "…"),
                    ("speed", f"{lv['tps']:.1f} tok/s" if lv.get("tps") else "…"),
                    ("tokens", n_(lv.get("tokens"))),
                    ("prompt", f"{n_(lv.get('prompt'))} ({n_(lv.get('cached'))} cached)" if lv.get("prompt") else "…")]
            if lv.get("step"):
                rows.insert(0, ("agent step", str(lv["step"])))
            self.query_one("#last", Static).update(section("Answer · live", kv(rows)))
            return
        if not s:
            self.query_one("#last", Static).update(section("Last answer", Text("—", style="grey50")))
            return
        read = None
        new = (s.get("prompt_tokens") or 0) - (s.get("cached_tokens") or 0)
        if s.get("ttft_s") and new >= 32:
            read = f"{new / s['ttft_s']:,.0f} tok/s"
        rows = []
        if s.get("steps"):
            rows.append(("agent steps", str(s["steps"])))
        if s.get("ttft_s") is not None:
            rows.append(("first token", f"{s['ttft_s']:.2f} s"))
        rows += [("speed", f"{s['decode_tps']:.1f} tok/s" if s.get("decode_tps") else "—"),
                 ("tokens", n_(s.get("completion_tokens")) +
                  (f" ({n_(s['reasoning_tokens'])} reasoning)" if s.get("reasoning_tokens") else "")),
                 ("prompt", f"{n_(s.get('prompt_tokens'))} ({n_(s.get('cached_tokens'))} cached)")]
        if read:
            rows.append(("prompt reading", read))
        rows += [("energy", f"{s.get('energy_j') or 0:,.0f} J · {s.get('j_per_token') or 0:.2f} J/tok"),
                 ("max temperature", f"{s.get('max_temp_c') or 0:.0f} °C")]
        if (s.get("throttled_s") or 0) > 0.05:
            rows.append(("throttled", f"{s['throttled_s']:.1f} s"))
        if s.get("mtp_cycles"):
            rows.append(("turbo accepted", f"{100 * s['mtp_accepted'] / s['mtp_cycles']:.0f}%"))
        if s.get("reasoning_cut"):
            rows.append(("reasoning", "cut at budget"))
        if s.get("finish") not in (None, "stop"):
            rows.append(("ended", {"length": "max tokens", "max_steps": "step limit"}.get(s["finish"], s["finish"])))
        self.query_one("#last", Static).update(section("Last answer", kv(rows)))

    def render_machine(self):
        t = self.tele
        if not t:
            self.query_one("#machine", Static).update(section("Machine", Text("server unreachable", style=HOT)))
            return
        temp = max(t.get("gpu_c") or 0, t.get("cpu_c") or 0)
        target = t.get("target_c") or 80
        w = t.get("watts") or 0
        ram_u, ram_t = t.get("ram_used_gb") or 0, t.get("ram_total_gb") or 16
        rows = Table.grid(expand=True)
        rows.add_column(style="grey62", no_wrap=True, width=12)
        rows.add_column(no_wrap=True)
        rows.add_column(justify="right", no_wrap=True)
        rows.add_row("temperature", bar(temp / TEMP_SCALE, 14, temp > target, target / TEMP_SCALE), f"{temp:.0f} °C")
        rows.add_row("power", bar(w / WATT_SCALE, 14), f"{w:.1f} W")
        rows.add_row("GPU", bar((t.get("gpu_pct") or 0) / 100, 14), f"{t.get('gpu_pct') or 0:.0f}%")
        rows.add_row("CPU", bar((t.get("cpu_pct") or 0) / 100, 14), f"{t.get('cpu_pct') or 0:.0f}%")
        rows.add_row("memory", bar(ram_u / ram_t, 14, ram_u / ram_t > 0.92), f"{ram_u:.1f}/{ram_t:.0f} GB")
        label = {"eco": "Eco", "balanced": "Balanced", "max": "Max"}.get(t.get("mode"), t.get("mode"))
        info = kv([("heat mode", f"{label} · {target:.0f} °C"),
                   ("turbo (MTP)", t.get("mtp_mode", "—") if t.get("mtp_available") else "not installed"),
                   ("speed cap", f"{t['tps_cap']:.0f} tok/s" if t.get("tps_cap") else "none")])
        self.query_one("#machine", Static).update(section("Machine · live", rows, info))

    def render_session(self):
        s = self.totals
        jpt = s["energy_j"] / s["tokens"] if s["tokens"] else 0
        self.query_one("#session", Static).update(section("Session", kv([
            ("answers", str(s["answers"])), ("tokens written", n_(s["tokens"])),
            ("energy", f"{s['energy_j']:,.0f} J" + (f" · {jpt:.2f} J/tok" if jpt else ""))])))

    def tick(self):
        if self.turn is not None:
            near = self.log_view.scroll_y >= self.log_view.max_scroll_y - 3
            self.turn.flush()
            if near:
                self.log_view.scroll_end(animate=False)
        if self.busy:
            self.render_last()
            self.render_ctx()

    # ---------------------------------------------------------- server polling

    @work(thread=True, exclusive=True, group="poll")
    def poll(self):
        try:
            t = self.server.request("GET", "/api/telemetry", timeout=3)
        except (OSError, RuntimeError):
            t = {}
        self.call_from_thread(self.set_tele, t)

    def set_tele(self, t: dict):
        self.tele = t
        self.render_top()
        self.render_machine()
        if not self.busy:
            self.render_ctx()
            self.render_last()
            self.render_session()

    @work(thread=True, exclusive=True, group="ctx")
    def refresh_context(self):
        body = {"messages": self.history, "enable_thinking": self.think,
                "agent": bool(self.agent_dir), "workspace": self.agent_dir}
        try:
            c = self.server.request("POST", "/api/context", body)
        except (OSError, RuntimeError):
            return
        self.call_from_thread(self.set_ctx, c)

    def set_ctx(self, c: dict):
        self.ctx = c
        self.render_ctx()

    # ---------------------------------------------------------- input

    def on_input_submitted(self, e: Input.Submitted):
        text = e.value.strip()
        if not text:
            return
        if text.startswith("/"):
            e.input.value = ""
            self.command(text)
            return
        if self.busy:
            self.notify("Still answering: Esc stops it.", severity="warning")
            return
        e.input.value = ""
        self.send(text)

    def command(self, line: str):
        cmd, _, arg = line[1:].partition(" ")
        arg = arg.strip()
        if cmd in ("quit", "exit", "q"):
            self.exit()
        elif cmd == "help":
            self.note(HELP)
        elif cmd == "new":
            self.action_new_chat()
        elif cmd == "agent":
            self.set_agent(arg or self.agent_dir or self.default_dir)
        elif cmd == "chat":
            self.set_agent(None)
        elif cmd == "think":
            self.think = arg != "off" if arg else not self.think
            self.show_mode()
            self.refresh_context()
        elif cmd == "mode":
            self.set_mode(arg)
        elif cmd in ("load", "eject"):
            self.load_eject(cmd)
        else:
            self.note(f"unknown command /{cmd} · /help for the list", error=True)

    def set_agent(self, path):
        if self.busy:
            return self.notify("Wait for the answer to finish.", severity="warning")
        if path:
            p = Path(path).expanduser().resolve()
            if not p.is_dir():
                return self.note(f"folder {p} does not exist", error=True)
            self.agent_dir = str(p)
            self.note(f"agent mode · folder {p}")
        else:
            self.agent_dir = None
            self.note("chat mode")
        self.show_mode()
        self.refresh_context()

    @work(thread=True)
    def set_mode(self, mode: str):
        try:
            r = self.server.request("POST", "/api/mode", {"mode": mode})
            self.call_from_thread(self.notify, f"Heat mode {r['mode']} · target {r['target_c']:.0f} °C")
        except (OSError, RuntimeError) as e:
            self.call_from_thread(self.note, str(e), True)
        self.poll()

    @work(thread=True, exclusive=True, group="load")
    def load_eject(self, what: str):
        try:
            self.server.request("POST", "/api/load" if what == "load" else "/api/unload", {}, timeout=300)
        except (OSError, RuntimeError) as e:
            self.call_from_thread(self.note, str(e), True)
        self.poll()

    # ---------------------------------------------------------- actions

    def action_new_chat(self):
        if self.busy:
            self.action_stop()
        self.history.clear()
        self.log_view.remove_children()
        self.turn, self.last = None, {}
        self.note("new conversation")
        self.refresh_context()

    def action_toggle_think(self):
        self.command("/think")

    def action_toggle_agent(self):
        self.set_agent(None if self.agent_dir else (self.default_dir or str(Path.home() / "QVex")))

    def action_cycle_mode(self):
        cur = self.tele.get("mode", "balanced")
        self.set_mode(MODES[(MODES.index(cur) + 1) % len(MODES)] if cur in MODES else "balanced")

    def action_load_eject(self):
        if self.busy:
            return self.notify("Stop the answer before ejecting.", severity="warning")
        self.load_eject("load" if self.tele.get("model_state") == "unloaded" else "eject")

    def action_stop(self):
        if not self.busy:
            return
        self.cancelled = True
        c = self.conn
        if c is not None and c.sock is not None:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)  # unblocks the reading thread; the server cancels the job
            except OSError:
                pass

    # ---------------------------------------------------------- one turn

    def send(self, text: str):
        self.history.append({"role": "user", "content": text})
        self.log_view.mount(Static(Text(text), classes="user"))
        self.turn = Turn()
        self.log_view.mount(self.turn)
        self.log_view.scroll_end(animate=False)
        self.busy, self.cancelled = True, False
        self.live = {"t0": time.perf_counter()}
        self.run_turn(list(self.history), self.agent_dir, self.think)

    def _set_conn(self, c):
        self.conn = c

    @work(thread=True, exclusive=True, group="turn")
    def run_turn(self, messages: list[dict], agent_dir: str | None, think: bool):
        out = {"stats": {}, "finish": None, "steps": 0, "error": None}
        ui = self.call_from_thread
        try:
            if agent_dir:
                body = {"messages": messages, "workspace": agent_dir, "thinking_budget": THINK_BUDGET if think else 0}
                for _, data in self.server.stream("/api/agent", body, self._set_conn):
                    ev = json.loads(data)
                    t = ev["type"]
                    if t in ("reasoning", "content"):
                        ui(self.on_text, t, ev["text"])
                    elif t == "start":
                        ui(self.on_start, ev)
                    elif t == "stats":
                        ui(self.on_live, ev)
                    elif t == "step":
                        ui(self.on_step, ev["n"])
                    elif t == "tool_call":
                        self.tool_call(ev)
                    elif t == "tool_result":
                        ui(self.on_tool_result, ev)
                    elif t == "error":
                        raise RuntimeError(ev["message"])
                    elif t == "done":
                        out.update(stats=ev["stats"], finish=ev["finish_reason"], steps=ev["steps"])
            else:
                body = {"messages": messages, "stream": True, "enable_thinking": think,
                        "thinking_budget": THINK_BUDGET if think else None, "max_tokens": self.max_tokens}
                for event, data in self.server.stream("/v1/chat/completions", body, self._set_conn):
                    if event == "start":
                        ui(self.on_start, json.loads(data))
                    elif event == "stats":
                        ui(self.on_live, json.loads(data))
                    elif data != "[DONE]":
                        obj = json.loads(data)
                        if "error" in obj:
                            raise RuntimeError(obj["error"]["message"])
                        ch = obj["choices"][0]
                        d = ch.get("delta", {})
                        if d.get("reasoning_content"):
                            ui(self.on_text, "reasoning", d["reasoning_content"])
                        if d.get("content"):
                            ui(self.on_text, "content", d["content"])
                        if ch.get("finish_reason"):
                            out.update(stats=obj.get("x_stats") or {}, finish=ch["finish_reason"])
        except Exception as e:
            if self.cancelled:
                out["finish"] = "stopped"
            else:
                out["error"] = str(e) or type(e).__name__
        finally:
            self.conn = None
            ui(self.turn_done, out)

    def on_start(self, ev: dict):
        self.live.update(prompt=ev.get("prompt_tokens"), cached=ev.get("cached_tokens"), tokens=0)

    def on_live(self, ev: dict):
        self.live.update(tokens=ev.get("tokens"), tps=ev.get("tps"))

    def on_step(self, n: int):
        self.live["step"] = n
        if n > 1 and self.turn:
            self.turn.answer.clear()

    def on_text(self, kind: str, text: str):
        if not self.live.get("tfirst"):
            self.live["tfirst"] = time.perf_counter()
        self.turn.add(kind, text)

    def tool_call(self, ev: dict):
        """Runs on the turn thread: show the card, ask in a dialog if needed, answer the server."""
        self.cards[ev["id"]] = self.call_from_thread(self.turn.tool, ev)
        if not ev["needs_approval"]:
            return
        done, answer = threading.Event(), {}

        def ask():
            def got(ok):
                answer["ok"] = bool(ok)
                done.set()
            self.push_screen(Approve(ev["name"], ev["args"]), got)
        self.call_from_thread(ask)
        while not done.wait(0.2):
            if self.cancelled:
                self.call_from_thread(self.pop_screen)
                return
        self.server.request("POST", "/api/agent/confirm", {"id": ev["id"], "approve": answer["ok"]})

    def on_tool_result(self, ev: dict):
        card = self.cards.pop(ev["id"], None)
        if card is None:
            return
        if not ev["approved"]:
            tail = ("   denied", "grey50")
        elif not ev["ok"]:
            tail = ("   " + (ev["output"].splitlines() or ["error"])[0], HOT)
        else:
            n = len(ev["output"].splitlines())
            tail = (f"   {'allowed · ' if card.asked else ''}done · {n} line{'s' if n != 1 else ''}", "grey50")
        card.update(Text.assemble(card.label, tail))

    def turn_done(self, out: dict):
        turn, s = self.turn, out["stats"] or {}
        turn.flush()
        reply = "".join(turn.answer).strip()
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        elif out["finish"] in (None, "stopped") or out["error"]:
            self.history.pop()  # nothing came back: don't keep an empty turn
        if out["error"]:
            turn.meta(out["error"], error=True)
        elif out["finish"] == "stopped":
            turn.meta("stopped")
        if s:
            self.last = dict(s, finish=out["finish"], steps=out["steps"])
            self.totals["answers"] += 1
            self.totals["tokens"] += s.get("completion_tokens") or 0
            self.totals["energy_j"] += s.get("energy_j") or 0
        self.busy = False
        self.live = {}
        self.render_last()
        self.render_session()
        self.refresh_context()
        self.poll()
        self.query_one("#input", Input).focus()


def run(server: Server, args):
    QVexApp(server, args).run()
