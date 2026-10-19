"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"), empty: $("empty"), emptyTitle: $("emptyTitle"), input: $("input"), form: $("composer"),
  send: $("sendBtn"), stop: $("stopBtn"), thinking: $("thinking"), maxTok: $("maxTok"), newChat: $("newChat"),
  dot: $("statusDot"), model: $("modelName"), modelState: $("modelState"), eject: $("ejectBtn"), modeSel: $("modeSel"), mtpPick: $("mtpPick"), mtpSel: $("mtpSel"),
  attachBtn: $("attachBtn"), fileInput: $("fileInput"), attached: $("attached"),
  envSeg: $("envSeg"), folderWrap: $("folderWrap"), folderBtn: $("folderBtn"), folderName: $("folderName"),
  folderMenu: $("folderMenu"), envHint: $("envHint"),
  mTtft: $("mTtft"), mTps: $("mTps"), mPre: $("mPre"), mTok: $("mTok"), mDur: $("mDur"),
  gpuVal: $("gpuVal"), gpuBar: $("gpuBar"), cpuVal: $("cpuVal"), cpuBar: $("cpuBar"),
  memVal: $("memVal"), memBar: $("memBar"), tempVal: $("tempVal"), tempBar: $("tempBar"),
  targetMark: $("targetMark"), wattVal: $("wattVal"), wattBar: $("wattBar"),
};

const STORE_KEY = "engine.chat.v1";
const TEMP_SCALE = 105, WATT_SCALE = 25;  // full-width of the temperature and power bars
let history = load();
let controller = null;
let attachments = [];  // [{name, text}] waiting for the next message

// ------------------------------------------------------------------ storage

function load() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || []; } catch { return []; }
}
function save() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(history)); } catch {}
}
function pref(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {}
  return null;
}

// ------------------------------------------------------------------ markdown (small, safe)

function esc(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function markdown(src) {
  const out = [];
  const parts = src.split(/```/);
  parts.forEach((part, i) => {
    if (i % 2 === 1) {
      const nl = part.indexOf("\n");
      const code = nl >= 0 ? part.slice(nl + 1) : part;
      out.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
      return;
    }
    let list = null;
    const flush = () => { if (list) { out.push(`</${list}>`); list = null; } };
    let para = [];
    const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; } };
    for (const line of part.split("\n")) {
      let m;
      if (!line.trim()) { flushPara(); flush(); continue; }
      if ((m = line.match(/^(#{1,3})\s+(.*)/))) { flushPara(); flush(); out.push(`<h3>${inline(m[2])}</h3>`); continue; }
      if ((m = line.match(/^\s*[-*]\s+(.*)/))) {
        flushPara(); if (list !== "ul") { flush(); out.push("<ul>"); list = "ul"; }
        out.push(`<li>${inline(m[1])}</li>`); continue;
      }
      if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) {
        flushPara(); if (list !== "ol") { flush(); out.push("<ol>"); list = "ol"; }
        out.push(`<li>${inline(m[1])}</li>`); continue;
      }
      flush(); para.push(line);
    }
    flushPara(); flush();
  });
  return out.join("");
}

// ------------------------------------------------------------------ number formatting

function num(n, d = 1) {
  return Number(n).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}
function big(el, value, unit) {
  if (value == null) { el.textContent = "—"; return; }
  el.innerHTML = unit ? `${esc(String(value))}<small>${esc(unit)}</small>` : esc(String(value));
}
function setBar(bar, frac, hot = false) {
  bar.style.width = `${Math.max(0, Math.min(1, frac || 0)) * 100}%`;
  bar.classList.toggle("hot", hot);
}

// ------------------------------------------------------------------ metrics board

const run = { t0: 0, tFirst: 0, prompt: 0, cached: 0, tokens: 0, tps: null, timer: null };

function boardStart() {
  Object.assign(run, { t0: performance.now(), tFirst: 0, prompt: 0, cached: 0, tokens: 0, tps: null });
  big(els.mTtft, "…"); big(els.mTps, "…"); big(els.mPre, "…"); big(els.mTok, 0); big(els.mDur, "0.0", "s");
  clearInterval(run.timer);
  run.timer = setInterval(() => big(els.mDur, num((performance.now() - run.t0) / 1000), "s"), 100);
}

function boardFirstToken() {
  if (run.tFirst) return;
  run.tFirst = performance.now();
  const ttft = (run.tFirst - run.t0) / 1000;
  big(els.mTtft, num(ttft, 2), "s");
  showPrefill(run.prompt - run.cached, ttft);
}

function showPrefill(processed, seconds) {
  // a few tokens read in a few ms say nothing about reading speed
  if (processed < 32 || !seconds) big(els.mPre, "cached");
  else big(els.mPre, num(processed / seconds, 0), "tok/s");
}

function boardLive(s) {
  if (s.tps) big(els.mTps, num(s.tps), "tok/s");
  if (s.tokens) big(els.mTok, s.tokens.toLocaleString("en-US"));
}

function boardDone(s) {
  clearInterval(run.timer);
  big(els.mDur, num((performance.now() - run.t0) / 1000), "s");
  if (!s) return;
  if (s.ttft_s != null) big(els.mTtft, num(s.ttft_s, 2), "s");
  big(els.mTps, s.decode_tps ? num(s.decode_tps) : null, "tok/s");
  big(els.mTok, (s.completion_tokens || 0).toLocaleString("en-US"));
  showPrefill((s.prompt_tokens || 0) - (s.cached_tokens || 0), s.ttft_s);
}

// ------------------------------------------------------------------ rendering

function metaText(s) {
  if (!s) return [];
  const parts = [];
  if (s.decode_tps) parts.push(`${num(s.decode_tps)} tok/s`);
  if (s.completion_tokens) parts.push(`${s.completion_tokens} tokens`);
  if (s.ttft_s != null) parts.push(`first token ${num(s.ttft_s, 2)} s`);
  if (s.cached_tokens) parts.push(`${s.cached_tokens}/${s.prompt_tokens} cached`);
  if (s.j_per_token) parts.push(`${num(s.j_per_token, 2)} J/token`);
  if (s.max_temp_c) parts.push(`max ${num(s.max_temp_c, 0)} °C`);
  if (s.throttled_s > 0.05) parts.push(`throttled ${num(s.throttled_s)} s`);
  if (s.mtp_cycles) parts.push(`turbo ${Math.round(100 * s.mtp_accepted / s.mtp_cycles)}% accepted`);
  return parts;
}

function fillMeta(meta, stats) {
  meta.innerHTML = "";
  metaText(stats).forEach((t) => { const s = document.createElement("span"); s.textContent = t; meta.appendChild(s); });
}

function renderMessage(m) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${m.role}`;
  if (m.role === "user" && m.files && m.files.length) {
    const chips = document.createElement("div");
    chips.className = "chips";
    m.files.forEach((f) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = f; chips.appendChild(c); });
    wrap.appendChild(chips);
  }
  if (m.role === "assistant" && m.steps) {
    wrap.appendChild(agentFlow(m));
    const note = agentNote(m);
    if (note) wrap.appendChild(note);
  } else {
    if (m.role === "assistant") {
      const det = document.createElement("details");
      det.className = "reasoning";
      det.innerHTML = `<summary>Reasoning</summary><div class="rtext"></div>`;
      det.hidden = !m.reasoning;
      det.querySelector(".rtext").textContent = m.reasoning || "";
      wrap.appendChild(det);
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    if (m.role === "user") { bubble.textContent = m.display ?? m.content; bubble.hidden = !bubble.textContent; }
    else bubble.innerHTML = markdown(m.content || "");
    wrap.appendChild(bubble);
  }
  if (m.role === "assistant") {
    const meta = document.createElement("div");
    meta.className = "meta";
    if (m.steps) fillAgentMeta(meta, m); else fillMeta(meta, m.stats);
    wrap.appendChild(meta);
  }
  els.messages.appendChild(wrap);
  return wrap;
}

function renderAll() {
  els.messages.querySelectorAll(".msg").forEach((n) => n.remove());
  els.empty.hidden = history.length > 0;
  history.forEach(renderMessage);
  scrollDown(true);
}

function scrollDown(force) {
  const el = els.messages;
  const near = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  if (force || near) el.scrollTop = el.scrollHeight;
}

// ------------------------------------------------------------------ attachments

function renderAttachments() {
  els.attached.innerHTML = "";
  els.attached.hidden = attachments.length === 0;
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.title = "Remove";
    c.textContent = `${a.name} · ${Math.ceil(a.text.length / 1000)}k chars ✕`;
    c.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    els.attached.appendChild(c);
  });
}

els.attachBtn.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", async () => {
  for (const f of els.fileInput.files) {
    if (f.size > 2_000_000) { alert(`${f.name} is too large (max 2 MB).`); continue; }
    attachments.push({ name: f.name, text: await f.text() });
  }
  els.fileInput.value = "";
  renderAttachments();
  els.input.focus();
});

function withAttachments(text) {
  if (!attachments.length) return text;
  const files = attachments.map((a) => `File: ${a.name}\n\`\`\`\n${a.text.replace(/\n$/, "")}\n\`\`\``).join("\n\n");
  return text ? `${files}\n\n${text}` : files;
}

// ------------------------------------------------------------------ chat

// push the user message and an empty assistant turn, render both, return the assistant node
function addTurn(text, msg) {
  const user = { role: "user", content: withAttachments(text), display: text, files: attachments.map((a) => a.name) };
  attachments = [];
  renderAttachments();
  history.push(user, msg);
  els.empty.hidden = true;
  renderMessage(user);
  return renderMessage(msg);
}

// server-sent events from a fetch response, as {event, obj}
async function* sse(res) {
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data || data === "[DONE]") continue;
      yield { event, obj: JSON.parse(data) };
    }
  }
}

async function send(text) {
  const msg = { role: "assistant", content: "", reasoning: "", stats: null };
  const node = addTurn(text, msg);
  const bubble = node.querySelector(".bubble");
  const det = node.querySelector("details.reasoning");
  const rtext = det.querySelector(".rtext");
  const meta = node.querySelector(".meta");
  bubble.classList.add("cursor");
  scrollDown(true);
  setBusy(true);
  boardStart();

  const budget = Number(els.thinking.value);
  const thinking = budget !== 0;
  if (thinking) { det.hidden = false; det.open = true; }

  controller = new AbortController();
  let renderPending = false;
  const scheduleRender = () => {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      bubble.innerHTML = markdown(msg.content);
      rtext.textContent = msg.reasoning;
      scrollDown(false);
    });
  };

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: history.slice(0, -1).map(({ role, content }) => ({ role, content })),
        stream: true,
        enable_thinking: thinking,
        thinking_budget: budget > 0 ? budget : null,
        max_tokens: Number(els.maxTok.value),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);

    for await (const { event, obj } of sse(res)) {
      if (event === "stats") { boardLive(obj); continue; }
      if (event === "start") { run.prompt = obj.prompt_tokens; run.cached = obj.cached_tokens; continue; }
      if (obj.error) throw new Error(obj.error.message);
      const ch = obj.choices && obj.choices[0];
      if (!ch) continue;
      if (ch.delta.reasoning_content) { boardFirstToken(); msg.reasoning += ch.delta.reasoning_content; scheduleRender(); }
      if (ch.delta.content) {
        boardFirstToken();
        if (!msg.content && msg.reasoning) det.open = false; // answer started: fold the thinking
        msg.content += ch.delta.content; scheduleRender();
      }
      if (obj.x_stats) msg.stats = obj.x_stats;
    }
  } catch (e) {
    if (e.name === "AbortError") msg.stopped = true;
    else {
      const err = document.createElement("div");
      err.className = "error";
      err.textContent = e.message;
      node.appendChild(err);
    }
  } finally {
    controller = null;
    boardDone(msg.stats);
    bubble.classList.remove("cursor");
    bubble.innerHTML = markdown(msg.content);
    rtext.textContent = msg.reasoning;
    det.hidden = !msg.reasoning;
    fillMeta(meta, msg.stats);
    if (msg.stopped && !msg.stats) meta.textContent = "stopped";
    if (!msg.content && !msg.reasoning) history.pop(); // nothing came back: don't keep an empty turn
    save();
    setBusy(false);
  }
}

// ------------------------------------------------------------------ agent

const AGENT_MAX_STEPS = 10;
const TOOL_LABEL = {
  list_files: "List files", read_file: "Read", search_text: "Search", write_file: "Write", run_python: "Run Python",
};
let agentDefault = "";
let agentOn = false;

function toolArg(name, a = {}) {
  if (name === "search_text") return a.path && a.path !== "." ? `${a.pattern} · in ${a.path}` : a.pattern || "";
  if (name === "read_file") return a.start_line > 1 ? `${a.path} · from line ${a.start_line}` : a.path || "";
  if (name === "run_python") return ((a.code || "").split("\n").find((l) => l.trim()) || "").trim();
  return a.path || ".";
}

function toolState(t, live) {
  const r = t.result;
  if (!r) return live ? (t.needs_approval ? "waiting for your approval" : "…") : "stopped";
  if (!r.approved) return "denied";
  if (!r.ok) return "error";
  return t.needs_approval ? "allowed · done" : "done";
}

function fold(label, text, open) {
  const d = document.createElement("details");
  d.className = "tfold";
  d.open = open;
  d.innerHTML = "<summary></summary><pre></pre>";
  d.querySelector("summary").textContent = label;
  d.querySelector("pre").textContent = text;
  return d;
}

// one compact card per tool call; buttons only while the run is live and waiting for consent
function toolCard(t, live) {
  const r = t.result;
  const waiting = live && t.needs_approval && !r;
  const card = document.createElement("div");
  card.className = "tool" + (waiting ? " waiting" : "") + (r && !r.approved ? " denied" : r && !r.ok ? " err" : "");
  const head = document.createElement("div");
  head.className = "thead";
  head.innerHTML = '<span class="tname"></span><span class="targ"></span><span class="tstate"></span>';
  head.querySelector(".tname").textContent = TOOL_LABEL[t.name] || t.name;
  const targ = head.querySelector(".targ");
  targ.textContent = targ.title = toolArg(t.name, t.args || {});
  head.querySelector(".tstate").textContent = toolState(t, live);
  card.appendChild(head);
  if (t.needs_approval) {
    const a = t.args || {};
    const py = t.name === "run_python";
    const text = py ? a.code : a.content;
    card.appendChild(fold(py ? "Proposed code" : "Proposed content", text ?? JSON.stringify(a, null, 2), waiting));
  }
  if (waiting) {
    const ask = document.createElement("div");
    ask.className = "ask";
    ask.innerHTML = '<button type="button" class="btn go">Allow</button>' +
                    '<button type="button" class="btn">Deny</button><span class="tdecision"></span>';
    const [yes, no] = ask.querySelectorAll("button");
    yes.onclick = () => confirmTool(t, true, ask);
    no.onclick = () => confirmTool(t, false, ask);
    card.appendChild(ask);
  }
  if (r) card.appendChild(fold(!r.approved ? "Outcome" : r.ok ? "Result" : "Error", r.output || "(no output)", false));
  return card;
}

async function confirmTool(t, approve, ask) {
  ask.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  const out = ask.querySelector(".tdecision");
  try {
    const r = await fetch("/api/agent/confirm", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: t.id, approve }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    out.textContent = approve ? "Allowed" : "Denied";
  } catch (e) {
    out.textContent = e.message;
    out.classList.add("error");
  }
}

function partNode(p, live) {
  if (p.kind === "reasoning") {
    const det = document.createElement("details");
    det.className = "reasoning";
    det.innerHTML = '<summary>Reasoning</summary><div class="rtext"></div>';
    det.querySelector(".rtext").textContent = p.text;
    return det;
  }
  if (p.kind === "content") {
    const b = document.createElement("div");
    b.className = "bubble";
    b.innerHTML = markdown(p.text);
    return b;
  }
  return toolCard(p, live);
}

function agentFlow(m) {
  const flow = document.createElement("div");
  flow.className = "flow";
  const dir = document.createElement("div");
  dir.className = "agent-dir";
  dir.hidden = !m.workspace;
  dir.textContent = `Folder ${m.workspace || ""}`;
  flow.appendChild(dir);
  m.steps.forEach((p) => flow.appendChild(partNode(p, false)));
  return flow;
}

function agentNote(m) {
  if (!m.agent || m.agent.finish !== "max_steps") return null;
  const n = document.createElement("div");
  n.className = "agent-note";
  n.textContent = `Reached the ${AGENT_MAX_STEPS}-step limit`;
  return n;
}

function fillAgentMeta(meta, m) {
  fillMeta(meta, m.stats);
  if (!m.agent) return;
  const extra = [`${m.agent.steps} ${m.agent.steps === 1 ? "step" : "steps"}`];
  if (m.stats && m.stats.energy_j) extra.push(`${num(m.stats.energy_j)} J`);
  extra.reverse().forEach((t) => { const s = document.createElement("span"); s.textContent = t; meta.prepend(s); });
}

async function sendAgent(text) {
  const msg = { role: "assistant", content: "", steps: [], stats: null };
  const node = addTurn(text, msg);
  const flow = node.querySelector(".flow");
  const dir = flow.querySelector(".agent-dir");
  const meta = node.querySelector(".meta");
  const tail = document.createElement("div");  // blinking cursor, always last in the flow
  tail.className = "bubble cursor";
  flow.appendChild(tail);
  scrollDown(true);
  setBusy(true);
  boardStart();
  controller = new AbortController();

  const nodes = new Map();  // step part -> its element
  const dirty = new Set();
  let last = null, renderPending = false;
  let tokBase = 0, stepTok = 0;  // live token count restarts at every step: keep a running total
  const paint = (p) => {
    const n = nodes.get(p);
    if (p.kind === "reasoning") n.querySelector(".rtext").textContent = p.text;
    else if (p.kind === "content") n.innerHTML = markdown(p.text);
  };
  const scheduleRender = (p) => {
    dirty.add(p);
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      dirty.forEach(paint);
      dirty.clear();
      scrollDown(false);
    });
  };
  const addPart = (p) => {
    msg.steps.push(p);
    const n = partNode(p, true);
    nodes.set(p, n);
    flow.insertBefore(n, tail);
    last = p;
    return n;
  };
  const streamText = (kind, chunk) => {
    if (!run.tFirst) { boardFirstToken(); big(els.mPre, null); }  // the agent stream has no prompt counts
    if (!last || last.kind !== kind) {
      const prev = last;
      const n = addPart({ kind, text: "" });
      if (kind === "reasoning") n.open = true;
      else if (prev && prev.kind === "reasoning") nodes.get(prev).open = false; // answer started: fold the thinking
    }
    last.text += chunk;
    scheduleRender(last);
  };

  try {
    const res = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        messages: history.slice(0, -1).map(({ role, content }) => ({ role, content })),
        workspace: agentDir(),
        thinking_budget: Number(els.thinking.value),
        max_steps: AGENT_MAX_STEPS,
      }),
    });
    if (!res.ok) {
      const body = await res.text();
      let detail = body;
      try { detail = JSON.parse(body).detail || body; } catch {}
      throw new Error(res.status === 400 ? detail : `HTTP ${res.status}: ${detail}`);
    }

    for await (const { obj } of sse(res)) {
      switch (obj.type) {
        case "workspace":
          msg.workspace = obj.path;
          dir.textContent = `Folder ${obj.path}`;
          dir.hidden = false;
          break;
        case "step": tokBase += stepTok; stepTok = 0; last = null; break;
        case "reasoning": streamText("reasoning", obj.text); break;
        case "content": streamText("content", obj.text); break;
        case "stats": stepTok = obj.tokens || 0; boardLive({ ...obj, tokens: tokBase + stepTok }); break;
        case "tool_call": {
          const { type, ...call } = obj;
          addPart({ kind: "tool", ...call, result: null });
          scrollDown(call.needs_approval);
          break;
        }
        case "tool_result": {
          const p = msg.steps.find((s) => s.kind === "tool" && s.id === obj.id);
          if (!p) break;
          p.result = { ok: obj.ok, approved: obj.approved, output: obj.output };
          const n = toolCard(p, true);
          nodes.get(p).replaceWith(n);
          nodes.set(p, n);
          scrollDown(false);
          break;
        }
        case "done": msg.stats = obj.stats; msg.agent = { steps: obj.steps, finish: obj.finish_reason }; break;
        case "error": throw new Error(obj.message);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") msg.stopped = true;
    else {
      const err = document.createElement("div");
      err.className = "error";
      err.textContent = e.message;
      node.insertBefore(err, meta);
    }
  } finally {
    controller = null;
    boardDone(msg.stats);
    big(els.mPre, null);
    tail.remove();
    msg.steps = msg.steps.filter((p) => {
      if (p.kind === "content" && !p.text.trim()) { nodes.get(p).remove(); return false; }
      if (p.kind !== "tool") paint(p);
      else if (!p.result) nodes.get(p).replaceWith(toolCard(p, false)); // cut short: no buttons left behind
      return true;
    });
    msg.content = msg.steps.filter((p) => p.kind === "content").map((p) => p.text.trim()).join("\n\n");
    fillAgentMeta(meta, msg);
    const note = agentNote(msg);
    if (note) node.insertBefore(note, meta);
    if (msg.stopped && !msg.agent) meta.textContent = "stopped";
    if (!msg.steps.length) history.pop(); // nothing came back: don't keep an empty turn
    save();
    setBusy(false);
  }
}

// ------------------------------------------------------------------ environment (under the composer)

const RECENT_MAX = 5;
function recentDirs() {
  try { return JSON.parse(pref("engine.agentRecent")) || []; } catch { return []; }
}
function agentDir() { return pref("engine.agentDir") || agentDefault; }
function baseName(p) { return (p || "").replace(/\/+$/, "").split("/").pop() || p; }

function useDir(path) {
  pref("engine.agentDir", path);
  pref("engine.agentRecent", JSON.stringify([path, ...recentDirs().filter((d) => d !== path)].slice(0, RECENT_MAX)));
  showFolder();
}

function showFolder() {
  const dir = agentDir();
  els.folderName.textContent = dir ? baseName(dir) : "—";
  els.folderBtn.title = dir;
}

async function loadWorkspace() {
  if (agentDefault) return;
  try {
    agentDefault = (await (await fetch("/api/agent/workspace")).json()).default || "";
    showFolder();
  } catch {}
}

function setEnv(env) {
  agentOn = env === "agent";
  pref("engine.agent", agentOn ? "1" : "0");
  els.envSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.env === env)));
  els.folderWrap.hidden = els.envHint.hidden = !agentOn;
  els.input.placeholder = agentOn
    ? "Ask the agent to work on the files in the folder. Enter to send."
    : "Type here. Enter to send, Shift+Enter for a new line.";
  if (agentOn) loadWorkspace();
  else closeMenu();
}

function closeMenu() { els.folderMenu.hidden = true; els.folderBtn.setAttribute("aria-expanded", "false"); }

function openMenu() {
  const current = agentDir();
  const dirs = [current, ...recentDirs().filter((d) => d !== current)].filter(Boolean).slice(0, RECENT_MAX);
  els.folderMenu.innerHTML = "";
  const head = document.createElement("div");
  head.className = "mlabel";
  head.textContent = "Recent folders";
  els.folderMenu.appendChild(head);
  dirs.forEach((d) => {
    const it = document.createElement("button");
    it.type = "button";
    it.className = "mitem" + (d === current ? " on" : "");
    it.setAttribute("role", "menuitemradio");
    it.setAttribute("aria-checked", String(d === current));
    it.innerHTML = `<b>${esc(baseName(d))}</b><small>${esc(d.replace(/^\/Users\/[^/]+/, "~"))}</small>`;
    it.addEventListener("click", () => { useDir(d); closeMenu(); });
    els.folderMenu.appendChild(it);
  });
  const pick = document.createElement("button");
  pick.type = "button";
  pick.className = "mitem pick-new";
  pick.setAttribute("role", "menuitem");
  pick.textContent = "Choose another folder…";
  pick.addEventListener("click", pickDir);
  els.folderMenu.appendChild(pick);
  els.folderMenu.hidden = false;
  els.folderBtn.setAttribute("aria-expanded", "true");
}

async function pickDir() {
  closeMenu();
  els.folderBtn.disabled = true;
  els.folderName.textContent = "choose in Finder…";
  try {
    const r = await fetch("/api/agent/pick", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ current: agentDir() }),
    });
    const body = await r.json();
    if (!r.ok) alert(body.detail || `HTTP ${r.status}`);
    else if (body.path) useDir(body.path);
  } catch {}
  els.folderBtn.disabled = false;
  showFolder();
}

function setBusy(b) {
  els.send.hidden = b;
  els.stop.hidden = !b;
}

// ------------------------------------------------------------------ telemetry

async function poll() {
  try {
    const t = await (await fetch("/api/telemetry")).json();
    const temp = Math.max(t.gpu_c || 0, t.cpu_c || 0);
    const target = t.target_c;

    els.gpuVal.textContent = t.gpu_pct != null ? `${num(t.gpu_pct, 0)}%` : "—";
    setBar(els.gpuBar, (t.gpu_pct || 0) / 100);
    els.cpuVal.textContent = t.cpu_pct != null ? `${num(t.cpu_pct, 0)}%` : "—";
    setBar(els.cpuBar, (t.cpu_pct || 0) / 100);
    if (t.ram_total_gb) {
      els.memVal.textContent = `${num(t.ram_used_gb, 1)} of ${num(t.ram_total_gb, 0)} GB`;
      setBar(els.memBar, t.ram_used_gb / t.ram_total_gb, t.ram_used_gb / t.ram_total_gb > 0.92);
    } else {
      els.memVal.textContent = `${num(t.memory_gb)} GB`;
    }
    els.tempVal.textContent = temp ? `${num(temp, 0)} °C` : "—";
    els.tempVal.title = `obiettivo ${num(target, 0)} °C`;
    setBar(els.tempBar, temp / TEMP_SCALE, temp > target);
    els.targetMark.style.left = `${(target / TEMP_SCALE) * 100}%`;
    els.wattVal.textContent = t.watts != null ? `${num(t.watts)} W` : "—";
    setBar(els.wattBar, (t.watts || 0) / WATT_SCALE);

    els.mtpPick.hidden = !t.mtp_available;
    if (document.activeElement !== els.mtpSel) els.mtpSel.value = t.mtp_mode;
    if (document.activeElement !== els.modeSel && els.modeSel.options.length) els.modeSel.value = t.mode;
    showModelState(t.model_state, t.busy);
  } catch {
    els.dot.className = "dot";
  }
}

// ------------------------------------------------------------------ load / eject

const STATE_TEXT = { loading: "loading…", unloading: "ejecting…", unloaded: "not loaded · press Load or send a message" };
let modelState = "loaded";

function showModelState(state, busy) {
  modelState = state || "loaded";
  els.modelState.textContent = STATE_TEXT[modelState] || "";
  els.dot.className = "dot " + ({ loading: "loading", unloading: "loading", unloaded: "off" }[modelState] || (busy ? "busy" : "ok"));
  els.eject.textContent = modelState === "unloaded" ? "Load" : "Eject";
  els.eject.disabled = busy || modelState === "loading" || modelState === "unloading";
}

els.eject.addEventListener("click", async () => {
  const action = modelState === "unloaded" ? "load" : "unload";
  showModelState(action === "load" ? "loading" : "unloading", false);
  try {
    const r = await fetch(`/api/${action}`, { method: "POST" });
    if (!r.ok) alert((await r.json()).detail || `HTTP ${r.status}`);
  } catch {}
  poll();
});

// ------------------------------------------------------------------ modes

async function loadModes() {
  const m = await (await fetch("/api/modes")).json();
  els.modeSel.innerHTML = "";
  m.modes.forEach((mode) => {
    const o = document.createElement("option");
    o.value = mode.id;
    o.textContent = `${mode.label} · ${mode.target_c.toFixed(0)} °C`;
    els.modeSel.appendChild(o);
  });
  els.modeSel.value = m.current;
}

async function post(url, body) {
  await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  poll();
}
els.modeSel.addEventListener("change", () => post("/api/mode", { mode: els.modeSel.value }));
els.mtpSel.addEventListener("change", () => post("/api/mtp", { mode: els.mtpSel.value }));

// ------------------------------------------------------------------ wiring

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if ((!text && !attachments.length) || controller) return;
  els.input.value = "";
  autosize();
  (agentOn ? sendAgent : send)(text);
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.form.requestSubmit(); }
});
function autosize() { els.input.style.height = "auto"; els.input.style.height = els.input.scrollHeight + 2 + "px"; }
els.input.addEventListener("input", autosize);
els.stop.addEventListener("click", () => controller && controller.abort());
els.newChat.addEventListener("click", () => {
  if (controller) controller.abort();
  history = []; attachments = []; save(); renderAttachments(); renderAll(); els.input.focus();
});

// only off and a short budget: longer reasoning measured no better on the tasks and sometimes
// never ended (bench/thinking_budget.py). Any stored "on" value becomes the short budget.
els.thinking.value = [null, "0"].includes(pref("engine.thinking")) ? "0" : "512";
els.thinking.addEventListener("change", () => pref("engine.thinking", els.thinking.value));
els.maxTok.value = pref("engine.maxTok") || "4096";
els.maxTok.addEventListener("change", () => pref("engine.maxTok", els.maxTok.value));
els.envSeg.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-env]");
  if (b) setEnv(b.dataset.env);
});
els.folderBtn.addEventListener("click", () => (els.folderMenu.hidden ? openMenu() : closeMenu()));
document.addEventListener("click", (e) => { if (!els.folderWrap.contains(e.target)) closeMenu(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeMenu(); });
setEnv(pref("engine.agent") === "1" ? "agent" : "chat");

fetch("/v1/models").then((r) => r.json()).then((m) => {
  const id = m.data[0].id;
  els.model.textContent = id;
  els.emptyTitle.textContent = id;
}).catch(() => {});
loadModes().catch(() => {});
renderAll();
poll();
setInterval(poll, 1000);
