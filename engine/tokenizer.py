"""Tokenizer + chat template, without transformers."""

import json
from datetime import datetime
from pathlib import Path

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def _raise(msg):
    raise TemplateError(msg)


class ChatTokenizer:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())

        tpl_file = model_dir / "chat_template.jinja"
        source = tpl_file.read_text() if tpl_file.exists() else cfg["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(
            x, indent=indent, ensure_ascii=ensure_ascii)
        self.template = env.from_string(source)

        self.stop_ids = set()
        for name in (cfg.get("eos_token"), "<|im_end|>", "<|endoftext|>"):
            if name and (i := self.tok.token_to_id(name)) is not None:
                self.stop_ids.add(i)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def render(self, messages: list[dict], add_generation_prompt=True, enable_thinking=True,
               tools: list[dict] | None = None) -> str:
        return self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )


class StreamDecoder:
    """Incremental detokenizer that never emits half a UTF-8 character."""

    def __init__(self, tok: ChatTokenizer):
        self.tok = tok
        self.pending: list[int] = []

    def push(self, token: int) -> str:
        self.pending.append(token)
        text = self.tok.decode(self.pending)
        if text.endswith("�") and len(self.pending) < 8:
            return ""
        self.pending = []
        return text

    def flush(self) -> str:
        text = self.tok.decode(self.pending) if self.pending else ""
        self.pending = []
        return text
