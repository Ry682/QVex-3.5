# QVex

An inference engine for **Qwen3.5-9B on Apple Silicon**, built on MLX with its own forward pass,
custom Metal kernels and a thermal governor designed for a fanless MacBook Air.
Web UI and OpenAI-compatible API included.

> Italian version: [README.it.md](README.it.md) · Quantization write-up (Italian): [reports/studio-quantizzazione.md](reports/studio-quantizzazione.md)

## Highlights

- **Own Qwen3.5 forward pass** (24 Gated DeltaNet + 8 gated-attention layers) with a Metal kernel for the DeltaNet recurrence; the state can be saved for rollback.
- **Thermal governor**: caps tok/s and pauses prefill to stay under a target temperature (Eco 70 °C / Balanced 80 °C / Max 95 °C). Power and temperature come from [`macmon`](https://github.com/vladkens/macmon), no sudo.
- **MTP speculative decoding** using Qwen's own MTP head as draft model (~490 MB download, only the MTP tensors are fetched via HTTP Range).
- **ours-q4**: a 4-bit quantization built from the bf16 weights with an importance matrix (imatrix). Same MLX format, same speed and memory, **2.7x lower KL** than the stock MLX 4-bit.
- **Agent mode** with sandboxed tools; writing files and running Python require confirmation.

## Requirements

- Apple Silicon Mac, macOS, 16 GB RAM (developed on a MacBook Air M5)
- [`uv`](https://github.com/astral-sh/uv) (the first run creates the Python 3.12 venv)
- [`macmon`](https://github.com/vladkens/macmon) (optional, for power/temperature telemetry): `brew install macmon`
- A Qwen3.5-9B MLX 4-bit model. Either:
  - **ours-q4** in `models/ours-q4/` (see "Getting the model" below), or
  - the LM Studio build `lmstudio-community/Qwen3.5-9B-MLX-4bit` in `~/.lmstudio/models/` (used as fallback, and required by the `quant/`, `tests/` and `bench/` scripts for config/tokenizer).

## Quick start

```bash
./start.sh
```

Open http://127.0.0.1:8080 and press **Load**. Options: `--model DIR`, `--port`, `--mode eco|balanced|max`, `--mtp-mode off|auto|on`, `--preload`.

Turbo mode (speculative decoding) needs the MTP head, downloaded once:

```bash
.venv/bin/python tools/fetch_mtp.py --bits 4
```

API example:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false,"enable_thinking":false}'
```

Responses include `x_stats` (tok/s, J/token, max temperature, cached tokens).

## ours-q4 results

Measured on 40 held-out evaluation windows (KL against the original bf16 model), MacBook Air M5:

| Model | KL ↓ | Same top token | Perplexity | Tasks | tok/s | J/token |
|---|---|---|---|---|---|---|
| bf16 original | 0 | 100% | 5.75 | – | – | – |
| **ours-q4** | **0.026** | **91.9%** | **5.89** | 21/24 | 24.2 | 0.26 |
| LM Studio MLX 4-bit | 0.069 | 86.7% | 5.98 | 21/24 | 25.2 | 0.26 |
| ours-mix 3/4 bit (4.0 bpw) | 0.066 | 86.9% | 6.13 | 21/24 | 26.1 | 0.31 |
| ours-q3 | 0.109 | 84.1% | 6.42 | 20/24 | 28.4 | 0.37 |

Caveats: only 24 verifiable tasks (a sanity check, not a benchmark); the energy runs always used the same model order and GPU clocks drift during a session, so part of the 3-bit energy penalty may be an ordering effect; one machine only.

### MTP speculative decoding (256 tokens, T=0.7)

| Prompt | Normal | MTP | Acceptance |
|---|---|---|---|
| code | 25.4 tok/s · 0.34 J/tok | 34.0 tok/s · 0.39 J/tok | 76% |
| prose | 24.6 tok/s · 0.36 J/tok | 32.8 tok/s · 0.40 J/tok | 62% |
| explanation | 25.6 tok/s · 0.33 J/tok | 37.2 tok/s · 0.41 J/tok | 73% |

MTP is faster but costs ~10–25% more energy per token (macOS raises GPU clocks), so the default `auto` mode only uses it while the chip is at least 5 °C under the target.

## Getting the model

Rebuilding ours-q4 from scratch needs ~20 GB of disk for the original weights:

```bash
.venv/bin/python -m quant.data                      # calibration and evaluation texts
.venv/bin/python -m quant.reference                 # layer-by-layer bf16 pass: imatrix + reference
.venv/bin/python -m quant.quantize cache --bits 4 3
.venv/bin/python -m quant.quantize sensitivity      # KL cost of each tensor at 3 bit
.venv/bin/python -m quant.build --name ours-q4 --all 4
.venv/bin/python -m quant.evaluate lmstudio ours-q4 # quality + tasks
.venv/bin/python -m quant.speed lmstudio ours-q4    # speed and energy
```

## Repository layout

```
engine/       the engine (Python + MLX + Metal kernels)
web/          browser UI
tools/        fetch and prepare weights
quant/        our quantization (imatrix, sensitivity, evaluation)
data/         calibration/evaluation texts, sensitivity results
tests/        correctness checks (parity vs mlx_lm, rollback, MTP)
bench/        speed, power and temperature measurements
reports/      write-ups
experiments/  unused experiments (native C/Metal engine, paused)
```

## Credits and license

Code: MIT, see [LICENSE](LICENSE). Model weights derived from [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) remain under Qwen's Apache-2.0 license. Inspired by antirez's ds4; built on [MLX](https://github.com/ml-explore/mlx). Calibration texts include public-domain prose from Project Gutenberg.
