# QVex

> English version: [README.md](README.md)

QVex è un motore di inferenza per Qwen3.5-9B (MLX 4-bit) su Apple Silicon, scritto su MLX
con un proprio forward pass, kernel Metal custom e un governor termico pensato
per un MacBook Air senza ventola. Interfaccia web e API compatibile OpenAI.

## Avvio

```bash
./start.sh
```

Poi apri http://127.0.0.1:8080. La prima volta crea da solo l'ambiente Python.
Il modello non viene caricato all'avvio: premi **Load** nell'interfaccia (o manda un messaggio). Con `--preload` lo carica subito.
Opzioni: `--model DIR`, `--port`, `--mode eco|balanced|max`, `--mtp-mode off|auto|on`, `--preload`.

Per la modalità Turbo (decodifica speculativa) serve la testa MTP. Si scarica una volta sola,
~490 MB invece dei 14 GB degli shard originali, leggendo solo i byte dei tensori MTP:

```bash
.venv/bin/python tools/fetch_mtp.py --bits 4
```

## Cartelle

```
LLM/
├── start.sh          avvia tutto
├── requirements.txt  librerie Python
├── engine/           il motore (Python + MLX + kernel Metal)
├── web/              l'interfaccia nel browser
├── tools/            scaricare e preparare i pesi
├── quant/            la nostra quantizzazione (imatrix, sensibilità, valutazione)
├── data/             testi di calibrazione/valutazione, imatrix, riferimento
├── tests/            controlli di correttezza
├── bench/            misure di velocità, watt e temperatura
├── results/          risultati delle misure (solo i riassunti sono versionati)
├── models/           pesi scaricati, testa MTP (non versionati)
└── experiments/      prove non usate dal motore (motore nativo C/Metal in pausa)
```

| File | Cosa fa |
|---|---|
| `engine/server.py` | FastAPI: `/v1/chat/completions` (stream e non), telemetria, modalità, espelli/carica |
| `engine/engine.py` | Thread di lavoro, prefill a blocchi, memoria della conversazione tra i turni, espulsione |
| `engine/model.py` | Forward pass Qwen3.5 (24 strati Gated DeltaNet + 8 attenzione con gate), caricamento pesi, cache |
| `engine/kernels.py` | Kernel Metal: ricorrenza DeltaNet (stato salvabile per il rollback); matvec multi-riga sperimentale |
| `engine/mtp.py` | Testa MTP di Qwen3.5 usata come modello bozza (condivide embedding e lm_head) |
| `engine/speculative.py` | Ciclo di decodifica: passi normali o MTP (bozza + verifica a 2 token) |
| `engine/thermal.py` | Governor: limita i tok/s e mette pause nel prefill per stare sotto la temperatura obiettivo |
| `engine/telemetry.py` | Potenza, temperatura, uso GPU/CPU e memoria via `macmon` (senza sudo) |
| `engine/sampler.py` | temperature / top-k / top-p / presence penalty |
| `engine/tokenizer.py` | Tokenizer + chat template senza `transformers` |
| `tools/fetch_mtp.py` | Scarica solo i tensori MTP (HTTP Range) e li quantizza |
| `tests/parity.py` | Confronto con `mlx_lm` (`ref` poi `ours`) |
| `tests/rollback_test.py`, `tests/mtp_teacher.py` | Rollback dello stato ricorrente, MTP contro decodifica normale |
| `tests/chat_smoke.py` | Prova veloce di una chat completa |
| `bench/bench.py` | Velocità + watt + temperatura (anche in modalità prolungata) |
| `bench/mtp_bench.py` | Normale vs MTP: tok/s, J/token, accettazione |

## Il nostro modello (qvex-q4)

Come fa ds4 di antirez, il motore usa pesi quantizzati da noi invece della versione MLX generica.
Si parte dai pesi originali bf16 di Qwen/Qwen3.5-9B e ogni matrice viene compressa a 4 bit
pesando le colonne secondo l'**imatrix**: quanto ogni canale d'ingresso è attivo su testi reali
(italiano, inglese, codice, chat). Errori sui canali importanti costano di più, quindi il
quantizzatore li evita (range con clipping cercato + scala/offset rifittati ai minimi quadrati pesati).
Il formato resta quello MLX: stessi kernel, stessa velocità, stessa memoria.

Misure su 40 finestre di valutazione mai viste in calibrazione (KL dal modello bf16 originale):

| Modello | KL ↓ | stessa parola dell'originale | perplessità | compiti | tok/s | J/token |
|---|---|---|---|---|---|---|
| bf16 originale | 0 | 100% | 5,75 | – | – | – |
| **qvex-q4** | **0,026** | **91,9%** | **5,89** | 21/24 | 24,2 | 0,26 |
| LM Studio MLX 4-bit | 0,069 | 86,7% | 5,98 | 21/24 | 25,2 | 0,26 |
| qvex-mix 3/4 bit (4,0 bpw) | 0,066 | 86,9% | 6,13 | 21/24 | 26,1 | 0,31 |
| qvex-q3 | 0,109 | 84,1% | 6,42 | 20/24 | 28,4 | 0,37 |

Le varianti a 3 bit sono un po' più veloci ma consumano più energia per token: spacchettare
3 bit costa più calcolo, macOS alza i clock della GPU (fino a 90 °C con qvex-q3). Per un Air senza
ventola il default è quindi qvex-q4. Il modello di LM Studio resta usabile con `--model`.

Ricostruirlo da zero (serve ~20 GB di disco per i pesi originali):

```bash
.venv/bin/python -m quant.data                      # testi di calibrazione e valutazione
.venv/bin/python -m quant.reference                 # bf16 strato per strato: imatrix + riferimento
.venv/bin/python -m quant.quantize cache --bits 4 3
.venv/bin/python -m quant.quantize sensitivity      # costo in KL di ogni pezzo a 3 bit
.venv/bin/python -m quant.build --name qvex-q4 --all 4
.venv/bin/python -m quant.evaluate lmstudio qvex-q4 # qualità + compiti
.venv/bin/python -m quant.speed lmstudio qvex-q4    # velocità ed energia
```

## Modalità termiche

| Modalità | Obiettivo |
|---|---|
| Eco | 70 °C |
| Balanced | 80 °C |
| Max | 95 °C |

## Decodifica speculativa (MTP)

Misure su MacBook Air M5, 256 token, temperatura 0,7:

| Prompt | Normale | MTP | Accettazione |
|---|---|---|---|
| codice | 25,4 tok/s · 0,34 J/tok | 34,0 tok/s · 0,39 J/tok | 76% |
| prosa | 24,6 tok/s · 0,36 J/tok | 32,8 tok/s · 0,40 J/tok | 62% |
| spiegazione | 25,6 tok/s · 0,33 J/tok | 37,2 tok/s · 0,41 J/tok | 73% |

L'MTP è più veloce ma costa ~10–25% di energia in più per token: il passaggio di
verifica a 2 token dura quanto uno a 1 token, però macOS alza i clock della GPU
(DVFS) e la potenza quasi raddoppia. Per questo il default è **auto**: MTP solo
finché il chip è almeno 5 °C sotto l'obiettivo e il governor non sta frenando.
In greedy l'output coincide con la decodifica normale a meno di pareggi bf16
(verificato con `tests/mtp_teacher.py`: scarti di 0–0,125 logit).

## API

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Ciao"}],"stream":false,"enable_thinking":false}'
```

Le risposte includono `x_stats` (tok/s, J/token, temperatura massima, token dalla cache).

## Espellere il modello

Il pulsante **Eject** in alto (o `POST /api/unload`) libera la memoria del modello,
circa 5 GB. Il server resta acceso: il modello si ricarica da solo al messaggio successivo
(1–2 s se il file è ancora nella cache del disco di macOS), oppure con **Load**
(`POST /api/load`).

## Modalità agente

Sotto il box del testo scegli l'ambiente: **Chat** o **Agent**. In modalità agente il modello lavora dentro una cartella (predefinita `~/QVex`): cliccandola scegli tra le recenti o apri la finestra del Finder con "Choose another folder…". Gli strumenti (`engine/agent.py`):

| Strumento | Cosa fa | Conferma |
|---|---|---|
| `list_files` | elenca file e cartelle | no |
| `read_file` | legge un file di testo, 200 righe alla volta | no |
| `search_text` | cerca una parola o un'espressione regolare | no |
| `write_file` | crea o sovrascrive un file | **sì** |
| `run_python` | esegue uno script Python nella cartella (30 s) | **sì** |

I percorsi fuori dalla cartella sono bloccati. Al massimo 10 passi per richiesta e risultati degli strumenti tagliati a 6.000 caratteri, per contenere token e calore. Ogni passo riusa lo stato del passo precedente, quindi rilegge solo la parte nuova.

API: `POST /api/agent/pick` apre la finestra del Finder, `POST /api/agent` (eventi SSE: `step`, `reasoning`, `content`, `tool_call`, `tool_result`, `done`) e `POST /api/agent/confirm` con `{"id", "approve"}`.
