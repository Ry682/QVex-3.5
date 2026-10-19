# Studio: il nostro modello quantizzato per Qwen3.5-9B

**Data:** 19 ottobre 2026 · **Macchina:** MacBook Air M5, 16 GB, senza ventola
**Ispirazione:** ds4 di antirez (pesi quantizzati su misura per un motore specializzato)

## In breve

- Abbiamo costruito **ours-q4**, una versione a 4 bit di Qwen3.5-9B quantizzata da noi a partire dai pesi originali.
- **A parità di dimensione, velocità ed energia**, è **2,7 volte più fedele** al modello originale della versione MLX di LM Studio. Sceglie la stessa parola dell'originale il **91,9%** delle volte, contro l'86,7%.
- Le varianti a 3 bit sono fino al 13% più veloci, ma **consumano più energia per parola** e scaldano di più (fino a 90 °C). Per un Air senza ventola non convengono.
- **Decisione:** ours-q4 è ora il modello predefinito del motore.

## 1. Il problema

La versione MLX standard comprime ogni gruppo di 64 pesi in modo "cieco": prende il minimo e il massimo e divide l'intervallo in 16 livelli (4 bit). Tratta tutti i pesi come ugualmente importanti.

Non lo sono. Alcuni canali d'ingresso di ogni matrice ricevono valori molto più grandi di altri quando il modello legge testo vero. Un errore su quei pesi pesa molto di più nella risposta finale.

## 2. Il metodo

| Fase | Cosa abbiamo fatto |
|---|---|
| Pesi originali | Scaricati i pesi bf16 di Qwen/Qwen3.5-9B (19,3 GB) |
| Testi | 104 finestre da 512 token. 64 per la calibrazione e 40 per la valutazione, mai condivise. Fonti: prosa italiana e inglese (Progetto Gutenberg), codice Python, 48 chat in italiano. |
| Riferimento | Il modello originale fatto girare **strato per strato**, perché intero non entra in 16 GB. Da qui abbiamo ricavato le previsioni "vere" e l'**imatrix**, cioè quanto è attivo ogni canale d'ingresso di ognuna delle 249 matrici. |
| Quantizzazione | Per ogni gruppo di pesi: ricerca del taglio migliore dei valori estremi (25 combinazioni) e scala/offset ricalcolati ai minimi quadrati, **pesando ogni colonna con l'imatrix**. Il formato resta quello MLX, quindi stessi kernel e stessa velocità. |
| Sensibilità | Ognuno dei 129 pezzi del modello portato da 4 a 3 bit uno alla volta, misurando quanto peggiora la fedeltà (ΔKL). |
| Varianti | Un modello tutto a 4 bit, due misti 3/4 bit (scegliendo i pezzi meno sensibili) e uno tutto a 3 bit. |

**Errore per matrice rispetto a MLX standard** (pesato con l'imatrix, più basso è meglio):

| Larghezza | Miglioramento |
|---|---|
| 4 bit | circa **1,8–2×** più basso (misurato sugli strati 24–31; la stessa prova su una matrice sintetica dava 1,89×) |
| 3 bit | in media **2,8×** più basso (da 1,8× a 4,3× sui 32 strati) |

## 3. Risultati

Valutazione sulle 40 finestre mai viste in calibrazione:
- **KL:** quanto le previsioni si allontanano dall'originale (0 = identiche).
- **Stessa parola:** quanto spesso la parola più probabile coincide con quella dell'originale.
- **Compiti:** 24 prove verificabili di calcolo, fatti, traduzione e codice eseguito.

| Modello | Dimensione | KL ↓ | Stessa parola ↑ | Perplessità ↓ | Compiti | tok/s | J/token |
|---|---|---|---|---|---|---|---|
| Originale bf16 | 17,9 GB | 0 | 100% | 5,75 | – | – | – |
| **ours-q4** | **5,04 GB** | **0,026** | **91,9%** | **5,89** | **21/24** | **24,2** | **0,26** |
| LM Studio MLX 4 bit | 5,04 GB* | 0,069 | 86,7% | 5,98 | 21/24 | 25,2 | 0,26 |
| ours-mix 4,0 bpw | 4,53 GB | 0,066 | 86,9% | 6,13 | 21/24 | 26,1 | 0,31 |
| ours-mix 3,8 bpw | 4,34 GB | 0,078 | 86,1% | 6,21 | 21/24 | 27,4 | 0,33 |
| ours-q3 | 4,05 GB | 0,109 | 84,1% | 6,42 | 20/24 | 28,4 | 0,37 |

*Il file di LM Studio pesa 5,95 GB perché include la parte per le immagini, che il motore non carica.

**Per tipo di testo** (KL, più basso è meglio):

| Modello | Italiano | Inglese | Codice | Chat |
|---|---|---|---|---|
| **ours-q4** | **0,029** | **0,023** | **0,021** | **0,029** |
| LM Studio MLX | 0,084 | 0,065 | 0,047 | 0,076 |

Il guadagno più grande è proprio sull'**italiano**: 2,9 volte più fedele.

**Compiti:** tutti i modelli a 4 bit e i misti sbagliano gli stessi 3, cioè due calcoli e un giorno della settimana. Il modello li sbaglia anche senza compressione, se il ragionamento è spento. ours-q3 sbaglia in più una funzione Python: è il primo segno di degrado.

## 4. Velocità ed energia

Misurate con il ciclo di decodifica del motore, 512 token, 2 giri, ogni prova partendo sotto i 50 °C (mediana dei due giri).

- **ours-q4 e LM Studio** leggono gli stessi byte: velocità ed energia sono uguali entro il rumore della misura (±10%).
- **Le varianti a 3 bit** sono più veloci, ma macOS alza il clock della GPU, perché spacchettare 3 bit richiede più calcolo. La potenza sale più della velocità: **+20–45% di energia per parola**, e ours-q3 ha toccato **90 °C**.
- **Limite della misura:** il clock della GPU saliva anche con l'avanzare della sessione, e i modelli sono stati provati sempre nello stesso ordine. Una parte dell'aumento di energia può dipendere dall'ordine. La tendenza però è coerente in entrambi i giri.

## 5. Cosa abbiamo imparato

1. **La calibrazione conta più dei bit.** Lo stesso formato a 4 bit, quantizzato con l'imatrix, divide per 2,7 la distanza dall'originale.
2. **Su questo Mac, meno bit non vuol dire meno calore.** Il collo di bottiglia è la memoria, ma con meno byte la GPU lavora di più e sale di clock. Per un Air senza ventola la scelta giusta è 4 bit fatti bene.
3. **Le parti più delicate** sono le uscite dell'attenzione e l'ingresso del primo strato. Le più robuste sono gli ingressi di alcuni strati DeltaNet (13, 22, 26).
4. **Su 16 GB serve disciplina di memoria.** Il primo tentativo teneva in memoria tutti i pesi originali, fino a 18 GB di processo e 14 GB di swap. Corretto leggendo ogni matrice solo quando serve.

## 6. Prossimi passi possibili

- **ours-q4+:** le parti più delicate a 5–6 bit (+1–2% di dimensione) per ridurre ancora il KL.
- **Conversazioni salvate su disco**, come ds4: niente rilettura dopo un riavvio.
- **Ripetere la misura di energia** in ordine alternato, per isolare l'effetto dei 3 bit dalla deriva dei clock.

## 7. Budget di ragionamento

Stessi 24 compiti con ours-q4, ragionamento a budget diversi (`bench/thinking_budget.py`, risultati in `results/thinking_budget.json`). Media per compito:

| Ragionamento | Giusti | Token | Energia |
|---|---|---|---|
| Spento | 22/24 | 40 | 17 J |
| **Breve (512)** | **24/24** | 460 | 107 J |
| Medio (2.048) | 24/24 | 908 | 158 J |
| Libero | 19/21* | 1.187 | 213 J |

*Misura fermata a 21 compiti: il risultato era già chiaro.

Il ragionamento libero è il peggiore: in 2 compiti su 21 il modello non ha mai chiuso il ragionamento (6.144 token, circa 1.150 J ciascuno, nessuna risposta). Breve risolve tutto con metà dell'energia. **Decisione:** l'interfaccia offre solo Spento e Breve.

## Riprodurre

Il codice è in `quant/`, i dati in `data/` e i risultati grezzi in `results/quant.json`. I comandi sono nel `README.md`, nella sezione "Il nostro modello".
