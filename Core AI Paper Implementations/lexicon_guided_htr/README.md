# Lexicon-Guided HTR — Pre-OCR Context Injection from the Reichstelefonbuch 1942

Handwriting recognition for the Austrian parish registers (Trauungsbücher,
Kurrent script) **guided by the 1.88M-entry town corpus** digitised from the
Reichstelefonbuch 1942. Instead of OCR → post-hoc correction, the domain
vocabulary is injected *before/during* decoding so the VLM resolves ambiguous
strokes against attested names while it is still looking at the pixels.

## Mapping to the papers in `Core AI Paper Implementations/`

| Paper | What we take from it |
|---|---|
| **Read Like Humans (ABINet, CVPR 2021)** | The iterative vision↔language loop: pass 1 produces a draft with explicit `⟦uncertainty masks⟧` (the cloze), the external language prior proposes fillers, and the model *re-reads the image* to resolve them — repeated until no masks remain. |
| **MegaRAG (Multimodal KG-based RAG)** | The reference corpus is not a flat word list but an entity-relation graph: `town —HAS_RESIDENT→ surname`. Retrieval pulls the parish's 1-hop neighbourhood (surnames attested in Althofen itself) and 2-hop neighbourhood (surnames aggregated over Kärnten), exploiting the fact that Austrian surnames are intensely geographically local. |
| **Dictionary-Guided Scene Text Recognition (CVPR 2021)** | (a) fuzzy edit-probability ranking of lexicon candidates per masked token; (b) the optional `logit_constrained.py` backend, which biases/masks decoder logits through a token trie built from the scoped lexicon. |
| **Qianfan-VL (domain-enhanced VLM)** | Motivation for the future step: the corrected transcriptions produced here become domain training data for fine-tuning (see `Fine Tuning OCR Model/`). |

## Architecture

```
 Reichstelefonbuch 1942 (5 master_RTB_*_clean.xlsx, 1.88M entries)
        │  build_lexicon.py  (Stage A — one-off)
        ▼
 lexicon.json.gz + kg_town_surnames.json.gz          12.6k towns × 85k surnames
        │  kg_retrieval.py   (Stage B — per parish)
        ▼
 ScopedLexicon("Althofen (Kärnten)")
   1-hop: surnames in Althofen   2-hop: surnames in Kärnten
   + given names, occupations, Carinthian place names
        │
        ▼                       pipeline.py (per page)
 ┌─ Pass 1: VLM transcribes page, wraps uncertain words in ⟦masks⟧
 │
 ├─ Retrieval: rapidfuzz ranks lexicon candidates per ⟦mask⟧
 │             (1-hop surnames get a score bonus)
 │
 └─ Pass 2..N: VLM re-reads the IMAGE with scoped lexicon + per-mask
               candidates in context; keeps the mask if no candidate is
               visually plausible. Iterates while masks remain.
        │
        ▼
 result.json  →  evaluate.py (CER/WER vs ground truth, masks resolved,
                              RTB entity hit-rate)
```

## Quick start

```bash
cd "Core AI Paper Implementations/lexicon_guided_htr"

# Stage A — build lexicon + KG from the RTB masters (one-off, ~4 min)
../../venv/bin/python build_lexicon.py

# Sanity-check retrieval for the parish
../../venv/bin/python kg_retrieval.py

# Inspect the exact prompts without a VLM endpoint
../../venv/bin/python pipeline.py --image x.jpg --dry-run | less

# Full run against the vLLM endpoint (Chandra, as in the Nectar setup)
VLM_BASE_URL=http://localhost:8010/v1 VLM_MODEL=chandra \
../../venv/bin/python pipeline.py \
  --image "../../Althofen_Marriage Book_input_images/Althofen_TrauungsbuchTomIV_1907_1936_00015.jpg" \
  --parish "Althofen (Kärnten)" --out result.json

# Same pipeline with Qwen3-VL served by vLLM
VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct" ../../venv/bin/python pipeline.py --image … 

# Evaluate draft vs guided final
../../venv/bin/python evaluate.py --result result.json [--truth gt.txt]
```

Serving the models on Nectar (examples):

```bash
vllm serve datalab-to/chandra --port 8010          # Chandra OCR
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8010   # Qwen3-VL
```

## Optional: logit-level constraint (experimental)

`logit_constrained.py` reproduces the strict CVPR mechanism with a local
HF Qwen3-VL: a token trie over the scoped lexicon adds `--bias` to every
lexicon-consistent next token (or, with `--hard`, forbids leaving the trie
mid-word). Needs local weights + GPU; cannot run through the remote chat API.

```bash
venv/bin/python logit_constrained.py --image page.jpg --bias 4.0
```

Caveat: `--hard` will mangle out-of-lexicon text (dates, Latin notes) — use
soft bias unless transcribing isolated name fields.

## Design notes / known limitations

* **Hallucination control** — the pass-2 prompt forbids replacing a mask
  unless the candidate is visually consistent, and `evaluate.py` reports CER
  (not only lexicon hit-rate) precisely to catch "valid but wrong name"
  substitutions. Compare draft vs final CER on a ground-truthed page before
  trusting the gains.
* **Temporal fit** — RTB is 1942; the Althofen registers span 1907–1936, so
  family-name overlap is high, but telephone-era institutional vocabulary is
  filtered out at build time (`INSTITUTION_WORDS` / suffix rules in
  `build_lexicon.py`).
* **Scaling the prompt** — the full lexicon never enters the prompt; only the
  KG-scoped slices (~500 terms ≈ 2–3k tokens). The *full* state surname pool
  is searched only by rapidfuzz when generating per-mask candidates.
* **Other parishes** — pass `--parish "Pernegg (Steiermark)"` etc.; the town
  is fuzzy-resolved against 12.6k region headings, and the state is derived
  from the heading's bracket suffix.
