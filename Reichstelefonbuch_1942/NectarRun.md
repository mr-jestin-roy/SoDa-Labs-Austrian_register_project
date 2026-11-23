# NectarRun — How to Run the Reichstelefonbuch OCR Pipeline

**Project:** Reichstelefonbuch 1942 Digitisation  
**Master script:** `digitizing python scripts/orchestrate_pipeline.py`

---

## Pipeline Overview

```
PDF → [pdf_page_to_png()]  →  600 DPI PNG
              ↓
   [cv_segment_test.py]    →  line/col overlays + cell crops
              ↓
  [ocr_cells_chandra.py]   →  OCR JSON
              ↓
    [json_to_xlsx.py]      →  Excel spreadsheet
```

### Scripts called internally

| Step | Script | How called | Env vars passed |
|------|--------|-----------|-----------------|
| 1 | `pdf_page_to_png()` (inline) | Direct Python via PyMuPDF | — |
| 2 | `cv_segment_test.py` | `subprocess.run()` | `INPUT_IMAGE`, `OUT_LINES`, `OUT_COLS`, `OUT_CELLS` |
| 3 | `ocr_cells_chandra.py` | `subprocess.run()` | `CELLS_DIR`, `GRIDS_DIR`, `OUT_JSON`, `VLLM_URL`, `PDF_FILENAME`, `FAILED_CSV` |
| 4 | `json_to_xlsx.py` | `subprocess.run()` | `IN_JSON`, `OUT_XLSX` |

### Key design details

- **Parallel execution** — `ThreadPoolExecutor` with 8 workers; pages run concurrently, log lines will interleave
- **Env-var communication** — child scripts receive all config via environment variables, not CLI args
- **Intermediate cleanup** — after success, `pages/page_XXXX.png` and `segments/page_XXXX/` are deleted unless `--keep-intermediates` is passed
- **OCR server** — `ocr_cells_chandra.py` talks to a local vLLM server at `localhost:8010` (configurable via `$VLLM_URL`)
- **Outputs kept** — only `results/page_XXXX_ocr.json` and `results/page_XXXX_ocr.xlsx`

---

## PDF Source Files

All 5 volumes must be present in the project root (`Reichstelefonbuch_1942/`) before running.

| File | Coverage | Size | Pages |
|------|----------|------|-------|
| `RTB_1_Aach-Berlichingen.pdf` | Aach → Berlichingen | 1.1 GB | 1,373 |
| `RTB_2_Berlin-Dotterwies.pdf` | Berlin → Dotterwies | 441 MB | — |
| `RTB_3_Drachhausen-Kaprun.pdf` | Drachhausen → Kaprun | 441 MB | — |
| `RTB_4_Karbow-Quittainen.pdf` | Karbow → Quittainen | 440 MB | — |
| `RTB_5_Raab-Zygry.pdf` | Raab → Zygry | 414 MB | — |
| `Verlagsprogramm.pdf` | Publisher programme (reference only) | 587 KB | — |

---

## Pre-requisites

### 1. Python environment

Use the project virtualenv:

```bash
source venv_layoutparser/bin/activate
```

Install dependencies if not already installed:

```bash
pip install -r requirements.txt
```

Key packages required:

| Package | Purpose |
|---------|---------|
| `pymupdf` (fitz) | PDF → PNG extraction |
| `opencv-python` | CV segmentation |
| `numpy` | Image array operations |
| `pillow` | DPI-preserving image save |
| `requests` | vLLM API calls |
| `pandas`, `openpyxl` | JSON → XLSX conversion |
| `layoutparser` | (legacy, kept for compatibility) |

### 2. vLLM inference server (Chandra model)

The OCR step requires a running vLLM server serving the Chandra vision model on `localhost:8010`.

Start the server before running the pipeline:

```bash
vllm serve chandra \
    --host 0.0.0.0 \
    --port 8010 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8
```

Verify it is up:

```bash
curl http://localhost:8010/v1/models
```

To use a remote vLLM server instead of localhost:

```bash
export VLLM_URL=http://<remote-host>:8010/v1/chat/completions
```

### 3. Output directories

The script creates these automatically, but confirm the project root has write access:

```
Reichstelefonbuch_1942/
├── pages/        ← temporary 600 DPI PNGs (deleted after success unless --keep-intermediates)
├── segments/     ← temporary cell crops (deleted after success unless --keep-intermediates)
└── results/      ← final JSON + XLSX outputs (kept)
```

### 4. GPU / compute

- The vLLM server requires a CUDA-capable GPU with sufficient VRAM to load the Chandra model.
- The orchestrator itself is CPU-bound (PDF extraction + CV segmentation). 8 parallel workers is the tested default.

---

## Running the Pipeline

### Process a page range

```bash
cd "Reichstelefonbuch_1942/digitizing python scripts"

python orchestrate_pipeline.py ../RTB_1_Aach-Berlichingen.pdf \
    --start 209 --end 1347 \
    --workers 8 \
    2>&1 | tee ../results/run.log
```

### Process specific pages

```bash
python orchestrate_pipeline.py ../RTB_1_Aach-Berlichingen.pdf \
    --pages 209,210,749,1200
```

### Keep intermediate files for debugging

```bash
python orchestrate_pipeline.py ../RTB_1_Aach-Berlichingen.pdf \
    --start 209 --end 215 \
    --keep-intermediates
```

### Run a different volume

```bash
python orchestrate_pipeline.py ../RTB_2_Berlin-Dotterwies.pdf \
    --start 1 --end 100 \
    --workers 8
```

---

## Outputs per page

For page 749 as an example:

| File | Location | Kept? |
|------|----------|-------|
| `page_0749.png` | `pages/` | No — deleted after success |
| `cv_segment_lines.png` | `segments/page_0749/` | No — deleted after success |
| `cv_segment_cols.png` | `segments/page_0749/` | No — deleted after success |
| `cells/*.png` | `segments/page_0749/cells/` | No — deleted after success |
| `page_0749_ocr.json` | `results/` | **Yes** |
| `page_0749_ocr.xlsx` | `results/` | **Yes** |

Use `--keep-intermediates` to retain `pages/` and `segments/` for debugging segmentation issues.

---

## Monitoring a run

The pipeline prints a timing summary at the end:

```
Page          PDF→PNG  Segment      OCR     XLSX    Total
page_0749      12.3s    18.7s    142.1s     2.4s   175.5s

✓ 1/1 pages processed successfully
⏱ Total run time: 175.5s (2.9 min)
```

Estimation formula printed automatically based on observed timings:
```
Per page      : ~175s (2.9 min)
100 pages     : ~291.7 min (4.9 hr)
1373 pages    : ~4007 min (66.8 hr)
```

---

## Known Issues

See `May13thHandoff.md` for full details and mitigations on:

1. **Vision token overflow** — JSON truncated mid-stream when composite grid image exceeds vLLM context budget
2. **Hallucinations on solid columns** — pages with no horizontal dividers produce a single oversized cell that overwhelms the model
