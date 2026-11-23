# RTB Volume 5 — JSON → XLSX Pipeline: RL Teacher Reference

> **Purpose of this document:** This is the ground-truth reference for evaluating a student LLM's solution. Each section describes exactly what the correct pipeline must do. Use it as the reward signal: a solution scores higher the more precisely it matches the behaviour described here.

---

## 1. Project Context

The **Reichstelefonbuch 1942** is a digitised German/Austrian telephone directory. The directory is split into five volumes. Each volume has been OCR-processed page by page; the OCR output is stored as JSON files. The project pipeline then converts these JSON files first into per-page Excel spreadsheets and then aggregates those into a single master spreadsheet per volume.

Volumes 1–4 have already been fully processed. **Volume 5 (`results_RTB_5_Raab-Zygry/`)** has completed OCR and now needs the same two-stage pipeline applied to it.

---

## 2. Directory Layout

```
RTB_results_1_2_3_4/                          ← working directory (BASE)
│
├── batch_convert_all_rtb.py                  ← Stage 1 script (reference implementation)
├── merge_pages_to_master.py                  ← Stage 2 script (reference implementation)
│
├── results_RTB_1_Aach-Berlichingen/          ← Vol 1: already has page_XXXX_ocr.json + .xlsx
├── results_RTB_2_Berlin-Dotterwies/          ← Vol 2: same
├── results_RTB_3_Drachhausen-Kaprun/         ← Vol 3: same
├── results_RTB_4_Karbow-Quittainen/          ← Vol 4: same
│
├── results_RTB_5_Raab-Zygry/                 ← Vol 5: 1349 JSON files, ZERO xlsx yet
│   ├── checkpoint.json
│   ├── page_0006_ocr.json
│   ├── page_0007_ocr.json
│   ├── ...
│   └── page_1353_ocr.json
│
├── master_RTB_1_Aach-Berlichingen.xlsx       ← Already produced masters for Vols 1–4
├── master_RTB_2_Berlin-Dotterwies.xlsx
├── master_RTB_3_Drachhausen-Kaprun.xlsx
└── master_RTB_4_Karbow-Quittainen.xlsx
```

**Key facts about RTB_5:**
- Pages span **0006 to 1353** (1349 JSON files total, excluding `checkpoint.json` and the log file).
- **Zero** `page_*_ocr.xlsx` files exist inside `results_RTB_5_Raab-Zygry/` yet.
- Both JSON formats appear in RTB_5 (see Section 4).

---

## 3. Two-Stage Pipeline Overview

```
Stage 1 ─ JSON → per-page XLSX
  Input:  results_RTB_5_Raab-Zygry/page_XXXX_ocr.json   (for every page)
  Output: results_RTB_5_Raab-Zygry/page_XXXX_ocr.xlsx   (same directory)

Stage 2 ─ per-page XLSX → master XLSX
  Input:  results_RTB_5_Raab-Zygry/page_*_ocr.xlsx       (all pages)
  Output: master_RTB_5_Raab-Zygry.xlsx                   (in BASE)
```

The correct new script should extend the reference scripts to cover RTB_5. It may run both stages in one script or keep them separate — that is at the student's discretion — but the **inputs, outputs, column schema, sort order, and XLSX formatting must exactly match the reference scripts**.

---

## 4. JSON File Structure — Two Formats

Every `page_XXXX_ocr.json` file is a **JSON array** of batch-record objects. The top-level structure is always:

```json
[
  {
    "batch": "01",
    "cells": ["col1_sec01.png"],
    "grid_layout": ...,
    "grid_image": ...,
    "data": <see below>
  },
  ...
]
```

The `data` field can be one of two formats.

---

### 4.1 Format 1 — Raw HTML (`data` is a `dict` with key `"raw"`)

```json
{
  "batch": "03",
  "cells": ["col2_sec04.png"],
  "data": {
    "raw": "<div data-label='Section-Header'>...</div><div data-label='Text'>...</div>"
  }
}
```

**How to detect:** `isinstance(data, dict) and "raw" in data`

**How to parse:** The HTML inside `data["raw"]` is parsed into regions. Two sub-cases exist:

#### 4.1a — With `Section-Header` divs (multi-region pages)

The HTML contains `<div data-label='Section-Header'>` elements. Each Section-Header marks the start of a new named region. Subsequent `Text` or `Table` divs belong to that region until the next Section-Header.

- Extract the region name from the heading (`<h1>`–`<h4>` or `<p>`) inside each Section-Header div.
- For `Text` divs: split on `<br />` tags, strip HTML, match the phone-extension regex (see Section 5), and append entries.
- For `Table` divs: parse `<tr>/<td>` pairs where column 0 is the extension and column 1 is the subscriber text.
- Labels `Image`, `Page-Header`, `Page-Footer`, `Table-Of-Contents` are **skipped**.

#### 4.1b — Without `Section-Header` divs (single-region pages)

All content is treated as a single region. The first non-extension line of text becomes the `region_name` if no entries have been added yet. Otherwise continuation text is appended to the last entry's `text` field.

**HTML cleanup rules (applied before parsing):**

1. Strip non-printable control characters (`\x00`–`\x08`, `\x0b`, `\x0c`, `\x0e`–`\x1f`).
2. If the number of `<` exceeds `>` by 1–3, append the corresponding number of `</div>` tags to close truncated HTML.
3. Unescape HTML entities (e.g. `&amp;` → `&`, `&uuml;` → `ü`).

---

### 4.2 Format 2 — Pre-parsed regions (`data` is a `list` of region dicts)

```json
{
  "batch": "01",
  "cells": ["col1_sec01.png"],
  "data": [
    {
      "region_name": "Raab",
      "entries": [
        { "extension": "2",   "text": "Mustermann Hans, Kaufmann" },
        { "extension": "17",  "text": "Stadt Raab, Gemeindeamt" }
      ]
    }
  ]
}
```

**How to detect:** `isinstance(data, list) and data and isinstance(data[0], dict) and "entries" in data[0]`

**How to handle:** Pass the region list directly to the row-building step without HTML parsing. The `region_name` and `entries[].extension` / `entries[].text` fields are used as-is (after HTML-unescaping the `region_name`).

**Both formats appear in RTB_5.** In the first 50 pages sampled, 223 batch-records are Format 1 and 1452 are Format 2.

---

## 5. Phone-Extension Regex

Used only during HTML parsing (Format 1 text blocks). It identifies lines that begin with a telephone extension number:

```python
_EXT_RE = re.compile(r'^(\(\d+\)|\d{1,6}[a-z]?)\s+(.+)$')
```

- Group 1: the extension — either `(digits)` or up to 6 digits optionally followed by one lowercase letter.
- Group 2: the subscriber text (everything after the extension and whitespace).

If a line does **not** match and there are existing entries, the text is appended to the last entry's `text` (continuation line).

---

## 6. Hallucination Guard

Before writing entries for a region, check whether the OCR hallucinated by repeating the same content. If **any `(extension, text)` pair appears 5 or more times** within a single batch-record's entries, that entire batch-record is **skipped** (not written to XLSX).

```python
from collections import Counter

def is_hallucinated(entries: list) -> bool:
    counts = Counter((e.get("extension", ""), e.get("text", "")) for e in entries)
    return any(v >= 5 for v in counts.values())
```

---

## 7. Batch-Record Sort Order (within a page)

Before processing, sort all batch-records within a page by their `cells` value so columns are processed left-to-right, top-to-bottom:

```python
def _cell_sort_key(rec):
    cell = rec.get("cells", [""])[0] if rec.get("cells") else ""
    m = re.match(r'col(\d+)_sec(\d+)', cell)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

records = sorted(records, key=_cell_sort_key)
```

---

## 8. XLSX Column Schema (Stage 1 Output)

Each `page_XXXX_ocr.xlsx` file has exactly **5 columns** in this order:

| # | Header | Width | Content |
|---|--------|-------|---------|
| A | `batch` | 20 | `{page_stem}_b{batch_num}`, e.g. `page_0209_b02` |
| B | `cells` | 18 | First element of the `cells` list, e.g. `col1_sec01.png` |
| C | `region` | 25 | Current section header / region name (carried forward across batches) |
| D | `phone_extension` | 14 | Extension string, e.g. `42` or `(42)` |
| E | `transcribed_text` | 70 | Subscriber text |

**Region carry-forward rule:** The `last_region_name` variable is updated whenever a non-empty `region_name` is encountered and used for all subsequent rows until the next non-empty region name, **even across batch-record boundaries within the same page**.

**Batch label construction:**
```python
page_id    = json_path.stem.replace("_ocr", "")   # "page_0006"
batch_num  = rec.get("batch", "")
batch_label = f"{page_id}_b{batch_num}" if batch_num else page_id
```

**Empty-entries rows:** When a region has no entries (entries list is empty and was not skipped), write one row with empty `phone_extension` and `transcribed_text`, preserving the batch and region.

**Control-character scrubbing:** All five cell values are passed through `_strip_control()` before writing to XLSX to remove `\x00`–`\x08`, `\x0b`, `\x0c`, `\x0e`–`\x1f`.

---

## 9. XLSX Formatting (Stage 1)

```python
header_font  = Font(bold=True, color="FFFFFF")
header_fill  = PatternFill("solid", fgColor="2F5496")   # dark blue
header_align = Alignment(horizontal="center", vertical="center")

ws.title       = "Reichstelefonbuch"
ws.freeze_panes = "A2"
```

---

## 10. Stage 2 — Merging Page XLSX Files into Master

### 10.1 Input

All `page_*_ocr.xlsx` files inside `results_RTB_5_Raab-Zygry/`.

### 10.2 Merge logic

1. Read each file with `pandas.read_excel(xlsx_file, sheet_name=0)`.
2. Skip files with zero data rows.
3. `pd.concat(all_dfs, ignore_index=True)` into one master DataFrame.
4. Sort by `(page_num, batch_num, col_num, sec_num)` using this key:

```python
import re

def extract_batch_cell_sort_key(batch: str, cells: str):
    page_match  = re.search(r'page_(\d+)', batch)
    page_num    = int(page_match.group(1)) if page_match else 99999

    batch_match = re.search(r'b(\d+)', batch)
    batch_num   = int(batch_match.group(1)) if batch_match else 999

    cell_match  = re.match(r'col(\d+)_sec(\d+)', str(cells))
    col_num     = int(cell_match.group(1)) if cell_match else 999
    sec_num     = int(cell_match.group(2)) if cell_match else 999

    return (page_num, batch_num, col_num, sec_num)
```

5. Drop the temporary sort-key column and reset the index.
6. Write to XLSX.

### 10.3 Output file

```
master_RTB_5_Raab-Zygry.xlsx
```

Saved in `BASE` (i.e., `RTB_results_1_2_3_4/`), **not** inside the `results_RTB_5_Raab-Zygry/` subdirectory.

### 10.4 Master XLSX Formatting

```python
header_font  = Font(bold=True, color="FFFFFF")
header_fill  = PatternFill("solid", fgColor="1F4E78")   # slightly darker blue than Stage 1
header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

ws.title       = "Reichstelefonbuch"
ws.freeze_panes = "A2"
```

Column widths in the master XLSX:

| Column | Width |
|--------|-------|
| A batch | 22 |
| B cells | 18 |
| C region | 30 |
| D phone_extension | 16 |
| E transcribed_text | 80 |

---

## 11. Required Dependencies

```bash
pip install openpyxl pandas
```

- `openpyxl` — for building and formatting XLSX files.
- `pandas` — for reading per-page XLSX files and concatenating DataFrames in Stage 2.
- Standard library only for Stage 1: `json`, `re`, `html`, `glob`, `pathlib`, `collections`.

---

## 12. Evaluation Rubric (RL Reward Signal)

A student solution is scored on the following criteria. Each criterion is binary (pass/fail) unless noted.

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Targets `results_RTB_5_Raab-Zygry/` (not RTB 1–4) | Critical |
| 2 | Correctly detects and handles **both** Format 1 and Format 2 JSON | Critical |
| 3 | Produces `page_XXXX_ocr.xlsx` inside `results_RTB_5_Raab-Zygry/` for all 1349 pages | Critical |
| 4 | Produces `master_RTB_5_Raab-Zygry.xlsx` in `RTB_results_1_2_3_4/` | Critical |
| 5 | Correct 5-column schema: `batch`, `cells`, `region`, `phone_extension`, `transcribed_text` | Critical |
| 6 | Batch label uses format `page_XXXX_b{N}` | High |
| 7 | Cells column contains the first element of the `cells` list (not the full list) | High |
| 8 | Region carry-forward works across batch-record boundaries | High |
| 9 | Records sorted by `(col, sec)` within each page before processing | High |
| 10 | Hallucination guard (skip batch-records where any pair repeats ≥ 5 times) | High |
| 11 | HTML cleanup: control chars stripped, truncated HTML repaired | Medium |
| 12 | HTML entities unescaped in both region names and transcribed text | Medium |
| 13 | Master XLSX sorted by `(page_num, batch_num, col_num, sec_num)` | High |
| 14 | XLSX header row styled with bold white text on blue fill, row frozen at A2 | Medium |
| 15 | Column widths match spec (Stage 1 and Stage 2 differ) | Low |
| 16 | Empty-entries rows written (one row per region even when entries is empty) | Medium |
| 17 | Master XLSX in `BASE` directory, not inside the RTB_5 subdirectory | High |

**Scoring:** A solution that satisfies all Critical + High criteria and produces runnable, correct code should be considered a full pass. Missing Medium or Low criteria results in a partial score.

---

## 13. Common Mistakes to Penalise

| Mistake | Explanation |
|---------|-------------|
| Adding RTB_5 to the existing `RTB_DIRS` list in `batch_convert_all_rtb.py` only | The student must also update `merge_pages_to_master.py` (or write a unified script). Both stages must produce RTB_5 output. |
| Hardcoding `RTB_5` output path inside `results_RTB_5_Raab-Zygry/` for the master file | The master XLSX must be in `BASE`, matching the pattern of `master_RTB_1_*`, `master_RTB_2_*`, etc. |
| Not handling Format 1 because RTB_5 is "mostly" Format 2 | Both formats appear in RTB_5. Skipping Format 1 will silently drop data. |
| Using the full `cells` list as a string instead of `cells[0]` | The `cells` column must be the first element (e.g. `col1_sec01.png`), not `['col1_sec01.png']`. |
| Forgetting the region carry-forward | Each row must use the most recently seen non-empty region name, even if the current batch-record's region_name is empty. |
| Writing `page_XXXX_ocr.xlsx` to `BASE` instead of the subdirectory | Per-page XLSX files go inside `results_RTB_5_Raab-Zygry/`, co-located with the JSON files. |

---

## 14. Concrete Example: Single Page Trace

**Input:** `results_RTB_5_Raab-Zygry/page_0006_ocr.json`  
**Format:** Format 2 (all records in this file have `data` as a list of region dicts)

```
Record 0: batch="01", cells=["col1_sec01.png"]
  region_name=""  entries=[{extension:"2", text:"(Band V) Raab: Amtsg"}]

Record 1: batch="02", cells=["col1_sec02.png"]
  region_name="...fbr. = ...fabrik; Kfm. = Kaufmann"  entries=[]
```

**Expected XLSX rows for page_0006_ocr.xlsx:**

| batch | cells | region | phone_extension | transcribed_text |
|-------|-------|--------|----------------|-----------------|
| page_0006_b01 | col1_sec01.png | _(empty, no prior region)_ | 2 | (Band V) Raab: Amtsg |
| page_0006_b02 | col1_sec02.png | ...fbr. = ...fabrik; Kfm. = Kaufmann | _(empty)_ | _(empty)_ |

Note: Row 2 has an empty `phone_extension` and `transcribed_text` because the region has no entries. The region name is written to the region column and becomes the `last_region_name` for subsequent records.

---

## 15. How to Run the Reference Scripts (for orientation)

The existing reference scripts can be run for Volumes 1–4 as:

```bash
cd /Users/jestinroy/Monash/SoDa-Labs-Austrian_register_project/Reichstelefonbuch_1942/RTB_results_1_2_3_4/

# Stage 1: JSON → per-page XLSX (currently only processes RTB 1–4)
python batch_convert_all_rtb.py

# Stage 2: per-page XLSX → master XLSX (currently only processes RTB 1–4)
python merge_pages_to_master.py
```

A correct student solution extends this to produce equivalent output for RTB_5. The cleanest approach is a **new standalone script** (e.g. `process_RTB_5.py`) that imports or re-implements the Stage 1 `convert_page_ocr_to_xlsx()` function and the Stage 2 `merge_directory()` function with the RTB_5 config added.

---

*End of teacher reference document.*
