# Yellow Pages from ReichTelefonBuch — Extraction Instructions

## What This Does

Scans all JSON files inside `RTB_results_1_2_3_4` for pages containing `(Branchenteil) NNN` references, and outputs a structured Excel file:

**Output:** `Yellow Pages from ReichTelefonBuch.xlsx`

---

## Prerequisites

Make sure Python 3 and `openpyxl` are installed:

```bash
pip3 install openpyxl
```

---

## Step 1 — Save the Script

Create a file called `extract_branchenteil.py` anywhere on your Mac (e.g. your Desktop):

```python
import json, glob, re, os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

PATTERN = re.compile(r'\(Branchenteil\)\s*(\d+)')
BASE = "/Users/jestinroy/Monash/SoDa-Labs-Austrian_register_project/Reichstelefonbuch_1942/RTB_results_1_2_3_4"

rows = []
folders = [f for f in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, f))]
if not folders:
    folders = ['']  # scan root if no subfolders

print(f"Found folders: {folders}")

for folder_name in sorted(folders):
    folder_path = os.path.join(BASE, folder_name) if folder_name else BASE
    json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    print(f"Scanning '{folder_name or BASE}': {len(json_files)} JSON files...")

    for json_file in json_files:
        filename = os.path.basename(json_file)
        page_match = re.search(r'page_(\d+)', filename)
        page_number = int(page_match.group(1)) if page_match else None

        try:
            with open(json_file, encoding="utf-8") as f:
                items = json.load(f)
        except Exception as e:
            print(f"  ERROR reading {filename}: {e}")
            continue

        if not isinstance(items, list):
            items = [items]

        for item in items:
            if "Branchenteil" not in json.dumps(item):
                continue

            batch = item.get("batch", "")
            cells = item.get("cells", [])
            cells_str = ", ".join(cells) if isinstance(cells, list) else str(cells)

            data = item.get("data", {})
            raw_html = ""
            if isinstance(data, dict):
                raw_html = data.get("raw", "")
            elif isinstance(data, list):
                for entry in data:
                    raw_html += (entry.get("raw", "") if isinstance(entry, dict) else json.dumps(entry)) + " "

            for m in PATTERN.finditer(raw_html):
                branchenteil_page = int(m.group(1))
                context_snippet = raw_html[max(0, m.start()-200):m.end()+50]
                context_clean = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', context_snippet)).strip()
                rows.append({
                    "folder_name": folder_name or os.path.basename(BASE),
                    "json_filename": filename,
                    "page_number": page_number,
                    "batch": batch,
                    "cells": cells_str,
                    "branchenteil_page_no": branchenteil_page,
                    "context": context_clean[:200],
                })

print(f"\nTotal matches found: {len(rows)}")

wb = Workbook()
ws = wb.active
ws.title = "Yellow Pages Index"
headers = [
    "Folder Name",
    "JSON Filename",
    "Page Number\n(from filename)",
    "Batch",
    "Cells",
    "Branchenteil\nPage No.",
    "Context (surrounding text)"
]
header_fill = PatternFill(start_color="FFD700", end_color="FFD700", fill_type="solid")
header_font = Font(bold=True, name="Calibri", size=11)
ws.append(headers)
for col_idx in range(1, len(headers)+1):
    cell = ws.cell(row=1, column=col_idx)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
ws.row_dimensions[1].height = 35

for row in rows:
    ws.append([
        row["folder_name"],
        row["json_filename"],
        row["page_number"],
        row["batch"],
        row["cells"],
        row["branchenteil_page_no"],
        row["context"]
    ])

for i, width in enumerate([22, 28, 16, 8, 40, 16, 60], start=1):
    ws.column_dimensions[get_column_letter(i)].width = width
ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions

output = "/Users/jestinroy/Monash/SoDa-Labs-Austrian_register_project/Reichstelefonbuch_1942/Yellow Pages from ReichTelefonBuch.xlsx"
wb.save(output)
print(f"\n✅ Saved: {output}")
```

---

## Step 2 — Run the Script

Open **Terminal** and run:

```bash
python3 ~/Desktop/extract_branchenteil.py
```

> Adjust the path if you saved the script somewhere other than your Desktop.

---

## Step 3 — Find the Output

The Excel file will be saved at:

```
/Users/jestinroy/Monash/SoDa-Labs-Austrian_register_project/Reichstelefonbuch_1942/Yellow Pages from ReichTelefonBuch.xlsx
```

---

## Output Columns Explained

| Column | Description |
|---|---|
| **Folder Name** | Subfolder inside `RTB_results_1_2_3_4` where the JSON was found |
| **JSON Filename** | e.g. `page_0491_ocr.json` |
| **Page Number** | Extracted from the filename (e.g. `491`) — the scanned image page |
| **Batch** | The `batch` field from the JSON item (e.g. `06`) |
| **Cells** | The `cells` captured in that batch (e.g. `col3_sec10.png, col3_sec11.png`) |
| **Branchenteil Page No.** | The number after `(Branchenteil)` — the **actual page in the physical Telefonbuch** |
| **Context** | Surrounding text showing the trade/category name (e.g. *Antiquitäten*, *Apotheken*) |

---

## JSON Structure Reference

Each `.json` file is a **list of batch items**. Each item looks like:

```json
{
  "batch": "06",
  "cells": ["col3_sec10.png", "col3_sec11.png"],
  "data": {
    "raw": "<div ...><p><b>Antiquitäten</b> (Branchenteil) 7</p></div>"
  }
}
```

The script searches the `raw` HTML field for the pattern `(Branchenteil) NNN` and records every match.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: openpyxl` | Run `pip3 install openpyxl` first |
| `FileNotFoundError` on BASE path | Check the `BASE` path in the script matches your actual folder |
| `0 matches found` | Confirm JSON files exist: `ls /Users/jestinroy/.../RTB_results_1_2_3_4/` |
