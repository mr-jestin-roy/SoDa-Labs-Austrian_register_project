"""
json_to_xlsx.py
───────────────
Converts ocr_results_chandra.json → ocr_results_chandra.xlsx

Columns: cell | region_name | extension | text

Run:
    python json_to_xlsx.py

Requires:
    pip install openpyxl
"""

import json
import os
import sys
from pathlib import Path

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    sys.exit("Missing: openpyxl  →  python -m pip install openpyxl")

# ── Paths (override via env vars for orchestration) ──────────────────────────
BASE      = Path(__file__).parent.parent   # Reichstelefonbuch_1942/
IN_JSON   = Path(os.getenv("IN_JSON",  BASE / "ocr_results_chandra.json"))
OUT_XLSX  = Path(os.getenv("OUT_XLSX", BASE / "ocr_results_chandra.xlsx"))

print("=" * 60)
print("▶ ENTERING: json_to_xlsx.py")
print("=" * 60)
print(f"  IN_JSON:  {IN_JSON}")
print(f"  OUT_XLSX: {OUT_XLSX}")
print("-" * 60)

# ── Load JSON ─────────────────────────────────────────────────────────────────
with open(IN_JSON, encoding="utf-8") as f:
    records = json.load(f)

# ── Hallucination detection ───────────────────────────────────────────────────
REPEAT_THRESHOLD = 5   # if the same (extension, text) appears this many times → skip whole cell

def is_hallucinated(entries: list) -> bool:
    """Return True if any single entry is repeated >= REPEAT_THRESHOLD times."""
    from collections import Counter
    counts = Counter((e.get("extension", ""), e.get("text", "")) for e in entries)
    return any(v >= REPEAT_THRESHOLD for v in counts.values())

# ── Flatten to rows ───────────────────────────────────────────────────────────
rows = []
skipped = []

for rec in records:
    cell        = rec.get("cell", "")
    data        = rec.get("data", {})

    # Skip records where JSON parse failed (stored under "raw" key)
    if isinstance(data, dict) and "raw" in data:
        skipped.append((cell, "JSON parse failed / truncated response"))
        continue

    region_name = data.get("region_name", "") if isinstance(data, dict) else ""
    entries     = data.get("entries", [])     if isinstance(data, dict) else []

    # Skip cells with repeat-token hallucinations
    if entries and is_hallucinated(entries):
        skipped.append((cell, f"hallucination detected ({len(entries)} entries, repeated content)"))
        continue

    if not entries:
        rows.append((cell, region_name, "", ""))
    else:
        for entry in entries:
            rows.append((
                cell,
                region_name,
                entry.get("extension", ""),
                entry.get("text", ""),
            ))

# ── Build workbook ────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Reichstelefonbuch 1942"

# Header style
header_font  = Font(bold=True, color="FFFFFF")
header_fill  = PatternFill("solid", fgColor="2F5496")
header_align = Alignment(horizontal="center", vertical="center")

HEADERS = ["cell", "region_name", "extension", "text"]
for col, title in enumerate(HEADERS, start=1):
    cell = ws.cell(row=1, column=col, value=title)
    cell.font   = header_font
    cell.fill   = header_fill
    cell.alignment = header_align

# Data rows
for row_idx, (cell, region_name, extension, text) in enumerate(rows, start=2):
    ws.cell(row=row_idx, column=1, value=cell)
    ws.cell(row=row_idx, column=2, value=region_name)
    ws.cell(row=row_idx, column=3, value=extension)
    ws.cell(row=row_idx, column=4, value=text)

# Column widths
ws.column_dimensions["A"].width = 22   # cell
ws.column_dimensions["B"].width = 30   # region_name
ws.column_dimensions["C"].width = 14   # extension
ws.column_dimensions["D"].width = 60   # text

# Freeze header row
ws.freeze_panes = "A2"

wb.save(OUT_XLSX)
print(f"Rows written : {len(rows)}")
print(f"Skipped      : {len(skipped)}")
for cell, reason in skipped:
    print(f"  ✗ {cell}: {reason}")
print(f"Saved        → {OUT_XLSX}")
print("=" * 60)
print("✔ EXITING: json_to_xlsx.py (SUCCESS)")
print("=" * 60)
