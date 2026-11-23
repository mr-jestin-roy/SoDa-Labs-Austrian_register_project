"""
batch_convert_all_rtb.py
────────────────────────
Batch convert ALL page_XXXX_ocr.json files across four RTB directories to XLSX

Processes:
  - results_RTB_1_Aach-Berlichingen/
  - results_RTB_2_Berlin-Dotterwies/
  - results_RTB_3_Drachhausen-Kaprun/
  - results_RTB_4_Karbow-Quittainen/

Handles both JSON formats:
  - Format 1: data: { "raw": "<html>..." } → parse HTML table
  - Format 2: data: [{ "region_name": "", "entries": [...] }] → parse entries

Output: page_XXXX_ocr.xlsx (same directory as input JSON)
Columns: batch | cells | region | phone_extension | transcribed_text

Run:
    python batch_convert_all_rtb.py
"""

import html as html_lib
import json
import os
import re
import sys
import glob
from pathlib import Path
from collections import Counter

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    sys.exit("Missing: openpyxl  →  pip install openpyxl")

BASE = Path(__file__).parent

RTB_DIRS = [
    BASE / "results_RTB_1_Aach-Berlichingen",
    BASE / "results_RTB_2_Berlin-Dotterwies",
    BASE / "results_RTB_3_Drachhausen-Kaprun",
    BASE / "results_RTB_4_Karbow-Quittainen",
    BASE / "results_RTB_5_Raab-Zygry",
]

# ── Regex & helper functions ──────────────────────────────────────────────────
_EXT_RE = re.compile(r'^(\(\d+\)|\d{1,6}[a-z]?)\s+(.+)$')

def _clean(raw: str) -> str:
    """Strip HTML tags then unescape HTML entities."""
    return html_lib.unescape(re.sub(r'<[^>]+>', '', raw)).strip()

def _strip_control(s: str) -> str:
    """Strip non-printable control characters."""
    if not s:
        return s
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)

def is_hallucinated(entries: list) -> bool:
    """Return True if any entry repeated >= 5 times."""
    counts = Counter((e.get("extension", ""), e.get("text", "")) for e in entries)
    return any(v >= 5 for v in counts.values())

# ── HTML parsing functions ────────────────────────────────────────────────────
def _add_line(region: dict, raw_line: str) -> None:
    text = _clean(raw_line)
    if not text:
        return
    em = _EXT_RE.match(text)
    if em:
        region["entries"].append({"extension": em.group(1), "text": em.group(2).strip()})
    elif region["entries"]:
        region["entries"][-1]["text"] += " " + text

def _parse_table(content: str, region: dict) -> None:
    for row in re.finditer(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row.group(1), re.DOTALL)
        if len(cells) >= 2:
            ext_raw = _clean(cells[0]).replace('\n', ' ').strip()
            text_raw = html_lib.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<br\s*/?>', ' ', cells[1])))
            text_raw = re.sub(r'\s+', ' ', text_raw).strip()
            ext_clean = ext_raw.strip('()')
            if re.match(r'^\d{1,6}$', ext_clean) and text_raw:
                ext_fmt = f"({ext_clean})" if ext_raw.startswith('(') else ext_clean
                region["entries"].append({"extension": ext_fmt, "text": text_raw})

_SKIP_LABELS = {"Image", "Page-Header", "Page-Footer", "Table-Of-Contents"}

def _parse_html_to_regions(html: str) -> list:
    """Parse HTML to regions (Format 1 handler)."""
    results = []
    current = None

    # With Section-Header: one region per heading
    if re.search(r'data-label=["\']Section-Header["\']', html):
        for m in re.finditer(
            r'<div[^>]*data-label=["\']([^"\']+)["\'][^>]*>(.*?)(?:</div>|(?=<div\s|$))',
            html, re.DOTALL,
        ):
            label, content = m.group(1), m.group(2)
            if label == "Section-Header":
                heading = (re.search(r'<h[1-4][^>]*>(.*?)</h[1-4]>', content, re.DOTALL)
                           or re.search(r'<p[^>]*>(.*?)</p>', content, re.DOTALL))
                name = _clean(heading.group(1)) if heading else ""
                if current is not None:
                    results.append(current)
                current = {"region_name": name, "entries": []}
            elif label in _SKIP_LABELS or current is None:
                continue
            elif label in ("Text", "Table"):
                if re.search(r'<table', content, re.IGNORECASE):
                    _parse_table(content, current)
                else:
                    for p_m in re.finditer(r'<p[^>]*>(.*?)</p>', content, re.DOTALL):
                        for line in re.split(r'<br\s*/?>', p_m.group(1)):
                            _add_line(current, line)
        if current is not None:
            results.append(current)
        return results

    # No Section-Header: single region
    current = {"region_name": "", "entries": []}
    for m in re.finditer(
        r'<div[^>]*data-label=["\']([^"\']+)["\'][^>]*>(.*?)(?:</div>|(?=<div\s|$))',
        html, re.DOTALL,
    ):
        label, content = m.group(1), m.group(2)
        if label in _SKIP_LABELS:
            continue
        if label in ("Text", "Table"):
            if re.search(r'<table', content, re.IGNORECASE):
                _parse_table(content, current)
            else:
                for p_m in re.finditer(r'<p[^>]*>(.*?)</p>', content, re.DOTALL):
                    for line in re.split(r'<br\s*/?>', p_m.group(1)):
                        text = _clean(line)
                        if not text:
                            continue
                        em = _EXT_RE.match(text)
                        if em:
                            current["entries"].append({"extension": em.group(1), "text": em.group(2).strip()})
                        elif not current["entries"] and not current["region_name"]:
                            current["region_name"] = text
                        elif current["entries"]:
                            current["entries"][-1]["text"] += " " + text

    if current["entries"] or current["region_name"]:
        results.append(current)
    return results

# ── Main conversion function ──────────────────────────────────────────────────
def convert_page_ocr_to_xlsx(json_path: Path) -> dict:
    """Convert single page_XXXX_ocr.json to XLSX."""

    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    # Sort records by cells (col1_sec01 → col1_sec02 → col2_sec01, etc.)
    def _cell_sort_key(rec):
        cell = rec.get("cells", [""])[0] if rec.get("cells") else ""
        m = re.match(r'col(\d+)_sec(\d+)', cell)
        return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

    records = sorted(records, key=_cell_sort_key)

    rows = []
    stats = {
        'total_batches': len(records),
        'rows_written': 0,
        'format1_parsed': 0,
        'format2_parsed': 0,
        'skipped': 0,
        'errors': []
    }

    last_region_name = ""
    page_id = json_path.stem.replace("_ocr", "")

    for rec in records:
        batch_num = rec.get("batch", "")
        batch_label = f"{page_id}_b{batch_num}" if batch_num else page_id
        cells = rec.get("cells", [""])[0] if rec.get("cells") else ""
        data = rec.get("data", {})

        regions = []

        # ── FORMAT 1: data is dict with "raw" HTML ────────────────────────
        if isinstance(data, dict) and "raw" in data:
            raw_html = data["raw"]

            # Strip control chars
            control_chars = [c for c in raw_html if ord(c) < 32 and c not in '\t\n\r']
            if control_chars:
                raw_html = _strip_control(raw_html)

            # Recover truncated HTML
            open_count = raw_html.count('<')
            close_count = raw_html.count('>')
            if open_count > close_count:
                diff = open_count - close_count
                if diff <= 3:
                    raw_html += ("</div>" * diff)

            # Parse HTML
            try:
                regions = _parse_html_to_regions(raw_html)
                if regions:
                    stats['format1_parsed'] += 1
                else:
                    stats['skipped'] += 1
            except Exception as e:
                stats['errors'].append(f"{batch_label}: {str(e)[:60]}")
                stats['skipped'] += 1

        # ── FORMAT 2: data is list of regions with entries ────────────────
        elif isinstance(data, list) and data and isinstance(data[0], dict) and "entries" in data[0]:
            regions = data
            stats['format2_parsed'] += 1

        # ── Process regions into rows ────────────────────────────────────
        if regions:
            for region in regions:
                raw_name = html_lib.unescape(region.get("region_name", ""))
                if raw_name:
                    last_region_name = raw_name
                section_header = last_region_name
                entries = region.get("entries", [])

                # Skip hallucinated entries
                if entries and is_hallucinated(entries):
                    stats['skipped'] += 1
                    continue

                # Add rows
                if not entries:
                    rows.append((batch_label, cells, section_header, "", ""))
                else:
                    for entry in entries:
                        rows.append((
                            batch_label,
                            cells,
                            section_header,
                            entry.get("extension", ""),
                            html_lib.unescape(entry.get("text", "")),
                        ))
                    stats['rows_written'] += len(entries)
        else:
            stats['skipped'] += 1

    # ── Build XLSX ────────────────────────────────────────────────────────
    out_xlsx = json_path.parent / f"{json_path.stem}.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reichstelefonbuch"

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5496")
    header_align = Alignment(horizontal="center", vertical="center")

    HEADERS = ["batch", "cells", "region", "phone_extension", "transcribed_text"]
    for col, title in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col, value=title)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align

    for row_idx, (batch, cells, region, ext, text) in enumerate(rows, start=2):
        ws.cell(row=row_idx, column=1, value=_strip_control(batch))
        ws.cell(row=row_idx, column=2, value=_strip_control(cells))
        ws.cell(row=row_idx, column=3, value=_strip_control(region))
        ws.cell(row=row_idx, column=4, value=_strip_control(ext))
        ws.cell(row=row_idx, column=5, value=_strip_control(text))

    ws.column_dimensions["A"].width = 20  # batch
    ws.column_dimensions["B"].width = 18  # cells
    ws.column_dimensions["C"].width = 25  # region
    ws.column_dimensions["D"].width = 14  # phone_extension
    ws.column_dimensions["E"].width = 70  # transcribed_text

    ws.freeze_panes = "A2"
    wb.save(out_xlsx)

    stats['output_file'] = str(out_xlsx)
    return stats

# ── Main execution ────────────────────────────────────────────────────────────
print("=" * 80)
print("▶ BATCH CONVERT: All page_XXXX_ocr.json → XLSX across three RTB directories")
print("=" * 80)

total_files = 0
total_rows = 0
total_errors = 0

for rtb_dir in RTB_DIRS:
    if not rtb_dir.exists():
        print(f"\n✗ Directory not found: {rtb_dir}")
        continue

    print(f"\n▶ Processing: {rtb_dir.name}")
    print("-" * 80)

    json_files = sorted(glob.glob(str(rtb_dir / "page_*_ocr.json")))
    print(f"  Found {len(json_files)} JSON files")

    for json_path in json_files:
        json_path = Path(json_path)
        try:
            stats = convert_page_ocr_to_xlsx(json_path)
            total_files += 1
            total_rows += stats['rows_written']

            if stats['errors']:
                total_errors += len(stats['errors'])
                status = f"⚠ {stats['rows_written']} rows, {len(stats['errors'])} errors"
            else:
                status = f"✔ {stats['rows_written']} rows"

            print(f"  {json_path.name:30} → {status}")

        except Exception as e:
            total_errors += 1
            print(f"  {json_path.name:30} → ✗ {str(e)[:50]}")

print("\n" + "=" * 80)
print(f"▶ SUMMARY")
print("=" * 80)
print(f"  Files processed: {total_files}")
print(f"  Total rows:      {total_rows}")
print(f"  Errors:          {total_errors}")
print("=" * 80)
print("✔ COMPLETE")
print("=" * 80)
