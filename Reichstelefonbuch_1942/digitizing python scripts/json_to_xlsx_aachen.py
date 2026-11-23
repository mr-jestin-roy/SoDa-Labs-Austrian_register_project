"""
json_to_xlsx_aachen.py
──────────────────────
Enhanced version for results_RTB_1_Aach-Berlichingen folder

Converts JSON OCR results → XLSX with detailed failure reporting

Handles:
  - JSON format variants (structured data with label/bbox)
  - Truncated HTML (missing closing tags)
  - Control characters (null bytes, etc.)
  - Valid HTML (normal path)

Columns: cell | region_name | phone_extension | transcribed_text

Run:
    IN_JSON=/path/to/page_0228_ocr.json \\
    OUT_XLSX=/path/to/page_0228_ocr.xlsx \\
    python json_to_xlsx_aachen.py

Requires:
    pip install openpyxl
"""

import html as html_lib
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    sys.exit("Missing: openpyxl  →  python -m pip install openpyxl")

# ── Paths (override via env vars for orchestration) ──────────────────────────
BASE      = Path(__file__).parent.parent   # Reichstelefonbuch_1942/
IN_JSON   = Path(os.getenv("IN_JSON",  BASE / "ocr_results_chandra.json"))
OUT_XLSX  = Path(os.getenv("OUT_XLSX", BASE / "ocr_results_chandra.xlsx"))

print("=" * 70)
print("▶ ENTERING: json_to_xlsx_aachen.py (Enhanced with Failure Tracking)")
print("=" * 70)
print(f"  IN_JSON:  {IN_JSON}")
print(f"  OUT_XLSX: {OUT_XLSX}")
print("-" * 70)

# ── Load JSON ─────────────────────────────────────────────────────────────────
with open(IN_JSON, encoding="utf-8") as f:
    records = json.load(f)

# Sort batches into strict section order (col1_sec01 → col1_sec02 → … → col4_secNN)
def _section_key(rec):
    cell = rec.get("cells", [""])[0]
    m = re.match(r'col(\d+)_sec(\d+)', cell)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

records = sorted(records, key=_section_key)

# ── HTML recovery parser ──────────────────────────────────────────────────────
_EXT_RE = re.compile(r'^(\(\d+\)|\d{1,6}[a-z]?)\s+(.+)$')


def _clean(raw: str) -> str:
    """Strip HTML tags then unescape HTML entities."""
    return html_lib.unescape(re.sub(r'<[^>]+>', '', raw)).strip()


def _strip_control(s: str) -> str:
    """Strip non-printable control characters that openpyxl cannot handle."""
    if not s:
        return s
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)


def _add_line(region: dict, raw_line: str) -> None:
    text = _clean(raw_line)
    if not text:
        return
    em = _EXT_RE.match(text)
    if em:
        region["entries"].append({"extension": em.group(1), "text": em.group(2).strip()})
    elif region["entries"]:
        region["entries"][-1]["text"] += " " + text


def _parse_list_group(content: str, region: dict) -> None:
    lis = list(re.finditer(r'<li[^>]*>(.*?)</li>', content, re.DOTALL))
    if lis:
        for li in lis:
            text = _clean(li.group(1))
            em = _EXT_RE.match(text)
            if em:
                region["entries"].append({"extension": em.group(1), "text": em.group(2).strip()})
            elif region["entries"] and text:
                region["entries"][-1]["text"] += " " + text
    else:
        # F33: List-Group with <p><br/> instead of <ul><li>
        for p_m in re.finditer(r'<p[^>]*>(.*?)</p>', content, re.DOTALL):
            for line in re.split(r'<br\s*/?>', p_m.group(1)):
                _add_line(region, line)


def _parse_table(content: str, region: dict) -> None:
    for row in re.finditer(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL):
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row.group(1), re.DOTALL)
        if len(cells) >= 2:
            ext_raw  = _clean(cells[0]).replace('\n', ' ').strip()
            text_raw = html_lib.unescape(re.sub(r'<[^>]+>', ' ', re.sub(r'<br\s*/?>', ' ', cells[1])))
            text_raw = re.sub(r'\s+', ' ', text_raw).strip()
            ext_clean = ext_raw.strip('()')
            if re.match(r'^\d{1,6}$', ext_clean) and text_raw:
                ext_fmt = f"({ext_clean})" if ext_raw.startswith('(') else ext_clean
                region["entries"].append({"extension": ext_fmt, "text": text_raw})


_SKIP_LABELS = {"Image", "Page-Header", "Page-Footer", "Table-Of-Contents"}


def _parse_html_to_regions(html: str) -> list:
    """Recover structured entries from a raw Chandra HTML blob."""
    results = []
    current = None

    # F55 / legacy Format A: Section-Header present → build one region per heading
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
            elif label == "List-Group":
                _parse_list_group(content, current)
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

    # No Section-Header: collect all into a single unnamed region.
    # First bold/plain text with no entries is treated as the region_name.
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
        elif label == "List-Group":
            _parse_list_group(content, current)

    if current["entries"] or current["region_name"]:
        results.append(current)
    return results


# ── Hallucination detection ───────────────────────────────────────────────────
REPEAT_THRESHOLD = 5

def is_hallucinated(entries: list) -> bool:
    """Return True if any single entry is repeated >= REPEAT_THRESHOLD times."""
    from collections import Counter
    counts = Counter((e.get("extension", ""), e.get("text", "")) for e in entries)
    return any(v >= REPEAT_THRESHOLD for v in counts.values())

# ── Enhanced failure tracking (categorized) ───────────────────────────────────
rows = []
skipped = {
    'format_variant_json': [],    # Structured JSON (label/bbox, not HTML)
    'truncated_html': [],         # Mismatched open/close tags
    'control_chars': [],          # Null bytes, etc.
    'parse_failed': [],           # HTML parsing failed
    'hallucination': [],          # Repeated entries detected
    'empty_result': []            # No entries extracted
}

last_region_name = ""
page_id = Path(IN_JSON).stem.replace("_ocr", "")

for rec in records:
    batch_num   = rec.get("batch", "")
    batch_label = f"{page_id}_b{batch_num}" if batch_num else page_id
    data        = rec.get("data", {})

    # ── Phase 2: Detect JSON format variant (structured data, not HTML) ────────
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and "label" in data[0] and "bbox" in data[0]:
            # This is structured OCR format (label/bbox), not raw HTML with text content
            skipped['format_variant_json'].append({
                'batch': batch_label,
                'item_count': len(data),
                'reason': 'Structured JSON format (label/bbox) - no text content to extract',
                'sample': str(data[0])[:100]
            })
            continue
        # Otherwise, treat as normal region list (proceed to region processing)

    # ── Phase 1: Strip control chars EARLY (before any HTML processing) ───────
    if isinstance(data, dict) and "raw" in data:
        raw_content = data["raw"]

        # Check if raw field contains JSON instead of HTML
        if raw_content.strip().startswith('[') or (raw_content.strip().startswith('{') and '"label"' in raw_content):
            # This is JSON format variant (label/bbox), not HTML
            skipped['format_variant_json'].append({
                'batch': batch_label,
                'reason': 'Raw field contains structured JSON (label/bbox) instead of HTML',
                'sample': raw_content[:80]
            })
            continue

        # Check for control characters
        control_chars = [c for c in raw_content if ord(c) < 32 and c not in '\t\n\r']
        if control_chars:
            skipped['control_chars'].append({
                'batch': batch_label,
                'chars_found': [hex(ord(c)) for c in set(control_chars)],
                'reason': 'Control characters detected and stripped'
            })
        # Strip control chars
        data["raw"] = _strip_control(raw_content)

    # ── Phase 3: Detect & Recover Truncated HTML ───────────────────────────────
    if isinstance(data, dict) and "raw" in data:
        raw_html = data["raw"]
        open_count = raw_html.count('<')
        close_count = raw_html.count('>')

        if open_count != close_count:
            diff = open_count - close_count

            # Attempt recovery only for small gaps (1-3 unclosed tags)
            if diff > 0 and diff <= 3:
                auto_closed = raw_html + ("</div>" * diff)
                try:
                    recovered = _parse_html_to_regions(auto_closed)
                    if recovered:
                        # Recovery succeeded
                        data = recovered
                        skipped['truncated_html'].append({
                            'batch': batch_label,
                            'open_tags': open_count,
                            'close_tags': close_count,
                            'recovered': True,
                            'reason': f'Truncated HTML recovered ({diff} missing closing tag(s))'
                        })
                    else:
                        # Recovery attempted but yielded no regions
                        skipped['truncated_html'].append({
                            'batch': batch_label,
                            'open_tags': open_count,
                            'close_tags': close_count,
                            'recovered': False,
                            'reason': f'Truncated HTML - recovery parse yielded no regions'
                        })
                        continue
                except Exception as e:
                    skipped['truncated_html'].append({
                        'batch': batch_label,
                        'open_tags': open_count,
                        'close_tags': close_count,
                        'recovered': False,
                        'error': str(e)[:100],
                        'reason': f'Truncated HTML - recovery parse error'
                    })
                    continue
            else:
                # Gap is too large or data is badly truncated
                skipped['truncated_html'].append({
                    'batch': batch_label,
                    'open_tags': open_count,
                    'close_tags': close_count,
                    'reason': f'Truncated HTML - gap too large ({diff} tags)'
                })
                continue

        # Normal HTML parsing (no truncation detected)
        recovered = _parse_html_to_regions(raw_html)
        if recovered:
            data = recovered
        else:
            skipped['parse_failed'].append({
                'batch': batch_label,
                'reason': 'HTML parsing failed / no regions extracted'
            })
            continue

    # ── Process regions into rows ──────────────────────────────────────────────
    regions = data if isinstance(data, list) else (
        [data] if isinstance(data, dict) and "region_name" in data else []
    )

    if not regions:
        skipped['empty_result'].append({
            'batch': batch_label,
            'reason': 'No regions to process'
        })
        continue

    batch_had_entries = False
    for region in regions:
        raw_name = html_lib.unescape(region.get("region_name", ""))
        if raw_name:
            last_region_name = raw_name
        section_header = last_region_name
        entries = region.get("entries", [])

        if entries and is_hallucinated(entries):
            skipped['hallucination'].append({
                'batch': batch_label,
                'entry_count': len(entries),
                'reason': f'Hallucination detected ({len(entries)} entries)'
            })
            continue

        if not entries:
            rows.append((batch_label, section_header, "", ""))
        else:
            batch_had_entries = True
            for entry in entries:
                rows.append((
                    batch_label,
                    section_header,
                    entry.get("extension", ""),
                    html_lib.unescape(entry.get("text", "")),
                ))

# ── Build workbook ────────────────────────────────────────────────────────────
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Reichstelefonbuch 1942"

# Header style
header_font  = Font(bold=True, color="FFFFFF")
header_fill  = PatternFill("solid", fgColor="2F5496")
header_align = Alignment(horizontal="center", vertical="center")

HEADERS = ["cell", "region_name", "phone_extension", "transcribed_text"]
for col, title in enumerate(HEADERS, start=1):
    cell = ws.cell(row=1, column=col, value=title)
    cell.font   = header_font
    cell.fill   = header_fill
    cell.alignment = header_align

# Data rows
for row_idx, (cell, region_name, extension_number, transcribed_text) in enumerate(rows, start=2):
    ws.cell(row=row_idx, column=1, value=_strip_control(cell))
    ws.cell(row=row_idx, column=2, value=_strip_control(region_name))
    ws.cell(row=row_idx, column=3, value=_strip_control(extension_number))
    ws.cell(row=row_idx, column=4, value=_strip_control(transcribed_text))

# Column widths
ws.column_dimensions["A"].width = 22   # cell (batch identifier)
ws.column_dimensions["B"].width = 30   # region_name
ws.column_dimensions["C"].width = 14   # phone_extension
ws.column_dimensions["D"].width = 60   # transcribed_text

# Freeze header row
ws.freeze_panes = "A2"

wb.save(OUT_XLSX)

# ── Generate failure report (Phase 5) ──────────────────────────────────────────
total_skipped = sum(len(v) for v in skipped.values())
total_batches = len(records)

print(f"\n▶ Processing Summary")
print("-" * 70)
print(f"  Total batches    : {total_batches}")
print(f"  Rows written     : {len(rows)}")
print(f"  Total skipped    : {total_skipped}")
print()

# Print failure categories
for category in sorted(skipped.keys()):
    failures = skipped[category]
    if failures:
        print(f"  {category}: {len(failures)}")

# Write failure report to JSON
if total_skipped > 0:
    failure_report = {
        'page_id': page_id,
        'total_batches': total_batches,
        'total_rows_written': len(rows),
        'total_skipped': total_skipped,
        'failures_by_category': {
            category: len(failures)
            for category, failures in skipped.items()
            if failures
        },
        'failure_details': {
            category: failures
            for category, failures in skipped.items()
            if failures
        }
    }

    failure_file = Path(str(OUT_XLSX).replace('.xlsx', '_failures.json'))
    with open(failure_file, 'w', encoding='utf-8') as f:
        json.dump(failure_report, f, indent=2, ensure_ascii=False)

    print(f"\n✔ Failure report  → {failure_file}")

print(f"\n✔ XLSX saved      → {OUT_XLSX}")
print("=" * 70)
print("✔ EXITING: json_to_xlsx_aachen.py (SUCCESS)")
print("=" * 70)


