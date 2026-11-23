"""
raw_to_json.py
──────────────
Post-processor for OCR result files that contain Chandra's HTML markup
stored under data.raw instead of a parsed JSON object.

Chandra outputs structured HTML with data-label attributes:
  - Section-Header / Page-Header  → region_name
  - List-Group <li> items         → entries
  - Text <p> lines                → entries

This script:
  1. Reads every *_ocr.json in RESULTS_DIR
  2. For records with data.raw (HTML), parses and converts to {region_name, entries}
  3. Deduplicates hallucination loops (consecutive identical lines)
  4. Rewrites the JSON file in-place (originals backed up as *.bak.json)

Usage:
    python raw_to_json.py                         # process all files
    python raw_to_json.py page_0210_ocr.json      # single file
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("Missing: beautifulsoup4  →  pip install beautifulsoup4")

BASE        = Path(__file__).parent.parent
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE / "results"))

# Regex: leading extension token (digits, sometimes with letters/spaces like "212 4")
ENTRY_RE = re.compile(r"^([A-Z]?\s?\d[\d\s]{0,6})\s+(.+)$")


# ── HTML parser ───────────────────────────────────────────────────────────────

def deduplicate_loop(entries: list[dict]) -> list[dict]:
    """Remove consecutive identical (extension, text) pairs — hallucination loops."""
    if not entries:
        return entries
    out = [entries[0]]
    for e in entries[1:]:
        if e != out[-1]:
            out.append(e)
    return out


def parse_entry_line(line: str) -> dict | None:
    """Try to split 'EXTENSION  rest of text' from a single line."""
    line = line.strip()
    if not line:
        return None
    m = ENTRY_RE.match(line)
    if m:
        return {"extension": m.group(1).strip(), "text": m.group(2).strip()}
    # No leading number — treat whole line as continuation text with no extension
    return {"extension": "", "text": line}


def html_to_structured(raw_html: str) -> dict:
    """
    Parse Chandra's HTML output into {region_name, entries}.

    Handles two formats:
      • <div data-label="List-Group"><ul><li>ENTRY</li>...</ul></div>
      • <div data-label="Text"><p>LINE\nLINE\n...</p></div>
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # ── Region name ────────────────────────────────────────────────────────
    region_name = ""
    for label in ("Section-Header", "Page-Header"):
        el = soup.find("div", attrs={"data-label": label})
        if el:
            region_name = el.get_text(separator=" ", strip=True)
            # Strip bold markers left by html.parser
            region_name = re.sub(r"\s+", " ", region_name).strip()
            break

    # ── Entries ────────────────────────────────────────────────────────────
    entries = []

    for div in soup.find_all("div", attrs={"data-label": True}):
        label = div["data-label"]

        if label == "List-Group":
            for li in div.find_all("li"):
                parsed = parse_entry_line(li.get_text(strip=True))
                if parsed:
                    entries.append(parsed)

        elif label == "Text":
            for p in div.find_all("p"):
                for line in p.get_text().splitlines():
                    parsed = parse_entry_line(line)
                    if parsed:
                        entries.append(parsed)

        elif label == "Complex-Block":
            # Advertisements / company blocks — collect as a single entry
            text = div.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            if text:
                entries.append({"extension": "", "text": f"[BLOCK] {text}"})

    entries = deduplicate_loop(entries)
    return {"region_name": region_name, "entries": entries}


# ── File processor ────────────────────────────────────────────────────────────

def process_file(path: Path) -> tuple[int, int]:
    """
    Convert raw HTML records in one JSON file.
    Returns (converted_count, already_ok_count).
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    converted = 0
    already_ok = 0
    changed = False

    for rec in records:
        data = rec.get("data", {})
        if not isinstance(data, dict):
            already_ok += 1
            continue

        if "raw" not in data:
            # Already structured or non-raw failure
            if "entries" in data:
                already_ok += 1
            continue

        raw_html = data["raw"]
        if not raw_html.strip().startswith("<"):
            # Not HTML — leave as-is
            continue

        structured = html_to_structured(raw_html)
        rec["data"] = structured
        rec["converted_from_raw"] = True
        converted += 1
        changed = True

    if changed:
        # Backup original
        bak = path.with_suffix(".bak.json")
        if not bak.exists():
            path.rename(bak)
        else:
            path.unlink()  # already backed up, just overwrite
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {path.name}  converted={converted}  already_ok={already_ok}")
    else:
        print(f"  — {path.name}  nothing to convert  already_ok={already_ok}")

    return converted, already_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        targets = [RESULTS_DIR / a for a in sys.argv[1:]]
    else:
        targets = sorted(RESULTS_DIR.glob("*_ocr.json"))

    if not targets:
        sys.exit(f"No *_ocr.json files found in {RESULTS_DIR}")

    print(f"Processing {len(targets)} file(s) in {RESULTS_DIR}\n")
    total_converted = total_ok = 0

    for path in targets:
        if not path.exists():
            print(f"  ✗ Not found: {path}")
            continue
        c, ok = process_file(path)
        total_converted += c
        total_ok += ok

    print(f"\nDone.  Total converted: {total_converted}  Already structured: {total_ok}")


if __name__ == "__main__":
    main()
