"""
postprocess_master_xlsx.py
──────────────────────────
Post-process all five master RTB XLSX files.

Four region value types (region_column_observations.md §2):
  2a  Clean town names          → keep; anchor the trickle-down
  2b  Page-index cross-refs     → subscriber rows kept, region fixed by trickle-down
  2c  Page boilerplate          → pure header rows extracted to page_boiler_RTB_X.xlsx
  2d  Noise / OCR garbage       → pure garbage header rows removed

"Pure header row" = phone_extension empty AND transcribed_text empty.
Subscriber rows (with real ext/text) are always kept regardless of their
current region label — trickle-down corrects the label.

Processing order (per volume):
  Step 1  Classify every row's region as 2a / 2b / 2c / 2d / empty
  Step 2  Extract pure-header 2c rows  → page_boiler_RTB_X.xlsx
  Step 3  Drop pure-header 2b / 2d / empty rows (ghost headers, no data)
  Step 4  Global trickle-down on remaining rows (reading order already correct)
          → invalid/empty region rows inherit last valid (2a) town name

Output (non-destructive, originals untouched):
  master_RTB_X_*_clean.xlsx        corrected master
  page_boiler_RTB_X_*.xlsx         extracted boilerplate rows

Run:
    python postprocess_master_xlsx.py
    python postprocess_master_xlsx.py --rtb 5
"""

import argparse
import re
import sys
from pathlib import Path

try:
    import pandas as pd
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
except ImportError:
    sys.exit("Missing: pandas openpyxl  →  pip install pandas openpyxl")

BASE = Path(__file__).parent

RTB_CONFIGS = [
    {"n": 1,
     "master":  BASE / "master_RTB_1_Aach-Berlichingen.xlsx",
     "clean":   BASE / "master_RTB_1_Aach-Berlichingen_clean.xlsx",
     "boiler":  BASE / "page_boiler_RTB_1_Aach-Berlichingen.xlsx"},
    {"n": 2,
     "master":  BASE / "master_RTB_2_Berlin-Dotterwies.xlsx",
     "clean":   BASE / "master_RTB_2_Berlin-Dotterwies_clean.xlsx",
     "boiler":  BASE / "page_boiler_RTB_2_Berlin-Dotterwies.xlsx"},
    {"n": 3,
     "master":  BASE / "master_RTB_3_Drachhausen-Kaprun.xlsx",
     "clean":   BASE / "master_RTB_3_Drachhausen-Kaprun_clean.xlsx",
     "boiler":  BASE / "page_boiler_RTB_3_Drachhausen-Kaprun.xlsx"},
    {"n": 4,
     "master":  BASE / "master_RTB_4_Karbow-Quittainen.xlsx",
     "clean":   BASE / "master_RTB_4_Karbow-Quittainen_clean.xlsx",
     "boiler":  BASE / "page_boiler_RTB_4_Karbow-Quittainen.xlsx"},
    {"n": 5,
     "master":  BASE / "master_RTB_5_Raab-Zygry.xlsx",
     "clean":   BASE / "master_RTB_5_Raab-Zygry_clean.xlsx",
     "boiler":  BASE / "page_boiler_RTB_5_Raab-Zygry.xlsx"},
]

# ── Region classifiers ────────────────────────────────────────────────────────

# 2b — page-index cross-references
_PAT_2B = [
    re.compile(r'[—–]'),
    re.compile(r'\(Band\s+[IVX]+\)'),
]

# 2c — page boilerplate (abbreviation legends, administrative headers)
_PAT_2C = [
    re.compile(r'^\s*\.\.\.fbr\.'),
    re.compile(r'^Allgemeine\b'),
    re.compile(r'^Abkürzungen'),
    re.compile(r'postalische Bezeichnungen'),
]

# 2a — valid town name structure
_TOWN_RE = re.compile(
    r'^[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß\-\s\.]+'
    r'(\s*(b\.|üb\.)\s+[A-ZÄÖÜ][^\[(\n]+)?'
    r'(\s*\([^)]+\))?'
    r'(\s*\[[^\]]+\])?$'
)

# Additional false-positive guards for 2a
_FP_GUARDS = [
    re.compile(r','),               # commas → subscriber or address data
    re.compile(r'^\w\.\s'),         # "A. Name" → person initial
    re.compile(r'K\.-G\.|A\.-G\.|GmbH|KG\b|OHG\b'),  # business-entity suffixes
    re.compile(r'^\w-\w'),          # "A-B" letter range
    re.compile(r'^[A-ZÄÖÜ]\s'),    # single capital + space → OCR artefact prefix
]


def normalise(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r'\s+', ' ', s.replace('\xa0', ' ')).strip()


def classify(raw) -> str:
    """Return '2a' | '2b' | '2c' | '2d' | 'empty'."""
    s = normalise(raw)
    if not s:
        return "empty"
    for p in _PAT_2C:
        if p.search(s):
            return "2c"
    for p in _PAT_2B:
        if p.search(s):
            return "2b"
    if _TOWN_RE.match(s) and not any(p.search(s) for p in _FP_GUARDS):
        return "2a"
    return "2d"


def is_pure_header(row) -> bool:
    """True when row carries no subscriber data."""
    def _empty(v) -> bool:
        s = normalise(str(v) if not isinstance(v, str) else v)
        return s == "" or s.lower() == "nan"
    return _empty(row.get("phone_extension", "")) and _empty(row.get("transcribed_text", ""))


# ── Trickle-down ──────────────────────────────────────────────────────────────

def trickle_down(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Walk rows in reading order (already sorted).
    Valid town (2a) → update last_valid, keep normalised value.
    Anything else   → inherit last_valid.
    Rows still empty after walk (before any heading) are dropped.
    """
    regions = []
    last_valid = ""
    replaced = 0
    unfillable = 0

    for raw in df["region"]:
        n = normalise(raw)
        cat = classify(n)
        if cat == "2a":
            last_valid = n
            regions.append(n)
        else:
            if last_valid:
                regions.append(last_valid)
                replaced += 1
            else:
                regions.append("")
                unfillable += 1

    df = df.copy()
    df["region"] = regions
    df = df[df["region"] != ""].reset_index(drop=True)
    return df, {"replaced": replaced, "unfillable": unfillable}


# ── XLSX writer ───────────────────────────────────────────────────────────────

def write_xlsx(df: pd.DataFrame, path: Path,
               header_colour: str = "1F4E78") -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reichstelefonbuch"

    hdr_font  = Font(bold=True, color="FFFFFF")
    hdr_fill  = PatternFill("solid", fgColor=header_colour)
    hdr_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

    headers = list(df.columns)
    for ci, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=ci, value=h)
        c.font, c.fill, c.alignment = hdr_font, hdr_fill, hdr_align

    for ri, row in enumerate(df.itertuples(index=False), 2):
        for ci, val in enumerate(row, 1):
            ws.cell(row=ri, column=ci, value=val)

    widths = {"batch": 22, "cells": 18, "region": 35,
              "phone_extension": 16, "transcribed_text": 80}
    for ci, h in enumerate(headers, 1):
        ws.column_dimensions[
            ws.cell(row=1, column=ci).column_letter
        ].width = widths.get(h, 20)

    ws.freeze_panes = "A2"
    wb.save(path)


# ── Per-volume processing ─────────────────────────────────────────────────────

def process(cfg: dict) -> dict:
    n      = cfg["n"]
    master = cfg["master"]
    clean  = cfg["clean"]
    boiler = cfg["boiler"]

    print(f"\n▶ RTB {n}  —  {master.name}")
    print("-" * 72)

    if not master.exists():
        print(f"  ✗ Not found: {master}")
        return {}

    print(f"  Reading … ", end="", flush=True)
    df = pd.read_excel(master, sheet_name=0, dtype=str)
    total = len(df)
    print(f"{total:,} rows")

    # Step 1 — classify
    df["_cat"]    = df["region"].apply(classify)
    df["_header"] = df.apply(is_pure_header, axis=1)

    cat_counts = df["_cat"].value_counts().to_dict()
    print(f"  Classification: 2a={cat_counts.get('2a',0):,}  "
          f"2b={cat_counts.get('2b',0):,}  "
          f"2c={cat_counts.get('2c',0):,}  "
          f"2d={cat_counts.get('2d',0):,}  "
          f"empty={cat_counts.get('empty',0):,}")

    # Step 2 — extract 2c pure-header rows → page_boiler
    boiler_mask = (df["_cat"] == "2c") & df["_header"]
    df_boiler   = df[boiler_mask].drop(columns=["_cat", "_header"])
    if len(df_boiler):
        write_xlsx(df_boiler, boiler, header_colour="7F6000")
        print(f"  2c boilerplate rows extracted : {len(df_boiler):,}  → {boiler.name}")
    else:
        print(f"  2c boilerplate rows extracted : 0")

    # Step 3 — drop pure-header rows that are 2b / 2d / empty (ghost headers)
    ghost_mask = df["_header"] & df["_cat"].isin(["2b", "2d", "empty"])
    n_ghost    = ghost_mask.sum()
    df = df[~(boiler_mask | ghost_mask)].drop(columns=["_cat", "_header"])
    print(f"  Ghost header rows dropped     : {n_ghost:,}  (2b/2d/empty, no subscriber data)")
    print(f"  Rows entering trickle-down    : {len(df):,}")

    # Step 4 — global trickle-down
    df, td_stats = trickle_down(df)
    print(f"  Regions fixed by trickle-down : {td_stats['replaced']:,}")
    print(f"  Rows removed (before heading) : {td_stats['unfillable']:,}")
    print(f"  Final rows in clean master    : {len(df):,}")

    write_xlsx(df, clean)
    print(f"  Written → {clean.name}")

    return {
        "volume": n,
        "original": total,
        "boiler_rows": len(df_boiler),
        "ghost_dropped": n_ghost,
        "td_replaced": td_stats["replaced"],
        "td_removed": td_stats["unfillable"],
        "final": len(df),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtb", type=int, choices=[1, 2, 3, 4, 5],
                        help="Process one volume (default: all 5)")
    args = parser.parse_args()

    configs = [c for c in RTB_CONFIGS if args.rtb is None or c["n"] == args.rtb]

    print("=" * 72)
    print("▶ POST-PROCESS: 2c extract + 2d drop + trickle-down → *_clean.xlsx")
    print("=" * 72)

    results = [r for cfg in configs if (r := process(cfg))]

    print("\n" + "=" * 72)
    print("▶ SUMMARY")
    print("=" * 72)
    g = {k: 0 for k in ("original", "boiler_rows", "ghost_dropped",
                         "td_replaced", "td_removed", "final")}
    for r in results:
        print(f"  RTB {r['volume']}: {r['original']:>8,} → {r['final']:>8,}"
              f"  (boiler {r['boiler_rows']:,} | ghost {r['ghost_dropped']:,}"
              f" | trickle {r['td_replaced']:,} | removed {r['td_removed']:,})")
        for k in g:
            g[k] += r[k]

    print(f"\n  Total original  : {g['original']:,}")
    print(f"  2c → page_boiler: {g['boiler_rows']:,}")
    print(f"  Ghost dropped   : {g['ghost_dropped']:,}")
    print(f"  Trickle-filled  : {g['td_replaced']:,}")
    print(f"  Removed (pre-hd): {g['td_removed']:,}")
    print(f"  Final rows      : {g['final']:,}")
    print("=" * 72)
    print("✔ COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
