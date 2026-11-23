# RTB Master XLSX Post-Processing Log

**Date:** 2026-06-01  
**Script:** `postprocess_master_xlsx.py`  
**Volumes:** RTB 1–5 (all five master XLSX files)

---

## What Was Done

Each master XLSX file had four types of values in the `region` column (documented in `region_column_observations.md`). This script cleaned them in three steps before writing `*_clean.xlsx` outputs.

### Step 1 — Classify every region value

| Type | Description | Count (all 5 vols) |
|------|-------------|-------------------|
| **2a** Valid town name | Capitalised town, optional `(district)` / `[province]` / `b.`/`üb.` | 701,060 rows |
| **2b** Page-index cross-ref | Contains `—` or `(Band X)` — running page headers misread as headings | 439,010 rows |
| **2c** Page boilerplate | Abbreviation legends (`...fbr. = ...fabrik`), `Allgemeine Abkürzungen` | 21,482 rows |
| **2d** Noise / OCR garbage | Fragments, special chars, business names, ads, person initials | 535,784 rows |
| **empty** | No region at all (page-boundary carry gap) | 254,042 rows² |

²The empty rows were fixable because the master is sorted in reading order — the previous page's last heading carries forward correctly via trickle-down.

---

### Step 2 — Extract 2c boilerplate to `page_boiler_RTB_X_*.xlsx`

Pure header rows (no phone_extension, no transcribed_text) whose region was a 2c boilerplate string were extracted to a separate file per volume. These include abbreviation legend rows and Berlin postal notation headers.

| Volume | Boilerplate rows extracted |
|--------|--------------------------|
| RTB 1 | 13 |
| RTB 2 | 494 |
| RTB 3 | 140 |
| RTB 4 | 5 |
| RTB 5 | 5 |
| **Total** | **657** |

RTB 2 (Berlin) has the highest count because Berlin's section opens with extensive abbreviation and postal-district legend rows.

---

### Step 3 — Drop ghost header rows (2b / 2d / empty, no subscriber data)

Pure header rows with no subscriber data (empty phone_extension AND empty transcribed_text) that were 2b, 2d, or empty were removed. These are OCR artefacts — phantom section markers with no useful content.

| Volume | Ghost rows dropped |
|--------|--------------------|
| RTB 1 | 44,325 |
| RTB 2 | 6,043 |
| RTB 3 | 4,671 |
| RTB 4 | 5,015 |
| RTB 5 | 8,133 |
| **Total** | **68,187** |

RTB 1 (Aachen) has the highest count due to dense sub-section headers across Aachen's many pages.

---

### Step 4 — Global trickle-down

Walk all remaining rows in reading order. If a row's region is not a valid 2a town name (including subscriber rows sitting under 2b/2c/2d headings), replace it with the last valid town name seen above. This is correct because:

- The master XLSX is sorted by `(page_num, batch_num, col_num, sec_num)`.
- The previous page's last heading is therefore directly above the first row of the next page.
- No per-page reset — carry-forward is global across the entire volume.

| Volume | Regions corrected by trickle-down |
|--------|-----------------------------------|
| RTB 1 | 64,613 |
| RTB 2 | 266,734 |
| RTB 3 | 362,966 |
| RTB 4 | 339,654 |
| RTB 5 | 250,031 |
| **Total** | **1,283,998** |

Rows still empty after trickle-down (appeared before any heading in the entire volume): **18 rows removed**.

---

## Final Row Counts

| Volume | Original | Final (clean) | Removed |
|--------|----------|---------------|---------|
| RTB 1 (Aach–Berlichingen) | 182,530 | 138,192 | 44,338 |
| RTB 2 (Berlin–Dotterwies) | 356,671 | 350,119 | 6,552 |
| RTB 3 (Drachhausen–Kaprun) | 511,141 | 506,329 | 4,812 |
| RTB 4 (Karbow–Quittainen) | 498,181 | 493,160 | 5,021 |
| RTB 5 (Raab–Zygry) | 403,787 | 395,648 | 8,139 |
| **TOTAL** | **1,952,310** | **1,883,448** | **68,862** |

All 1,883,448 rows in the clean masters have a non-empty, valid region label.

---

## Output Files

| File | Description |
|------|-------------|
| `master_RTB_X_*_clean.xlsx` | Corrected master (use this for town corpus work) |
| `page_boiler_RTB_X_*.xlsx` | Extracted boilerplate rows (reference only) |

---

## Known Limitation

The `TOWN_RE` pattern accepts any string starting with a capital letter followed by letters/hyphens/spaces — this means some **advertisement sections** inside the RTB (e.g. `LEIPZIGER LEICHTMETALL-WERK RACKWITZ`) are classified as 2a and become trickle-down anchors. For most use cases this is acceptable: the advertisement heading is at least stable and localised. A future refinement could require the presence of a `(district)` or `[province]` qualifier to confirm genuine town status.

---

*Script:* `postprocess_master_xlsx.py` | *Observations:* `region_column_observations.md`
