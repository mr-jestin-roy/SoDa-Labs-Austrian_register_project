# Region Column — Data Scientist Observations
## Reichstelefonbuch 1942, Volumes 1–5

---

## 1. What the `region` Column Is

The `region` column is a **carried-forward section header**. Every time the OCR pipeline encounters a new town heading on a page, it records that heading as the region and propagates it to every subsequent row until the next town heading is encountered. This means every subscriber row is already labelled with the town it belongs to — the data is pre-tagged at ingestion time.

Looking at the image (page 006, RTB Vol. 5), this maps directly to the bold town headings printed in the original book:

```
Raab (Oberdonau)        ← bold heading = region value
  12  Daxel Josef ...   ← subscriber rows carrying that region
  15  Pilstl, Karl ...
Raabs (Thaya)           ← next bold heading = new region value
  ...
```

This is confirmed in the data:

| batch | cells | region | phone_extension | transcribed_text |
|---|---|---|---|---|
| page_0006_b03 | col1_sec03.png | Raab (Oberdonau) | 12 | Daxel Josef Autounternahmer |
| page_0006_b03 | col1_sec03.png | Raab (Oberdonau) | 15 | Pilstl, Karl, Gastwirt |
| page_0006_b03 | col1_sec03.png | Raab (Oberdonau) | 28 | Parzer, Felix, Fleischerei |

---

## 2. Four Types of Values in the `region` Column

Across all five volumes (~1.95 million rows, ~57,000 unique region values) the values fall into four distinct categories:

### 2a. Clean Town Names ✅ *(the signal)*

These are the actual geographic town entries. They follow a consistent typographic structure inherited from the book's layout:

```
Town (AdminDistrict)
Town (AdminDistrict)[Province]
Town b. NearbyCity[Province]
Town üb. NearbyCity[Province]
```

Examples:
- `Raab (Oberdonau)`
- `Rackschütz (üb. Neumarkt)[Niederschlesien]`
- `Dransfeld (Hann.)`
- `Karlsbad (Sudetenland)`
- `Dragen (Elbe) [Hann.]`

Abbreviations used in the book:
- `b.` = *bei* (near)
- `üb.` = *über* (via / routed through)
- `Hann.` = Hannover, `Mecklb.` = Mecklenburg, `Rheinpr.` = Rheinprovinz, etc.

Approximate count of clean town names per volume (conservative regex filter):

| Volume | Unique region values | Est. clean town names |
|--------|---------------------|-----------------------|
| RTB 1 (Aach–Berlichingen) | 20,863 | ~3,980 |
| RTB 2 (Berlin–Dotterwies) | 7,332 | ~1,688 |
| RTB 3 (Drachhausen–Kaprun) | 8,415 | ~3,229 |
| RTB 4 (Karbow–Quittainen) | 9,562 | ~3,930 |
| RTB 5 (Raab–Zygry) | 11,967 | ~4,128 |

RTB 1 has the highest noise ratio because Aachen (a large city) spans many pages with dense sub-headings. RTB 2 (Berlin) is similarly noisy. RTBs 3–5 are smaller towns with cleaner heading structure.

---

### 2b. Page-Index Cross-Reference Lines ⚠️ *(navigational artefact)*

The printed book has running headers at the top of each page showing the **alphabetical range** covered on that page, e.g.:

```
Raab: Amtsgericht — Rackwitz: Schmidt
```

The OCR picks these up as section headers. They appear as region values containing `—` or `–` dashes, and often include `(Band X) N` page-number references:

```
richt — Rackwitz: Schmidt
Rackwitz: Schmidt — Radebe
Imann — Peter (Band V) 13
Karlsbad: Beitlich — Gewer
```

These are **not town names**. They are index labels spanning multiple actual towns. In RTB 5 alone, 9,253 rows carry these cross-reference strings as their region label — meaning those subscribers were extracted from OCR batches where the running header was misidentified as the section heading.

This is a significant contamination: **the subscriber rows under a cross-ref region label belong to one or more towns in that alphabetical range**, but we do not know which one without further disambiguation.

---

### 2c. Abbreviation/Legend Header Rows ℹ️ *(page boilerplate)*

Each section of the book begins with a legend row explaining abbreviations:

```
...fbr. = ...fabrik; Kfm. = Kaufmann; Kol.-W. = Kolonialwaren; ...
```

These appear a handful of times per volume (5–11 instances) and are easy to filter out by string prefix.

---

### 2d. Noise / OCR Fragments 🔴 *(artefacts)*

Everything else — partial words, street fragments, numbers, business names, \xa0-contaminated strings, multi-line OCR errors. Examples:

```
'4Burg'
'589 Jurk-Sirenen-Fabrik'
'landstraße 143'
'Rackwitz \xa0\xa0\xa0\xa0\xa0\xa0 b. Leipzig(Prov. Sachsen)'   ← valid town but with \xa0 padding
'VI \xa0\xa0 WUVUVIS...'
'n der Nute'
```

The `\xa0` (non-breaking space) contamination is particularly important: several otherwise-valid town names have `\xa0` padding between the town name and the district qualifier, inserted by the OCR because the book used wide typographic spacing in the heading. These are recoverable by normalising whitespace.

---

## 3. Key Structural Insight for Building the Town Corpus

### The ordering is already correct

Because the rows are sorted by `(page_num, batch_num, col_num, sec_num)`, the sequence of `region` transitions faithfully reproduces the **left-to-right, top-to-bottom, page-by-page reading order** of the printed book. The alphabetical order of towns is therefore preserved in the data — you do not need to re-sort to reconstruct the book's sequence.

A `region` transition point (where `region[i] != region[i-1]`) marks exactly where one town's subscriber block ends and the next begins.

There are **14,877 region transitions** in RTB 5 alone across 403,787 rows — giving an average block size of ~27 rows per region value.

### The carry-forward means no lookup needed

Every subscriber row already has its town label. To build a town corpus you do not need to join or interpolate — the tag is already on each row. The task reduces to:

1. **Identify which region values are genuine town names** (filter out types 2b, 2c, 2d above)
2. **Normalise the town name strings** (strip `\xa0`, collapse whitespace, standardise `b.`/`üb.` abbreviations)
3. **Group subscribers by normalised town name**

---

## 4. Proposed Filtering Strategy for Town Corpus Extraction

### Step 1 — Exclude obvious non-towns

```python
import re

EXCLUDE_PATTERNS = [
    r'—|–',                          # page-index cross-refs
    r'\(Band\s+[IVX]+\)',            # Band reference
    r'^\s*\.\.\.fbr\.',              # abbreviation legend
    r'^Allgemeine\b',                # "Allgemeine Abkürzungen" header
    r'^Abkürzungen',                 # same
    r'postalische Bezeichnungen',    # Berlin boilerplate
]

def is_non_town(s):
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, s):
            return True
    return False
```

### Step 2 — Normalise candidate town strings

```python
import unicodedata

def normalise_region(s):
    s = s.replace('\xa0', ' ')          # non-breaking space → regular space
    s = re.sub(r'\s+', ' ', s).strip() # collapse whitespace
    return s
```

### Step 3 — Validate as a town name

A clean town name has a predictable structure:

```python
TOWN_RE = re.compile(
    r'^[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß\-\s\.]+'   # capitalised town name
    r'(\s*(b\.|üb\.)\s+[A-ZÄÖÜ][^\[(\n]+)?'  # optional "b. City" / "üb. City"
    r'(\s*\([^)]+\))?'                      # optional (AdminDistrict)
    r'(\s*\[[^\]]+\])?$'                    # optional [Province]
)
```

### Step 4 — Extract the canonical town name component

From a normalised string like `Rackschütz (üb. Neumarkt)[Niederschlesien]`:

```python
def parse_town(s):
    # Split off parenthesised district and bracketed province
    m = re.match(r'^(.+?)(\s*\(.*)?(\s*\[.*)?$', s)
    base = m.group(1).strip() if m else s
    district = ...   # from group(2)
    province = ...   # from group(3)
    return base, district, province
```

---

## 5. Problem Cases that Require Manual Review or Heuristics

| Problem | Example | Impact |
|---------|---------|--------|
| **Page-index spans multiple towns** | `Rackwitz: Schmidt — Radebe` | Rows under this label could belong to Rackwitz, Radebeul, or a town in between |
| **`\xa0` padding** | `Rackwitz \xa0\xa0\xa0\xa0 b. Leipzig` | Normalisation fixes this; must do before matching |
| **OCR errors in town names** | `Raabes (Thaya) [Niederdonauf]` (should be Niederdonau) | Fuzzy matching or manual correction needed |
| **Town name as abbreviation** | `Raase üb. Freudenthal[Ostsudetenland]` | Valid town but needs \xa0 cleaning first |
| **Berlin special case (RTB2)** | Thousands of sub-section headings, not town names | Treat Berlin as a single entity; sub-headings are street/district labels |
| **Large city sub-entries** | `Karlsbad: Spar. — Karlsbrunn` | Cross-ref lines that carry forward across many subscriber rows |
| **Multi-line OCR** | `E\n        L\n        E\n        U\n        D` | Strip and ignore |
| **Typo in volume label** | `Raabes (Thaya) [Niederdonauf]` | Niederdonau(f) — trailing `f` is an OCR slip |

---

## 6. What a Correct Town Corpus Looks Like

After filtering and normalisation, each entry in the corpus should be:

```
{
  "canonical_name": "Rackwitz",
  "admin_district": "üb. Leipzig",
  "province": "Prov. Sachsen",
  "rtb_volume": 5,
  "page_range": [6, 8],          # pages this town appears on
  "subscriber_count": 47,        # non-empty phone_extension rows
  "first_batch": "page_0006_b03",
  "last_batch":  "page_0008_b12"
}
```

The key point: **the subscriber count and page range are directly derivable from the already-correctly-ordered master XLSX files without any further sorting or lookups**, because the carry-forward tagging is already done and the rows are already in reading order.

---

*Observations written 2026-05-28. Based on: master_RTB_1 through master_RTB_5, batch_convert_all_rtb.py, merge_pages_to_master.py.*
