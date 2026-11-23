"""Break RTB-1942 transcribed_text into structured fields and tag each entry
with its Town and 1942 Reichsgau, using the Matricula parish mapping.

Input : RTB_master_source/master_RTB_*_clean.xlsx
        columns: batch, cells, region, phone_extension, transcribed_text
        (region = the town heading, e.g. "Pernegg (Steiermark)")
Output: RTB_mapped/master_RTB_*_mapped.csv
        columns: batch, cells, region, Town, ReichsGau, Diocese, gau_source,
                 Surname, First_Name, Profession, Street_Address, Phone_Number,
                 raw_transcribed_text

── Why two tiers for Reichsgau assignment ──────────────────────────────────
`output/parishes.csv` (built from Matricula) maps many parishes to a
*compound* reichsgau_1942 value (e.g. "Wien, Niederdonau" for 567 rows,
"Niederdonau, Steiermark" for 31) because a single diocese can straddle Gau
borders. A parish-name lookup alone therefore often can't resolve a single
Gau. But the RTB's OWN bracket/paren qualifier (e.g. "Pernegg (Steiermark)"
vs "Pernegg b. Hötzelndorf (Niederdonau)") already states the actual 1942
administrative Gau for towns inside the Ostmark — that's how the 1942
directory itself distinguished same-named towns. So:

  Tier 1 (gau_source="rtb_qualifier"): if the bracket/paren content names a
          known 1942 Reichsgau directly, use it. Most reliable — no fuzzy
          matching, comes straight from the source transcription.
  Tier 2 (gau_source="matricula_match"): only when tier 1 finds nothing
          (town has no recognisable Gau qualifier, e.g. many German-Reich
          towns proper). Look up the bracket-stripped town name against the
          parish index; if every matching parish agrees on one Gau, use it.
  gau_source="ambiguous"/"unmatched" otherwise — left for manual review
  rather than guessed.

── Surname/First_Name/Profession/Street split ──────────────────────────────
transcribed_text is comma-separated but the field count varies (2-12+ commas
seen in the corpus). Position alone is not reliable ("Geschw. Jäger,
Kunstmühle" has no first name at all). Instead: field[0] is always the
surname; each subsequent field is classified as First_Name if it matches a
token in this corpus's own given-names lexicon (built by build_lexicon.py
from this same OCR text, so it already contains the corpus's abbreviation
conventions: "Rich.", "Jos.", "Gust.", "Gg." ...), as Street_Address if it
looks like a street/house-number token, else Profession. This is a heuristic
over noisy 1940s OCR text — expect misclassifications on the long tail; it is
not a claim of ground truth.
"""
from __future__ import annotations

import collections
import csv
import difflib
import gzip
import json
import re
import sys
import time
from pathlib import Path

import openpyxl

HERE = Path(__file__).parent
PROJECT_ROOT = HERE.parent
SRC_DIR = HERE / "RTB_master_source"
OUT_DIR = HERE / "RTB_mapped"
PARISHES_CSV = HERE / "output" / "parishes.csv"
LEXICON_GZ = (PROJECT_ROOT / "Core AI Paper Implementations" / "lexicon_guided_htr"
              / "artifacts" / "lexicon.json.gz")

# Titles/prefixes that occasionally land in the given-names lexicon as noise
# (build_lexicon.py mines them from OCR text; a handful of high-frequency
# non-name tokens slip through). Excluded so "Dr." never masquerades as a
# first name.
NOT_A_FIRST_NAME = {
    "dr", "ing", "med", "phil", "jur", "ww", "wwe", "fr", "frau", "wohn",
    "wohnung", "kfm", "sen", "jun", "witwe", "geschw", "nr",
}

STREET_RE = re.compile(
    r"(str|straße|strasse|gasse|platz|weg|ring|allee)\b", re.IGNORECASE)
HOUSENUM_RE = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)

BRACKET_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")

_UMLAUT = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})


def norm(s: str) -> str:
    return s.strip().lower().translate(_UMLAUT)


# 1942 Reichsgau canonical names (Ostmark) + common abbreviation forms seen
# in RTB bracket qualifiers. Matched against normalized bracket content.
CANONICAL_GAU = {
    "wien": "Wien",
    "niederdonau": "Niederdonau",
    "nieder-donau": "Niederdonau",
    "niederdon": "Niederdonau",
    "oberdonau": "Oberdonau",
    "ober-donau": "Oberdonau",
    "oberdon": "Oberdonau",
    "steiermark": "Steiermark",
    "steierm": "Steiermark",
    "steiermk": "Steiermark",
    "karnten": "Kärnten",
    "karnt": "Kärnten",
    "salzburg": "Salzburg",
    "tirol": "Tirol-Vorarlberg",
    "vorarlberg": "Tirol-Vorarlberg",
    "tirol-vorarlberg": "Tirol-Vorarlberg",
}


# German-Reich Länder/Provinzen/regions that show up as RTB bracket
# qualifiers OUTSIDE the Ostmark. Seen directly in the corpus (Bay., Hann.,
# Westf., Sachs., Rheinpr., Oberschles., ...) plus their unabbreviated forms.
# A bracket matching one of these means "definitely not in Austria" — Tier 2
# (Matricula parish-name lookup) must NOT run for these, or a common town
# name (e.g. "Berg") coincidentally shared with an Austrian parish produces a
# false "ambiguous"/matched result for a town that was never in the Ostmark.
NON_AUSTRIA_MARKERS = {
    "bay", "bayern", "oberbay", "niederbay", "oberpfalz", "mittelfranken",
    "oberfranken", "unterfranken", "mainfranken", "franken",
    "baden", "wrttb", "wrtb", "wurttemberg", "hess-nass", "hessen-nassau",
    "hessen", "hess", "nassau", "pfalz", "saarpfalz", "saarland",
    "sachs", "sachsen", "prov sachsen", "prov sachs", "thur", "thuringen",
    "hann", "hannover", "westf", "westfalen", "rheinpr", "rheinprovinz",
    "rhein", "mark brandenburg", "brandenburg", "pomm", "pommern",
    "mecklb", "mecklenburg", "holstein", "schlesw-holst",
    "schleswig-holstein", "oberschles", "oberschlesien", "niederschles",
    "niederschlesien", "schlesien", "riesengeb", "erzgeb", "vogtl",
    "ostpr", "ostpreussen", "westpr", "westpreussen", "neumark", "danzig",
    "sudetenland", "sudetenl", "wartheland", "bohmen", "mahren",
    "protektorat", "elsass", "lothringen", "saale", "neckar", "harz",
    "donau",  # bare "Donau" (not Nieder-/Ober-donau) is a river-region label
    # used for various non-Ostmark towns, e.g. "an der Donau" in the Reich.
}


def _fuzzy_gau(token_norm: str) -> str | None:
    """Typo-tolerant Tier-1 match, e.g. OCR 'Niederdonaau'/'Niederdonauf' ->
    'Niederdonau'. Only applied to tokens that resemble a Gau in length."""
    best_gau, best_ratio = None, 0.0
    for key, gau in CANONICAL_GAU.items():
        if len(key) < 5 or abs(len(key) - len(token_norm)) > 2:
            continue
        ratio = difflib.SequenceMatcher(None, key, token_norm).ratio()
        if ratio > best_ratio:
            best_ratio, best_gau = ratio, gau
    return best_gau if best_ratio >= 0.85 else None


def match_gau_in_brackets(region: str) -> tuple[str | None, bool]:
    """Tier 1. Returns (gau_or_None, saw_non_austria_marker)."""
    saw_non_austria = False
    for token in BRACKET_RE.findall(region):
        t = norm(token).rstrip(".")
        if t in CANONICAL_GAU:
            return CANONICAL_GAU[t], saw_non_austria
        fuzzy = _fuzzy_gau(t)
        if fuzzy:
            return fuzzy, saw_non_austria
        if t in NON_AUSTRIA_MARKERS:
            saw_non_austria = True
    return None, saw_non_austria


def strip_brackets(region: str) -> str:
    return BRACKET_RE.sub("", region).strip()


# Vienna itself is never bracket-qualified in the RTB — it's written bare
# ("Wien", "Wien.", district numbers like "Wien VI"/"Wien. VIII"), or with a
# "Wien-<suburb>" / "Wien - <suburb>" prefix for Greater-Vienna exchange
# areas (e.g. "Wien-Mödling", "Wien-Atzgersdorf"), or as "...Stadt Wien" in
# directory-section headers. None of that has a bracket, so Tier 1's
# bracket-regex silently missed ~4,500+ Vienna-city rows in an earlier pass.
# Anchored word-boundary so it does NOT match "Wiener Neustadt" / "Wienerneu-
# dorf" / "Wienhausen" (unrelated towns that happen to start with "Wien").
WIEN_CITY_RE = re.compile(r"^wien\.?($|[\s\-])", re.IGNORECASE)
STADT_WIEN_RE = re.compile(r"\bstadt\s+wien\b", re.IGNORECASE)


# "Erläuterung der Vorortsabkürzungen" ("Glossary of the suburb
# abbreviations") is a running reference/glossary header repeated across many
# RTB pages, not a place. OCR truncates it inconsistently across occurrences
# ("Erl", "Erla", "Erlä", "Erläute", "Erläuter", ...), and two of those
# truncations ("Erla"/"Erlä") happen to normalize to the same key as the real
# Austrian parish "Erla" (Niederdonau) — which silently misattributed
# ~28,000 rows of glossary/index noise (bare page numbers as
# "transcribed_text", not real Surname/Firstname/Profession entries) to
# Niederdonau. Recognize and exclude exactly this phrase family (and its
# truncations) before any Gau resolution runs.
_BOILERPLATE_CANONICAL = (
    "erlauterung der vorortsabkurzungen",
    "erlauterung der orts- und vorortsabkurzungen",
    "erlauterung der ortsabkurzungen",
)


def is_boilerplate_region(region: str) -> bool:
    r = norm(region).rstrip(".").strip()
    if not r or len(r) < 2:
        return False
    return any(phrase.startswith(r) for phrase in _BOILERPLATE_CANONICAL)


def is_vienna_city(region: str) -> bool:
    r = region.strip()
    return bool(WIEN_CITY_RE.match(r)) or bool(STADT_WIEN_RE.search(r))


# ── Matricula parish index (Tier 2 fallback) ────────────────────────────────
def load_parish_index() -> dict[str, list[tuple[str, str]]]:
    idx: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    with open(PARISHES_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = norm(re.sub(r"/[a-z]+$", "", row["parish_name"], flags=re.I))
            idx[name].append((row["reichsgau_1942"].strip(), row["region_name"].strip()))
    return idx


def lookup_parish(town: str, idx: dict) -> tuple[str, str, str]:
    """Tier 2: base-town-name lookup. Returns (gau, diocese, source_tag).

    A parish's own reichsgau_1942 value can itself be compound (e.g. "Wien,
    Niederdonau" — the Vienna archdiocese straddles both Gaue), so we split
    every candidate's value on ',' before deciding whether the town resolves
    to a single Gau."""
    key = norm(town)
    candidates = idx.get(key)
    if not candidates:
        return "", "", "unmatched"
    gaus: set[str] = set()
    for g, _ in candidates:
        if not g or "N/A" in g:
            continue
        gaus.update(part.strip() for part in g.split(","))
    if len(gaus) == 1:
        gau = next(iter(gaus))
        dioceses = "; ".join(sorted({d for g, d in candidates if gau in g}))
        return gau, dioceses, "matricula_match"
    return ", ".join(sorted(gaus)), "", "ambiguous"


def resolve_gau(region: str, parish_idx: dict) -> tuple[str, str, str]:
    """Full tiered resolution for one `region` string -> (gau, diocese, source_tag)."""
    if is_boilerplate_region(region):
        return "", "", "boilerplate"
    if is_vienna_city(region):
        return "Wien", "", "rtb_qualifier"
    gau, saw_non_austria = match_gau_in_brackets(region)
    if gau:
        return gau, "", "rtb_qualifier"
    if saw_non_austria:
        # bracket names a German-Reich Land/Provinz explicitly -> definitely
        # not Ostmark; skip Matricula lookup to avoid a false match on a
        # shared town name.
        return "", "", "not_austria"
    return lookup_parish(strip_brackets(region), parish_idx)


# ── Given-names lexicon (drives First_Name detection) ───────────────────────
def load_given_names() -> set[str]:
    with gzip.open(LEXICON_GZ, "rt", encoding="utf-8") as fh:
        lex = json.load(fh)
    return {
        norm(n) for n, c in lex["given_names"].items()
        if norm(n) not in NOT_A_FIRST_NAME and len(n) >= 2
    }


def classify_field(field: str, given_names: set[str]) -> str:
    """'name' | 'street' | 'other' for one comma-separated transcribed_text field."""
    f = field.strip().rstrip(".")
    if not f:
        return "other"
    if HOUSENUM_RE.match(f) or STREET_RE.search(field):
        return "street"
    key = norm(f)
    if key in given_names:
        return "name"
    # abbreviation form, e.g. "Rich" -> "Richard": prefix match, min length 3
    if len(key) >= 3:
        for gn in given_names:
            if len(gn) > len(key) and gn.startswith(key):
                return "name"
    return "other"


def parse_transcribed_text(text: str, given_names: set[str]) -> tuple[str, str, str, str]:
    """-> (Surname, First_Name, Profession, Street_Address)"""
    if not text:
        return "", "", "", ""
    parts = [p.strip() for p in text.split(",")]
    surname = parts[0]
    first_name = ""
    profession_parts: list[str] = []
    street_parts: list[str] = []
    found_name = False
    for p in parts[1:]:
        kind = "name" if (not found_name and classify_field(p, given_names) == "name") else \
               classify_field(p, given_names)
        if kind == "name" and not found_name:
            first_name = p
            found_name = True
        elif kind == "street":
            street_parts.append(p)
        else:
            profession_parts.append(p)
    return surname, first_name, ", ".join(profession_parts), ", ".join(street_parts)


# ── Per-file processing ──────────────────────────────────────────────────────
FIELDNAMES = [
    "batch", "cells", "region", "Town", "ReichsGau", "Diocese", "gau_source",
    "Surname", "First_Name", "Profession", "Street_Address", "Phone_Number",
    "raw_transcribed_text",
]


def process_file(path: Path, out_path: Path, parish_idx: dict, given_names: set[str]) -> dict:
    stats = collections.Counter()
    gau_cache: dict[str, tuple[str, str, str]] = {}

    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows_iter = ws.iter_rows(min_row=2, values_only=True)

    with open(out_path, "w", newline="", encoding="utf-8") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for batch, cells, region, phone_extension, transcribed_text in rows_iter:
            stats["rows"] += 1
            region = region or ""

            if region not in gau_cache:
                gau_cache[region] = resolve_gau(region, parish_idx)
            gau, diocese, src = gau_cache[region]
            stats[src] += 1

            surname, first_name, profession, street = parse_transcribed_text(
                transcribed_text or "", given_names)

            writer.writerow({
                "batch": batch, "cells": cells, "region": region,
                "Town": strip_brackets(region),
                "ReichsGau": gau, "Diocese": diocese, "gau_source": src,
                "Surname": surname, "First_Name": first_name,
                "Profession": profession, "Street_Address": street,
                "Phone_Number": phone_extension or "",
                "raw_transcribed_text": transcribed_text or "",
            })
    wb.close()
    return stats


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    parish_idx = load_parish_index()
    given_names = load_given_names()
    print(f"Loaded {len(parish_idx)} normalized parish-name keys, "
          f"{len(given_names)} given-name tokens.\n")

    files = sorted(SRC_DIR.glob("master_RTB_*_clean.xlsx"))
    if not files:
        print(f"No source files found in {SRC_DIR}", file=sys.stderr)
        sys.exit(1)

    grand_total = collections.Counter()
    for f in files:
        out_path = OUT_DIR / (f.stem.replace("_clean", "") + "_mapped.csv")
        t0 = time.time()
        stats = process_file(f, out_path, parish_idx, given_names)
        grand_total.update(stats)
        dt = time.time() - t0
        n = stats["rows"] or 1
        print(f"{f.name}")
        print(f"  -> {out_path.name}  ({n} rows, {dt:.1f}s)")
        for tag in ("rtb_qualifier", "matricula_match", "ambiguous", "not_austria", "boilerplate", "unmatched"):
            print(f"     {tag:16s}: {stats[tag]:7d}  ({stats[tag]/n:.1%})")
        print()

    n = grand_total["rows"] or 1
    print("=" * 60)
    print(f"TOTAL rows processed: {n}")
    for tag in ("rtb_qualifier", "matricula_match", "ambiguous", "unmatched"):
        print(f"  {tag:16s}: {grand_total[tag]:8d}  ({grand_total[tag]/n:.1%})")


if __name__ == "__main__":
    main()
