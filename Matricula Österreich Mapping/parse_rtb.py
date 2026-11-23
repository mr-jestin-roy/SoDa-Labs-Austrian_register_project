"""
Parse RTB (Reichstelefonbuch 1942) master xlsx files into structured columns.

Input columns:  batch, cells, region, phone_extension, transcribed_text
Output columns: batch, cells, region, town, region_tag_1942, surname, first_name,
                 profession, street_address, phone_number, entry_type, transcribed_text

Parsing rules (validated against samples from RTB_1 and RTB_4):

region column -> town / region_tag_1942
  The LAST bracket-or-paren group in the string is the administrative tag
  (Land / Provinz / Reichsgau, e.g. "[Baden]", "(Steiermark)"). Any earlier
  parenthetical (e.g. "(Hegau)" in "Aach (Hegau) [Baden]") is a locality
  sub-descriptor and stays part of the town name.
  For the 7 Austrian Reichsgaue, OCR spelling variants are normalized to a
  canonical name (e.g. "Niederdonauf", "Niederdonaau" -> "Niederdonau").
  Non-Austrian tags (Baden, Bayern, Hann., ...) are kept as raw text --
  strictly most of these are "Land"/"Provinz" in 1942 usage, not "Reichsgau",
  but they go in this column for a consistent single lookup field.

transcribed_text -> surname / first_name / profession / street_address
  Comma-segment heuristic:
    >=3 segments: surname, first_name, profession, (address = remaining joined)
    2 segments:   business-keyword check on segment 0 ->
                    business match: profession=segment0, street_address=segment1
                    else:           surname=segment0, profession=segment1
    1 segment (no comma): unstructured -- kept as profession/business text only,
                    surname/first_name left blank (forcing a split here would
                    just be a guess, not an extraction)
  entry_type column records which branch was used, so ambiguous/unstructured
  rows are identifiable rather than silently mis-split.
"""
import concurrent.futures
import re
from pathlib import Path

import openpyxl

ROOT = Path(__file__).parent
IN_DIR = ROOT / "RTB_master_results"
OUT_DIR = ROOT / "RTB_parsed"

FILES = [
    "master_RTB_1_Aach-Berlichingen_clean.xlsx",
    "master_RTB_2_Berlin-Dotterwies_clean.xlsx",
    "master_RTB_3_Drachhausen-Kaprun_clean.xlsx",
    "master_RTB_4_Karbow-Quittainen_clean.xlsx",
    "master_RTB_5_Raab-Zygry_clean.xlsx",
]

# Canonical Austrian Reichsgau names + regex to catch OCR variants.
AUSTRIAN_REICHSGAU = [
    ("Wien", re.compile(r"^wien\b", re.I)),
    ("Niederdonau", re.compile(r"^niederdona", re.I)),
    ("Oberdonau", re.compile(r"^oberdona", re.I)),
    ("Steiermark", re.compile(r"^steiermark", re.I)),
    ("Kärnten", re.compile(r"^k[äa]rnten|^kaernten", re.I)),
    ("Salzburg", re.compile(r"^salzburg", re.I)),
    ("Tirol-Vorarlberg", re.compile(r"^tirol|^vorarlberg", re.I)),
]

# Business/institution keyword cues for 2-segment disambiguation.
BUSINESS_KEYWORDS = re.compile(
    r"\b(Gasthof|Gasthaus|Gemeinde|Genossenschaft|Verein|NSDAP|Molkerei|Spar-|"
    r"Darlehns|Bahnhofshotel|Hotel|Pension|Kaffee|Restaurant|Bäckerei|Sägewerk|"
    r"Fahrschule|Mühle|Kunstmühle|Pastorat|Gendarmerie|Forstrevier|Post\b|"
    r"Kirche|Schule|Verwaltung|AG\b|GmbH|eGmbH|Krankenhaus|Apotheke|Brauerei)",
    re.I,
)

TAG_RE = re.compile(r"[\(\[]([^\(\)\[\]]+)[\)\]]")

# Groß-Wien suburb entries carry no bracket/paren tag at all -- the RTB writes
# them as "Wien-<Ortsteil>", "Wien", or "Stadt Wien" directly (the "Wien-"
# prefix itself *is* the province tag, confirmed against genuinely separate
# towns like "Wiener Neustadt (Niederdonau)" which do carry their own tag and
# so never reach this fallback). \b after "Wien" avoids false positives like
# "Wienerwald"/"WIENER HEIM" (no word boundary between "Wien" and "er").
WIEN_PREFIX_RE = re.compile(r"^(?:Stadt\s+)?Wien\b[\s\-]*(.*)$", re.I)


def normalize_reichsgau_tag(raw_tag: str) -> str:
    raw_tag = raw_tag.strip()
    for canonical, pattern in AUSTRIAN_REICHSGAU:
        if pattern.match(raw_tag):
            return canonical
    return raw_tag


def parse_region(region: str):
    if not region:
        return "", ""
    matches = list(TAG_RE.finditer(region))
    if matches:
        last = matches[-1]
        town = region[: last.start()].strip()
        tag = normalize_reichsgau_tag(last.group(1))
        return town, tag

    m = WIEN_PREFIX_RE.match(region.strip())
    if m:
        suburb = m.group(1).strip()
        return (suburb if suburb else "Wien"), "Wien"

    return region.strip(), ""


def parse_transcribed_text(text: str):
    if not text:
        return "", "", "", "", "unstructured"
    segments = [s.strip() for s in text.split(",")]
    segments = [s for s in segments if s != ""]

    if len(segments) >= 3:
        surname, first_name, profession = segments[0], segments[1], segments[2]
        street_address = ", ".join(segments[3:])
        return surname, first_name, profession, street_address, "personal_3plus"

    if len(segments) == 2:
        if BUSINESS_KEYWORDS.search(segments[0]):
            return "", "", segments[0], segments[1], "business_2field"
        return segments[0], "", segments[1], "", "personal_2field"

    # 1 segment, no comma -- can't safely split into surname/first_name.
    return "", "", segments[0] if segments else text.strip(), "", "unstructured"


def process_file(fname: str):
    in_path = IN_DIR / fname
    out_path = OUT_DIR / fname.replace("_clean.xlsx", "_parsed.xlsx")

    wb_in = openpyxl.load_workbook(in_path, read_only=True)
    ws_in = wb_in["Reichstelefonbuch"]

    wb_out = openpyxl.Workbook(write_only=True)
    ws_out = wb_out.create_sheet("Reichstelefonbuch")
    ws_out.append(
        [
            "batch", "cells", "region", "town", "region_tag_1942",
            "surname", "first_name", "profession", "street_address",
            "phone_number", "entry_type", "transcribed_text",
        ]
    )

    rows = ws_in.iter_rows(min_row=2, values_only=True)
    header = next(iter(ws_in.iter_rows(min_row=1, max_row=1, values_only=True)))
    n = 0
    for row in rows:
        batch, cells, region, phone_extension, transcribed_text = row[:5]
        town, region_tag = parse_region(region or "")
        surname, first_name, profession, street_address, entry_type = parse_transcribed_text(
            transcribed_text or ""
        )
        ws_out.append(
            [
                batch, cells, region, town, region_tag,
                surname, first_name, profession, street_address,
                phone_extension, entry_type, transcribed_text,
            ]
        )
        n += 1

    wb_in.close()
    OUT_DIR.mkdir(exist_ok=True)
    wb_out.save(out_path)
    return fname, n


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as ex:
        for fname, n in ex.map(process_file, FILES):
            print(f"{fname:45s} {n:8d} rows -> {fname.replace('_clean.xlsx','_parsed.xlsx')}")


if __name__ == "__main__":
    main()
