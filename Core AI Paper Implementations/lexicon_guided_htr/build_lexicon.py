"""Stage A — build the domain lexicon + town↔surname knowledge graph
from the digitised Reichstelefonbuch 1942 master spreadsheets.

This is the "Reference PDF → Domain Vocabulary List" step of the
pre-OCR context-injection pipeline, with the MegaRAG twist that the
vocabulary is stored as an entity-relation structure (town —HAS_RESIDENT→
surname) instead of a flat list, so retrieval can pull a geographically
scoped 1-hop / 2-hop neighbourhood at inference time.

Input  : Reichstelefonbuch_1942/RTB_results_1_2_3_4/master_RTB_*_clean.xlsx
         (columns: batch, cells, region, phone_extension, transcribed_text)
Output : artifacts/lexicon.json.gz           {towns, states, surnames,
                                              given_names, occupations}
         artifacts/kg_town_surnames.json.gz  {town: {surname: count}}

Run:
    venv/bin/python build_lexicon.py
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

import config

# A word that looks like a German surname: capitalised, letters/umlauts/
# hyphen/apostrophe only, 2–30 chars.  Excludes obvious institution words.
NAME_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ]?[a-zäöüß]+)?$")
INSTITUTION_WORDS = {
    "Gasthof", "Gasthaus", "Gemeindeverwaltung", "Gemeinde", "Hotel",
    "Pension", "Cafe", "Café", "Kaffee", "Apotheke", "Bahnhof", "Post",
    "Postamt", "Pfarramt", "Pfarrhof", "Schule", "Volksschule", "Kloster",
    "Sparkasse", "Raiffeisenkasse", "Molkerei", "Brauerei", "Sägewerk",
    "Forstamt", "Zollamt", "Finanzamt", "Amtsgericht", "Krankenhaus",
    "Geschw", "Witwe", "Firma", "Der", "Die", "Das",
    # ditto / cross-reference forms used by the directory, not names
    "Ders", "Dies", "Derselbe", "Dieselbe", "Dasselbe", "Desgl", "Siehe",
}
# Institutional compounds: reject by suffix (Gemeindeamt, Reichsbahn,
# Gendarmerieposten, Ortsbauernschaft, Feuerwehr, …)
INSTITUTION_SUFFIX_RE = re.compile(
    r"(amt|verwaltung|verband|posten|station|dienst|bahn|wehr|kasse|"
    r"schule|kirche|heim|werke?|schaft|stelle|anstalt|verein|lager|"
    r"betrieb|fabrik|haus|mühle|meisterei|kommando|leitung|zentrale)$",
    re.IGNORECASE)
# Abbreviated or full given names: "Jos.", "Richard", "Karl"
GIVEN_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+\.?$")
# Last (…) or […] group in a region heading, e.g. "Velden (Wörther See)[Kärnten]"
STATE_RE = re.compile(r"[\(\[]([^()\[\]]+)[\)\]]\s*$")


def split_region(region: str):
    """'Aach (Hegau) [Baden]' -> ('Aach', 'Baden')"""
    region = str(region).strip()
    m = STATE_RE.search(region)
    state = m.group(1).strip() if m else None
    base = re.split(r"[\(\[]", region, maxsplit=1)[0].strip()
    return base, state


def parse_entry(text: str):
    """'Jäger, Richard, Viehandlung.' -> ('Jäger', ['Richard'], 'Viehandlung')"""
    parts = [p.strip(" .") for p in str(text).split(",")]
    parts = [p for p in parts if p]
    if not parts:
        return None, [], None
    first = parts[0]
    surname = None
    if (NAME_RE.match(first) and first not in INSTITUTION_WORDS
            and not INSTITUTION_SUFFIX_RE.search(first)):
        surname = first
    given = []
    if surname and len(parts) > 1:
        for tok in parts[1].split():
            if GIVEN_RE.match(tok) and tok.rstrip(".") not in INSTITUTION_WORDS:
                given.append(tok.rstrip("."))
            else:
                given = []
                break
    occupation = None
    if surname and len(parts) > 2:
        occupation = parts[-1]
        if not (2 < len(occupation) < 60) or any(c.isdigit() for c in occupation):
            occupation = None
    return surname, given, occupation


def main():
    files = sorted(config.RTB_MASTER_DIR.glob(config.RTB_MASTER_GLOB))
    if not files:
        sys.exit(f"No corpus files matching {config.RTB_MASTER_GLOB} "
                 f"in {config.RTB_MASTER_DIR}")

    towns, states = Counter(), Counter()
    surnames, given_names, occupations = Counter(), Counter(), Counter()
    town_state = {}
    kg = defaultdict(Counter)  # full region string -> surname counts

    for f in files:
        print(f"Reading {f.name} …", flush=True)
        df = pd.read_excel(f, usecols=["region", "transcribed_text"])
        df = df.dropna(subset=["region"])
        for region, text in zip(df["region"], df["transcribed_text"]):
            region = str(region).strip()
            base, state = split_region(region)
            towns[base] += 1
            if state:
                states[state] += 1
                town_state[region] = state
            if pd.isna(text):
                continue
            surname, given, occupation = parse_entry(text)
            if surname:
                surnames[surname] += 1
                kg[region][surname] += 1
            for g in given:
                given_names[g] += 1
            if occupation:
                occupations[occupation] += 1

    lexicon = {
        "towns": dict(towns),
        "states": dict(states),
        "town_state": town_state,
        "surnames": {k: v for k, v in surnames.items()
                     if v >= config.MIN_SURNAME_FREQ},
        "given_names": {k: v for k, v in given_names.items()
                        if v >= config.MIN_GIVEN_NAME_FREQ},
        "occupations": {k: v for k, v in occupations.items()
                        if v >= config.MIN_OCCUPATION_FREQ},
    }

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(config.LEXICON_PATH, "wt", encoding="utf-8") as fh:
        json.dump(lexicon, fh, ensure_ascii=False)
    with gzip.open(config.TOWN_SURNAME_KG_PATH, "wt", encoding="utf-8") as fh:
        json.dump({t: dict(c) for t, c in kg.items()}, fh, ensure_ascii=False)

    print(f"\nLexicon  → {config.LEXICON_PATH}")
    print(f"KG       → {config.TOWN_SURNAME_KG_PATH}")
    print(f"towns={len(towns):,}  surnames={len(lexicon['surnames']):,}  "
          f"given={len(lexicon['given_names']):,}  "
          f"occupations={len(lexicon['occupations']):,}  "
          f"kg towns={len(kg):,}")


if __name__ == "__main__":
    main()
