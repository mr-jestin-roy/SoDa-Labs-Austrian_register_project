"""build_kg_v2.py — updated RTB-1942 knowledge graph from ALL 5 volumes.

Input : Matricula Österreich Mapping/RTB_mapped/master_RTB_*_mapped.csv
        (1,883,448 rows; region, Town, ReichsGau, gau_source, Phone_Number, raw_transcribed_text …)
Surname rule: build_lexicon.parse_entry (v1) PLUS an institution filter (Reichspost, Gendarmerie,
…-Apotheke, Reichs…, …erei); filtered rows still count in entry_starts.

Output (artifacts/kg_v2/):
  kg_region_surnames.json.gz   region  -> {surname: count}        (as v1, all volumes)
  kg_gau_surnames.json.gz      ReichsGau -> {surname: count}      (only rows whose Gau resolved)
  kg_region_meta.json.gz       region  -> {town, gau, gau_source, volume, rows}
  surname_corpus.tsv.gz        surname, count, n_regions, n_volumes   (sorted by count)
  entry_starts.tsv.gz          first comma field of every entry (surnames AND organisations), count
  stats.json                   row counts, distinct names at frequency thresholds, v1 vs v2
Run:  python build_kg_v2.py
"""
import gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from build_lexicon import parse_entry  # noqa: E402  (same surname heuristics as v1)

SRC = HERE.parent.parent / "Matricula Österreich Mapping" / "RTB_mapped"
OUT = HERE / "artifacts" / "kg_v2"
V1 = HERE / "artifacts" / "kg_town_surnames.json.gz"
VOL_RE = re.compile(r"master_RTB_(\d)_")

# v2 additions on top of v1's rule: institution / abbreviation tokens that v1 let through as
# "surnames" (found in the top of the v1 frequency list). They stay in entry_starts.tsv.gz.
EXTRA_NON_SURNAMES = {
    "Reichspost", "Bürgermeister", "Dass", "Gastst", "Gasth", "Gastw", "Dieselben", "Försterei",
    "Gendarmerie", "Wehrmacht", "Pfarrant", "Polizei", "Feuerwache", "Ortsgruppe", "Kreisleitung",
    "Landrat", "Landratsamt", "Oberförsterei", "Revierförsterei", "Ziegelei", "Gärtnerei", "Bäckerei",
    "Fleischerei", "Metzgerei", "Schmiede", "Tischlerei", "Brennerei", "Druckerei", "Wäscherei",
    "Kanzlei", "Bücherei", "Konditorei", "Schlachthof", "Kreissparkasse", "Stadtsparkasse",
}
INSTITUTION_PREFIX_RE = re.compile(r"^(Reichs|Kreis|Stadt|Landes|Gau|Orts|Amts)[a-zäöüß]{4,}")
INSTITUTION_HYPHEN_RE = re.compile(
    r"-(Apotheke|Drogerie|Theater|Garage|Hotel|Stube|Bau|Werk|Werke|Haus|Lichtspiele|Kino|Bank|"
    r"Buchhandlung|Hütte|Mühle|Brauerei|Fürsorge|Etage|Naturdünge|Kühler|Reifen)\b")


def is_person_surname(s):
    return not (s in EXTRA_NON_SURNAMES or INSTITUTION_PREFIX_RE.match(s)
                or INSTITUTION_HYPHEN_RE.search(s) or s.endswith("erei"))


def main():
    files = sorted(SRC.glob("master_RTB_*_mapped.csv"))
    assert len(files) == 5, files
    OUT.mkdir(parents=True, exist_ok=True)

    kg = defaultdict(Counter); gau_kg = defaultdict(Counter)
    meta = {}; region_rows = Counter()
    surn = Counter(); surn_regions = defaultdict(set); surn_vols = defaultdict(set)
    starts = Counter()
    st = {"rows_total": 0, "rows_per_volume": {}, "rows_with_region": 0, "rows_with_text": 0,
          "rows_with_surname": 0, "rows_gau_resolved": 0}

    for f in files:
        vol = int(VOL_RE.search(f.name).group(1))
        df = pd.read_csv(f, dtype=str, usecols=["region", "Town", "ReichsGau", "gau_source",
                                                "raw_transcribed_text"], keep_default_na=False)
        st["rows_total"] += len(df); st["rows_per_volume"][vol] = len(df)
        print(f"vol {vol}: {len(df):,} rows", flush=True)
        for region, town, gau, gsrc, text in df.itertuples(index=False):
            region = region.strip()
            if not region:
                continue
            st["rows_with_region"] += 1
            region_rows[region] += 1
            if region not in meta:
                meta[region] = {"town": town, "gau": gau or None, "gau_source": gsrc, "volume": vol}
            text = text.strip()
            if not text:
                continue
            st["rows_with_text"] += 1
            first = re.sub(r"\s+", " ", text.split(",")[0]).strip(" .")
            if first:
                starts[first[:80]] += 1
            s, _, _ = parse_entry(text)
            if not s:
                continue
            if not is_person_surname(s):
                st["rows_institution_filtered"] = st.get("rows_institution_filtered", 0) + 1
                continue
            st["rows_with_surname"] += 1
            kg[region][s] += 1
            surn[s] += 1; surn_regions[s].add(region); surn_vols[s].add(vol)
            if gau and gsrc in ("rtb_qualifier", "matricula_match"):
                gau_kg[gau][s] += 1; st["rows_gau_resolved"] += 1

    for r in meta:
        meta[r]["rows"] = region_rows[r]

    def dump(name, obj):
        with gzip.open(OUT / name, "wt", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)

    dump("kg_region_surnames.json.gz", {r: dict(c) for r, c in kg.items()})
    dump("kg_gau_surnames.json.gz", {g: dict(c) for g, c in gau_kg.items()})
    dump("kg_region_meta.json.gz", meta)
    with gzip.open(OUT / "surname_corpus.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("surname\tcount\tn_regions\tn_volumes\n")
        for s, c in surn.most_common():
            fh.write(f"{s}\t{c}\t{len(surn_regions[s])}\t{len(surn_vols[s])}\n")
    with gzip.open(OUT / "entry_starts.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("entry_start\tcount\n")
        for s, c in starts.most_common():
            fh.write(f"{s}\t{c}\n")

    st.update({
        "regions": len(meta), "regions_with_surnames": len(kg),
        "region_surname_pairs": sum(len(c) for c in kg.values()),
        "gaue": sorted(gau_kg), "distinct_surnames": len(surn),
        "distinct_surnames_freq_ge": {k: sum(v >= k for v in surn.values()) for k in (2, 3, 5, 10, 50)},
        "surname_token_coverage_freq_ge": {k: round(sum(v for v in surn.values() if v >= k) / max(1, sum(surn.values())), 4)
                                           for k in (2, 3, 5, 10, 50)},
        "distinct_entry_starts": len(starts),
        "distinct_entry_starts_freq_ge2": sum(v >= 2 for v in starts.values()),
    })
    if V1.exists():
        v1 = json.load(gzip.open(V1))
        v1_names = {s for c in v1.values() for s in c}
        st["v1"] = {"regions": len(v1), "region_surname_pairs": sum(len(c) for c in v1.values()),
                    "distinct_surnames": len(v1_names),
                    "v2_surnames_not_in_v1": len(set(surn) - v1_names),
                    "v1_surnames_not_in_v2": len(v1_names - set(surn))}
    pern = {r: dict(kg[r]) for r in kg if "pernegg" in r.lower()}
    st["pernegg_regions"] = {r: {"rows": region_rows[r], "surnames": pern.get(r, {})}
                             for r in meta if "pernegg" in r.lower()}
    json.dump(st, open(OUT / "stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in st.items() if k != "gaue"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
