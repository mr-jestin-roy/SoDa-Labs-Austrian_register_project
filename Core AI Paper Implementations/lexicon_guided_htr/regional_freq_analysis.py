"""Regional surname-frequency analysis over the RTB-1942 town↔surname graph.

This is the *analysis-only* prototype of the proposed Neo4j / Graph-RAG design.
It builds the same Country → State(≈Gau) → Town(Ort) → Surname graph the design
sketched, but in-memory (networkx) over the REAL build_lexicon.py artifacts, so
you can explore regional frequency distributions without standing up a server.

Two corrections to the original design, forced by what the real data actually
says (see `python regional_freq_analysis.py --demo`):

  1. There is no reliable Kreis layer in the RTB corpus — geography is
     Town → State only (state markers come from the directory headings and are
     themselves OCR, hence the fuzzy `same_state` match, reused from
     kg_retrieval).

  2. The "trust" signal is NOT raw regional count vs a fixed 50/200 cutoff.
     A town hapax is worth trusting when the name is *over-represented in the
     region relative to the nation* — i.e. regional LIFT — not when it happens
     to be nationally common. (The design's "Steiermark Hermann = 763" was in
     fact the national count; the true regional count is 3.)

Usage:
    python regional_freq_analysis.py --demo
    python regional_freq_analysis.py --region Steiermark --top 20
    python regional_freq_analysis.py --town "Pernegg (Steiermark)"
    python regional_freq_analysis.py --export-neo4j out_dir/   # CSV for neo4j-admin import
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import config

# ── corpus-consistent noise filter + state matcher ───────────────────────────
# Kept byte-for-byte in sync with kg_retrieval.py (inlined rather than imported
# so this analysis has no networkx dependency). If those rules change there,
# mirror them here.
NOISE_WORDS = {
    "Gendarmerie", "Notariat", "Bürgermeister", "Dass", "Dies", "Ders",
    "Kirche", "Pfarre", "Polizei", "Zollwache", "Finanz", "Magistrat",
    "Dr", "Ing", "Med", "Phil", "Jur", "Ww", "Wwe", "Fr", "Frau", "Wohn",
    "Wohnung", "Kfm", "Sen", "Jun", "Witwe", "Geschw", "Bauer", "Landwirt",
    "Poststelle", "Güterabfertigung",
}
NOISE_RE = re.compile(
    r"(amt|verwaltung|verband|posten|station|dienst|bahn(en)?|wehr|kasse|"
    r"schule|kirche|heim|werke?|schaft|stelle|anstalt|verein|lager|gericht|"
    r"betrieb|fabrik|haus|meisterei|kommando|leitung|zentrale|apotheke|"
    r"kommission|vorsteher|vorstehung|meister)$|"
    r"^(reichs|wehrmacht|partei|orts|kreis|landes|staats)", re.IGNORECASE)
_UMLAUT = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})


def is_noise(term: str) -> bool:
    return term in NOISE_WORDS or bool(NOISE_RE.search(term))


def _norm_state(s: str) -> str:
    return s.lower().translate(_UMLAUT).strip()


def same_state(a: str, b: str) -> bool:
    na, nb = _norm_state(a), _norm_state(b)
    if len(na) < 5 or len(nb) < 5:
        return na == nb
    return na in nb or nb in na


# ──────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────
def _load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Corpus:
    kg: dict           # town -> {surname: count}
    town_state: dict   # town -> raw state marker
    surnames: dict     # surname -> national RTB count
    # derived
    national_total: int

    @classmethod
    def load(cls) -> "Corpus":
        kg = _load(config.TOWN_SURNAME_KG_PATH)
        lex = _load(config.LEXICON_PATH)
        sn = lex["surnames"]
        return cls(
            kg=kg,
            town_state=lex["town_state"],
            surnames=sn,
            national_total=sum(sn.values()) or 1,
        )

    # -- geography -------------------------------------------------------
    def towns_in_region(self, region: str) -> list[str]:
        """All KG towns whose (OCR'd) state marker matches `region` under the
        same tolerant rule retrieval uses (case/diacritic/compound-insensitive)."""
        out = []
        for town in self.kg:
            st = self.town_state.get(town)
            if st and same_state(st, region):
                out.append(town)
        return out

    # -- frequency ------------------------------------------------------
    def regional_counts(self, towns: list[str], drop_noise: bool = True) -> collections.Counter:
        """Aggregate surname → count over a set of towns (the 2-hop roll-up)."""
        agg: collections.Counter = collections.Counter()
        for t in towns:
            for sn, c in self.kg.get(t, {}).items():
                if drop_noise and is_noise(sn):
                    continue
                agg[sn] += c
        return agg


# ──────────────────────────────────────────────────────────────────────────
# Trust metric
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class SurnameTrust:
    surname: str
    local: int          # count in the queried town
    regional: int       # count across the region
    national: int       # RTB-wide count
    regional_total: int
    region_share: float       # regional / regional_total
    national_share: float     # national / national_total
    lift: float               # region_share / national_share  (∞ ⇒ region-only)
    label: str

    def as_row(self) -> dict:
        return {
            "surname": self.surname,
            "local": self.local,
            "regional": self.regional,
            "national": self.national,
            "lift": round(self.lift, 1) if math.isfinite(self.lift) else float("inf"),
            "label": self.label,
        }


def classify(
    surname: str,
    local: int,
    regional: int,
    national: int,
    regional_total: int,
    national_total: int,
    lift_hi: float = 5.0,
    nat_floor: int = 50,
) -> SurnameTrust:
    """Honest trust label for a (town, surname) pair.

    The decision axis is regional LIFT — how over-represented the name is in the
    region vs the nation — guarded by the data-density caveat that killed the
    naive prior: a single regional mention proves nothing.
    """
    region_share = regional / regional_total if regional_total else 0.0
    national_share = national / national_total if national_total else 0.0
    lift = (region_share / national_share) if national_share > 0 else (
        math.inf if regional > 0 else 0.0
    )

    if regional <= 1 and national == 0:
        label = "Untrusted hapax (1 noisy pixel, no corroboration)"
    elif national >= nat_floor and lift < 2.0:
        # common nationwide, not regionally distinctive: a generic prior will
        # 'authenticate' it whether or not it's the right reading — low value.
        label = "Nationally common (weak regional evidence)"
    elif lift >= lift_hi and regional >= 2:
        label = "Regionally distinctive (trust-worthy)"
    elif regional >= 2:
        label = "Regionally attested (moderate)"
    else:
        label = "Thin regional signal"

    return SurnameTrust(
        surname, local, regional, national, regional_total,
        region_share, national_share, lift, label,
    )


# ──────────────────────────────────────────────────────────────────────────
# Analyses
# ──────────────────────────────────────────────────────────────────────────
def region_profile(corp: Corpus, region: str, top: int = 20) -> dict:
    towns = corp.towns_in_region(region)
    counts = corp.regional_counts(towns)
    total = sum(counts.values())
    hapax = sum(1 for c in counts.values() if c == 1)
    return {
        "region": region,
        "towns": len(towns),
        "distinct_surnames": len(counts),
        "total_mentions": total,
        "hapax_surnames": hapax,
        "hapax_fraction": round(hapax / len(counts), 3) if counts else 0.0,
        "max_count": counts.most_common(1)[0] if counts else None,
        "top": counts.most_common(top),
        "_counts": counts,
        "_total": total,
    }


def town_trust_table(corp: Corpus, town: str) -> list[SurnameTrust]:
    """Trust assessment for every surname attested in `town`, scored against
    its region and the nation."""
    region = corp.town_state.get(town)
    if region is None:
        raise KeyError(f"town not in town_state map: {town!r}")
    reg_towns = corp.towns_in_region(region)
    reg_counts = corp.regional_counts(reg_towns)
    reg_total = sum(reg_counts.values())

    rows = []
    for sn, local in corp.kg.get(town, {}).items():
        if is_noise(sn):
            continue
        rows.append(classify(
            sn, local,
            regional=reg_counts.get(sn, 0),
            national=corp.surnames.get(sn, 0),
            regional_total=reg_total,
            national_total=corp.national_total,
        ))
    rows.sort(key=lambda r: (r.lift if math.isfinite(r.lift) else 1e9), reverse=True)
    return rows


def build_graph(corp: Corpus, region: str | None = None):
    """Materialise the Country→State→Town→Surname graph (optionally one region).

    Optional helper for downstream graph algorithms; needs `networkx`."""
    import networkx as nx  # lazy: the profile/trust analyses don't need it
    G = nx.DiGraph()
    G.add_node("Deutsches Reich", kind="Country", year=1942)
    towns = corp.towns_in_region(region) if region else list(corp.kg)
    for town in towns:
        st = corp.town_state.get(town) or "?"
        gau = _norm_state(st) or "?"
        G.add_node(gau, kind="State")
        G.add_edge(gau, "Deutsches Reich", rel="PART_OF")
        G.add_node(town, kind="Town")
        G.add_edge(town, gau, rel="PART_OF")
        for sn, c in corp.kg.get(town, {}).items():
            if is_noise(sn):
                continue
            G.add_node(sn, kind="Surname", national=corp.surnames.get(sn, 0))
            G.add_edge(sn, town, rel="APPEARS_IN", count=c)
    return G


def export_neo4j(corp: Corpus, out: Path) -> None:
    """Emit node/edge CSVs with COMPOSITE town identity (name+state), which the
    original design's `UNIQUE(Ort.name)` constraint would have wrongly collapsed."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "states.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["gauId:ID(Gau)", "name"])
        for g in sorted({_norm_state(s) for s in corp.town_state.values() if s}):
            w.writerow([g, g])
    with (out / "towns.csv").open("w", newline="") as f, \
         (out / "town_in_state.csv").open("w", newline="") as e:
        w = csv.writer(f); w.writerow(["ortId:ID(Ort)", "name", "state"])
        we = csv.writer(e); we.writerow([":START_ID(Ort)", ":END_ID(Gau)"])
        for town in corp.kg:
            st = corp.town_state.get(town) or "?"
            gau = _norm_state(st) or "?"
            tid = f"{town}|{gau}"            # composite ⇒ no name collisions
            w.writerow([tid, town, gau]); we.writerow([tid, gau])
    with (out / "appears_in.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow([":START_ID(Surname)", ":END_ID(Ort)", "count:int"])
        for town, sns in corp.kg.items():
            gau = _norm_state(corp.town_state.get(town) or "?") or "?"
            tid = f"{town}|{gau}"
            for sn, c in sns.items():
                if not is_noise(sn):
                    w.writerow([sn, tid, c])
    print(f"Wrote Neo4j import CSVs to {out}/")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def _print_trust(rows: list[SurnameTrust], limit: int = 30) -> None:
    print(f"{'surname':<22}{'loc':>4}{'reg':>5}{'natl':>7}{'lift':>8}  label")
    print("-" * 90)
    for r in rows[:limit]:
        lift = "inf" if not math.isfinite(r.lift) else f"{r.lift:.1f}"
        print(f"{r.surname[:21]:<22}{r.local:>4}{r.regional:>5}{r.national:>7}{lift:>8}  {r.label}")


def demo(corp: Corpus) -> None:
    print("=" * 78)
    print("DEMO — reality-check of the proposed Graph-RAG design on real RTB-1942 data")
    print("=" * 78)

    prof = region_profile(corp, "Steiermark", top=10)
    print(f"\n[Region profile] Steiermark")
    print(f"  towns matched           : {prof['towns']}")
    print(f"  distinct surnames       : {prof['distinct_surnames']}")
    print(f"  total mentions          : {prof['total_mentions']}")
    print(f"  hapax surnames          : {prof['hapax_surnames']} "
          f"({prof['hapax_fraction']:.0%} of distinct)")
    print(f"  most frequent           : {prof['max_count']}")
    print(f"  top-10                  : {prof['top']}")

    print("\n[Design claim check] 'Hermann in Steiermark = 763'")
    h_reg = prof["_counts"].get("Hermann", 0)
    h_nat = corp.surnames.get("Hermann", 0)
    print(f"  real REGIONAL count     : {h_reg}")
    print(f"  real NATIONAL count     : {h_nat}   <-- the design's 763 was national, not regional")

    print("\n[Town trust table] Pernegg (Steiermark) — the active research town")
    try:
        rows = town_trust_table(corp, "Pernegg (Steiermark)")
        _print_trust(rows, limit=15)
    except KeyError as e:
        print("  ", e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="reality-check on Steiermark/Pernegg")
    ap.add_argument("--region", help="print a region frequency profile")
    ap.add_argument("--town", help="print a town's per-surname trust table")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--export-neo4j", metavar="DIR", help="emit neo4j-admin import CSVs")
    args = ap.parse_args()

    corp = Corpus.load()

    if args.demo or not any([args.region, args.town, args.export_neo4j]):
        demo(corp)
    if args.region:
        prof = region_profile(corp, args.region, top=args.top)
        for k, v in prof.items():
            if not k.startswith("_"):
                print(f"{k:>18}: {v}")
    if args.town:
        _print_trust(town_trust_table(corp, args.town), limit=args.top)
    if args.export_neo4j:
        export_neo4j(corp, Path(args.export_neo4j))


if __name__ == "__main__":
    main()
