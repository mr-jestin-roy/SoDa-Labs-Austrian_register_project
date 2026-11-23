"""Stage B — geographically scoped lexicon retrieval over the town↔surname
knowledge graph (MegaRAG-style 1-hop / 2-hop neighbourhood expansion) plus
fuzzy candidate matching for draft OCR tokens (post-hoc edit-probability,
cf. Dictionary-Guided Scene Text Recognition, CVPR 2021).

Hop semantics for a parish query (e.g. "Althofen (Kärnten)"):
  1-hop : surnames attested in the parish town itself        (strongest prior)
  2-hop : surnames aggregated over all towns in the same state ("Kärnten")
  fallback: globally most frequent surnames
"""
import gzip
import json
import re
from dataclasses import dataclass, field

import networkx as nx
from rapidfuzz import fuzz, process

import config

# Retrieval-time noise filter — catches directory artefacts that survive the
# build-time heuristics, without requiring a corpus rebuild to tune.
NOISE_WORDS = {
    "Gendarmerie", "Notariat", "Bürgermeister", "Dass", "Dies", "Ders",
    "Kirche", "Pfarre", "Polizei", "Zollwache", "Finanz", "Magistrat",
    # not given names: titles, abbreviations, directory boilerplate
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


def is_noise(term: str) -> bool:
    return term in NOISE_WORDS or bool(NOISE_RE.search(term))


_UMLAUT = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "s"})


def _norm_state(s: str) -> str:
    return s.lower().translate(_UMLAUT).strip()


def same_state(a: str, b: str) -> bool:
    """Tolerant state-marker comparison: 'steiermark' == 'Steiermark' ==
    'Weststeiermark'; 'Karnten' == 'Kärnten'. RTB headings are OCR output
    themselves, so case/diacritic/compound variants are common."""
    na, nb = _norm_state(a), _norm_state(b)
    if len(na) < 5 or len(nb) < 5:
        return na == nb
    return na in nb or nb in na


@dataclass
class ScopedLexicon:
    """The candidate vocabulary handed to the VLM for one page."""
    parish: str
    matched_town: str | None
    state: str | None
    town_surnames: list[str] = field(default_factory=list)    # 1-hop
    state_surnames: list[str] = field(default_factory=list)   # 2-hop
    given_names: list[str] = field(default_factory=list)
    occupations: list[str] = field(default_factory=list)
    nearby_towns: list[str] = field(default_factory=list)
    # full (untruncated) state surname pool — used for fuzzy candidate
    # search only, never injected into the prompt wholesale
    state_surname_pool: list[str] = field(default_factory=list)

    @property
    def all_surnames(self) -> list[str]:
        seen, out = set(), []
        for s in self.town_surnames + self.state_surnames:
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out


class LexiconKG:
    def __init__(self, lexicon: dict, town_surnames: dict):
        self.lex = lexicon
        self.kg_raw = town_surnames  # full region string -> {surname: count}
        self.town_state = lexicon.get("town_state", {})
        # state -> aggregated surname counter, lazily built
        self._state_surnames: dict[str, dict] = {}
        # global fallback pool for fuzzy candidate search (never in prompts)
        self._global_surnames = [s for s in lexicon["surnames"]
                                 if not is_noise(s)]

    # ── loading ──────────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "LexiconKG":
        with gzip.open(config.LEXICON_PATH, "rt", encoding="utf-8") as fh:
            lexicon = json.load(fh)
        with gzip.open(config.TOWN_SURNAME_KG_PATH, "rt", encoding="utf-8") as fh:
            kg = json.load(fh)
        return cls(lexicon, kg)

    def as_networkx(self, towns: list[str] | None = None) -> nx.Graph:
        """Materialise (a slice of) the KG as a networkx graph for analysis
        or export to Neo4j; not needed on the hot path."""
        g = nx.Graph()
        for town in towns or self.kg_raw:
            g.add_node(town, kind="town", state=self.town_state.get(town))
            for surname, count in self.kg_raw.get(town, {}).items():
                g.add_node(surname, kind="surname")
                g.add_edge(town, surname, weight=count)
        return g

    # ── retrieval ────────────────────────────────────────────────────────────
    def match_town(self, query: str) -> str | None:
        """Fuzzy-resolve a parish name against full region headings."""
        hit = process.extractOne(query, self.kg_raw.keys(),
                                 scorer=fuzz.WRatio, score_cutoff=70)
        return hit[0] if hit else None

    def _surnames_for_state(self, state: str) -> dict:
        if state not in self._state_surnames:
            agg: dict[str, int] = {}
            for town, surnames in self.kg_raw.items():
                town_st = self.town_state.get(town)
                if town_st and same_state(town_st, state):
                    for s, c in surnames.items():
                        agg[s] = agg.get(s, 0) + c
            self._state_surnames[state] = agg
        return self._state_surnames[state]

    def scoped_lexicon(self, parish: str = config.DEFAULT_PARISH) -> ScopedLexicon:
        town = self.match_town(parish)
        state = self.town_state.get(town) if town else None

        def top(counter: dict, n: int) -> list[str]:
            ranked = sorted(counter.items(), key=lambda kv: -kv[1])
            return [k for k, _ in ranked if not is_noise(k)][:n]

        # merge heading variants that differ only by case/diacritics
        # ("Pernegg (Steiermark)" vs "Pernegg (steiermark)" are RTB OCR
        # artefacts of the same town)
        town_counts: dict[str, int] = {}
        if town:
            key = _norm_state(town)
            for heading, surnames in self.kg_raw.items():
                if _norm_state(heading) == key:
                    for s, c in surnames.items():
                        town_counts[s] = town_counts.get(s, 0) + c
        town_surnames = top(town_counts, config.TOP_TOWN_SURNAMES)
        if state:
            state_pool = self._surnames_for_state(state)
            state_surnames = top(state_pool, config.TOP_STATE_SURNAMES)
            nearby = [t for t, s in self.town_state.items()
                      if same_state(s, state)]
            nearby = sorted(nearby,
                            key=lambda t: -sum(self.kg_raw.get(t, {}).values()))[:40]
        else:
            state_pool = self.lex["surnames"]
            state_surnames = top(state_pool, config.TOP_STATE_SURNAMES)
            nearby = []

        return ScopedLexicon(
            parish=parish, matched_town=town, state=state,
            town_surnames=town_surnames,
            state_surnames=[s for s in state_surnames if s not in set(town_surnames)],
            given_names=top(self.lex["given_names"], config.TOP_GLOBAL_GIVEN_NAMES),
            occupations=top(self.lex["occupations"], config.TOP_GLOBAL_OCCUPATIONS),
            nearby_towns=nearby,
            state_surname_pool=[s for s in state_pool if not is_noise(s)],
        )

    # ── fuzzy candidate generation for draft tokens ──────────────────────────
    def candidates_for(self, token: str, scoped: ScopedLexicon,
                       top_k: int = config.FUZZY_TOP_K,
                       extra_pool: list[str] | None = None
                       ) -> list[tuple[str, float]]:
        """Rank lexicon entries by edit similarity to an uncertain draft token.
        1-hop surnames get a +5 score bonus so parish-local names win ties."""
        pools = [
            (scoped.town_surnames, 5.0),
            (scoped.state_surname_pool, 2.0),
            (scoped.given_names, 0.0),
            (scoped.nearby_towns, 0.0),
            (scoped.occupations, 0.0),
            (extra_pool or [], 0.0),
            (self._global_surnames, 0.0),   # nationwide fallback
        ]
        ranked: dict[str, float] = {}
        for pool, bonus in pools:
            if not pool:
                continue
            # plain Levenshtein ratio: WRatio's substring matching scores
            # short fragments ("Grab" for "Grabmayer") spuriously high
            for cand, score, _ in process.extract(
                    token, pool, scorer=fuzz.ratio,
                    score_cutoff=config.FUZZY_SCORE_CUTOFF, limit=top_k):
                if abs(len(cand) - len(token)) > max(2, 0.4 * len(token)):
                    continue
                ranked[cand] = max(ranked.get(cand, 0.0), score + bonus)
        return sorted(ranked.items(), key=lambda kv: -kv[1])[:top_k]


if __name__ == "__main__":
    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon()
    print(f"parish={scoped.parish!r} -> town={scoped.matched_town!r} "
          f"state={scoped.state!r}")
    print("1-hop town surnames:", scoped.town_surnames[:20])
    print("2-hop state surnames:", scoped.state_surnames[:20])
    print("candidates for 'Wieser':", kg.candidates_for("Wieser", scoped))
    print("candidates for garbled 'Hu6er':", kg.candidates_for("Hu6er", scoped))
