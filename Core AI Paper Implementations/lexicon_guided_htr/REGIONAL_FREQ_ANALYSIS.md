# Regional frequency analysis (Graph-RAG design, analysis-only prototype)

`regional_freq_analysis.py` is the analysis-only realisation of the proposed
Neo4j / Graph-RAG surname design, run over the **real** `build_lexicon.py`
artifacts (`artifacts/kg_town_surnames.json.gz`, `artifacts/lexicon.json.gz`)
in-memory — no Neo4j server required.

```bash
python regional_freq_analysis.py --demo                 # reality-check
python regional_freq_analysis.py --region Steiermark --top 20
python regional_freq_analysis.py --town "Pernegg (Steiermark)"
python regional_freq_analysis.py --export-neo4j out/    # optional CSVs for neo4j-admin
```

Only `--export-neo4j` and `build_graph()` need `networkx`; everything else is
stdlib.

## What the real data says (vs the original design)

| Design assumption | Reality in RTB-1942 |
|---|---|
| Hierarchy Ort → Kreis → Gau | No reliable Kreis layer; geography is **Town → State** only, and state markers are themselves OCR (hence the fuzzy `same_state` match). |
| `UNIQUE(Ort.name)` | Town names repeat ("Pernegg" appears 3×). Export uses **composite `name\|state` ids**. |
| "Hermann in Steiermark = 763" | **763 is the *national* count.** Real Steiermark regional count = **3**. The design conflated the two axes. |
| Raw regional count vs fixed 50/200 cutoff = trust | Fails on data density: **82% of Steiermark surnames are regional hapaxes**, max regional count is 60. A fixed-count rule trusts nothing or trusts noise. |

## The trust metric used instead: regional **lift**

`lift = (regional_share) / (national_share)` — how over-represented a name is in
the region relative to the nation. A town hapax is trustworthy when it's
*regionally distinctive* (high lift, ≥2 regional mentions), not when it's merely
nationally common. Guarded by the density caveat: a single regional mention with
zero national support is treated as un-corroborated noise.

Pernegg (Steiermark) example:

```
surname     loc  reg  natl   lift  label
Lenz          1    3   437    2.4  Regionally attested (moderate)
Hermann       1    3   763    1.4  Nationally common (weak regional evidence)
Pischely      1    1     0    inf  Untrusted hapax (no corroboration)
```

## Bearing on the HTR accuracy question

This confirms, not contradicts, the earlier finding that the dict/region prior
gave ~0 OOV gain: the regional signal is **too sparse** (82% hapax, single-digit
counts) to disambiguate confident-but-wrong readings. The lift metric is a more
honest prior, but it does not manufacture evidence the corpus lacks — the
valid-but-wrong / cascade errors still need pass-2 pixels, not a richer prior.
The graph's durable value here is **aggregation + auditability**, not OCR rescue.
