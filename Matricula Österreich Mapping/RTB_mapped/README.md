# RTB-1942 × Matricula Reichsgau mapping — output

Produced by `../build_rtb_reichsgau_mapping.py` from the 5 copied master
files in `../RTB_master_source/`. 1,883,448 rows total.

## Columns
`batch, cells, region, Town, ReichsGau, Diocese, gau_source, Surname, First_Name, Profession, Street_Address, Phone_Number, raw_transcribed_text`

- **Town** — `region` with all `(...)`/`[...]` qualifiers stripped.
- **ReichsGau** — the 1942 administrative Gau, when resolvable (see below).
- **gau_source** — how ReichsGau was decided: `rtb_qualifier` (from the
  entry's own bracket, e.g. "Pernegg (Steiermark)"), `matricula_match`
  (fallback via parish-name lookup), `ambiguous` (multiple candidate Gaue,
  left unresolved rather than guessed), `not_austria` (bracket explicitly
  names a German-Reich Land/Provinz, e.g. "[Bay.]" — Matricula lookup was
  skipped since it only covers Austria), `unmatched` (no usable signal).
- **Surname / First_Name / Profession / Street_Address** — heuristic split
  of `raw_transcribed_text`. First-name detection uses this corpus's own
  given-names lexicon (from `lexicon_guided_htr/artifacts/lexicon.json.gz`),
  so it already knows this corpus's abbreviation conventions ("Rich.",
  "Jos.", "Gg."). Not ground truth — expect misclassifications on unusual
  entries (titles fused to a name, place names used as street stand-ins).

## Totals across all 5 files

| gau_source | rows | share |
|---|---|---|
| rtb_qualifier | 30,149 | 1.6% |
| matricula_match | 31,878 | 1.7% |
| ambiguous | 954 | 0.1% |
| not_austria + unmatched | ~1.82M | ~96.6% |

Only ~3.3% of all RTB rows resolve to a Gau — expected, since the RTB spans
the whole German Reich and Austria/Ostmark is a small fraction of it.

## Known limitations / next steps
- `unmatched` (62% of all rows) mostly means "no bracket qualifier at all,
  or an unrecognized Kreis-level qualifier" — not necessarily "not Austria".
  A Land→Kreis lookup table would resolve more of these but wasn't built here.
- `ambiguous` rows (954) list every candidate Gau in the ReichsGau column as
  a plain comma-separated value, e.g. `Kärnten, Steiermark` — worth a manual
  pass for high-value towns.
- Surname/First_Name split is rule-based on comma position + a given-names
  lexicon; it will misparse compound business names and titles fused to a
  first name (e.g. "Dr. Josef").
