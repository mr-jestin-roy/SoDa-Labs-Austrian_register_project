# Part A — Dictionary-Guided Re-ranking: OOV evaluation on Pernegg

Test page: **8758 Sterberegister, Seite 11-12 (1895)**, 14 name fields, manual GT.
Dictionary: Reichstelefonbuch 1942 scoped to **Pernegg (Steiermark)** — 1-hop town
surnames = 7, 2-hop region (Steiermark) = 1,980 surnames, national = 82,991.
Re-ranker: confusion-weighted edit distance + region-frequency prior
(`W_RTB=6, W_FREQ=4`), with a visual-similarity floor τ and OOV fallback.

## Surname buckets (by RTB attestation)

| bucket | n | surnames |
|---|---|---|
| region-attested | 4 | Sommer, Posch, Reiter, Wallner |
| national-only | 6 | Steinbauer, Gregor, Schötterl, Niederdorfer, Ranner, Feldhofer |
| OOV (not in RTB at all) | 4 | Grabmayer, Grassnegger, Grossegger, Strassegger |

## Result — CER by bucket, baseline vs τ-sweep

| τ (SIM_FLOOR) | repl | all | region | national | OOV |
|---|---|---|---|---|---|
| **baseline** | 0 | **0.171** | **0.337** | **0.101** | **0.043** |
| 66 | 21 | 0.186 | 0.326 | 0.111 | 0.101 |
| 72 | 29 | 0.190 | 0.347 | 0.111 | 0.087 |
| 78 | 23 | 0.179 | 0.347 | 0.111 | 0.043 |
| 84 | 4 | 0.175 | 0.337 | 0.111 | 0.043 |
| 88 | 2 | 0.179 | 0.337 | 0.111 | 0.058 |

**Net OOV-accuracy improvement after this post-inference treatment: ≈ 0 (and
negative at aggressive τ).** No τ improves the aggregate, and the only τ that
nudges `region` (0.337→0.326 at τ=66, ~1 character over 4 names) simultaneously
*doubles* OOV error (0.043→0.101) by over-correcting names the baseline read
correctly.

## Why — two structural walls (both reproduce the report's thesis)

1. **Valid-but-wrong is unreachable.** The baseline misread *Sommer* (region,
   reg=4 / nat=801) as *Pommer*. But `Pommer` is itself a valid national RTB
   surname, so the guard that prevents over-correction skips it entirely.
   Dropping the guard does **not** help: `Pommer` is a perfect visual
   self-match (sim=100), and the region prior cannot overcome the 6-point
   visual gap to `Sommer` (sim=94). Worse, the region-attested-but-wrong
   `Pammer` (hop-bonus=2) outranks `Sommer` — so an unguarded re-ranker would
   flip `Pommer → Pammer`, still wrong. Only re-reading the pixels can decide.

   ```
   token "Pommer"  (after dropping the valid-word guard)
   cand      sim  hop  freq   TOTAL
   Pommer    100   0    46    115.4   <- wins: perfect self-match
   Pammer     94   2     1    108.9   <- region prior pulls a WRONG name up
   Sommer     94   1     4    106.6   <- the truth, ranked 3rd
   ```

2. **Region-common names are the most misread, in the least fixable way.**
   Baseline CER is *highest* on exactly the region-attested bucket (0.337) and
   *lowest* on OOV (0.043). The names RTB could help with are the ones the VLM
   got wrong — but it got them wrong as valid-but-wrong substitutions or
   far-edit-distance cascade failures, neither reachable from text alone.

## Conclusion

Region-frequency-weighted dictionary guidance is sound in principle, but as a
**post-inference string re-ranker** it yields no OOV gain on this corpus — the
same negative result the post-OCR arm reported, now quantified at bucket
granularity. The improvement requires injecting the region lexicon **during
decoding / pixel re-reading** (pipeline.py pass 2), where `Pommer` vs `Sommer`
vs `Pammer` is resolved against the strokes, not the string. The re-ranker's
value is therefore as the *candidate generator* feeding pass 2, not as a
standalone corrector.

Artifacts in this folder:
- `oov_eval.json` — full sweep, per-bucket CER/WER, surname→bucket map.
- `Seite 11-12__dictguided.xlsx` — corrected copy (source untouched).
- `Seite 11-12__replacements.json` — replacement log.
