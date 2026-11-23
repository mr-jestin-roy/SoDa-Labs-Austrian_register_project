"""Post-OCR lexicon correction — the classic 'Option C' baseline.

Takes an *existing* transcription (e.g. the previous single-pass Chandra
xlsx for Pernegg) and fuzzy-replaces unknown capitalised words with the
closest surname/place/given-name from the parish-scoped RTB lexicon.
No image access, no VLM — pure edit-distance repair.

This is implemented as a comparison arm, not as the primary method: it
cannot fix misreadings that landed far away in edit space ("Güldingler"
for "Auszügler") and it cannot fix valid-but-wrong words ("Pommer" read
for "Sommer" — both are real surnames, only the pixels can decide).

Usage (writes a corrected copy + replacement log, originals untouched):
    venv/bin/python postocr_correct.py \
        --xlsx ".../Seite 11-12.xlsx" --parish "Pernegg (Steiermark)" \
        --out pernegg_run/prev_postocr_seite_11-12.txt
"""
import argparse
import re
from pathlib import Path

from rapidfuzz import fuzz, process

import config
from kg_retrieval import LexiconKG, ScopedLexicon, is_noise

WORD_RE = re.compile(r"[A-ZÄÖÜ][a-zäöüß\-]{3,}")
REPLACE_CUTOFF = 86.0   # stricter than in-pipeline retrieval: no pixels to verify
MIN_MARGIN = 3.0        # best must beat runner-up by this much (ambiguity guard)

# German words that are capitalised mid-register but are not entities
SKIP = {"Jahre", "Lebensjahre", "Krankheit", "Todesart", "Religion",
        "Geschlecht", "Namen", "Haus", "Monat", "Seite", "Uhr", "Nachts",
        "Abends", "Mittags", "Vormittag", "Nachmittag"}


def build_valid_and_pool(kg: LexiconKG, scoped: ScopedLexicon,
                         aux: list[str]) -> tuple[set, list]:
    pool = list(dict.fromkeys(
        scoped.town_surnames + scoped.state_surname_pool
        + scoped.given_names + scoped.nearby_towns + aux))
    valid = set(pool) | set(kg.lex["surnames"]) | set(kg.lex["given_names"]) \
        | set(kg.lex["towns"]) | set(kg.lex["occupations"]) | set(aux)
    return valid, pool


def correct_text(text: str, kg: LexiconKG, scoped: ScopedLexicon,
                 aux: list[str]) -> tuple[str, list[dict]]:
    valid, pool = build_valid_and_pool(kg, scoped, aux)
    log = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        if word in SKIP or word in valid or is_noise(word):
            return word
        hits = process.extract(word, pool, scorer=fuzz.ratio,
                               score_cutoff=REPLACE_CUTOFF, limit=2)
        hits = [h for h in hits
                if abs(len(h[0]) - len(word)) <= max(2, 0.4 * len(word))]
        if not hits:
            return word
        if len(hits) > 1 and hits[0][1] - hits[1][1] < MIN_MARGIN:
            return word  # ambiguous — leave it
        log.append({"from": word, "to": hits[0][0],
                    "score": round(hits[0][1], 1)})
        return hits[0][0]

    return WORD_RE.sub(fix, text), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, type=Path)
    ap.add_argument("--parish", default=config.DEFAULT_PARISH)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from pernegg_compare import xlsx_to_text
    from pipeline import AUXILIARY_VOCAB
    aux = [w.strip() for w in AUXILIARY_VOCAB.split(",")]

    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon(args.parish)
    text = xlsx_to_text(args.xlsx)
    fixed, log = correct_text(text, kg, scoped, aux)

    args.out.write_text(fixed, encoding="utf-8")
    print(f"{len(log)} replacements → {args.out}")
    for entry in log:
        print(f"  {entry['from']} → {entry['to']}  ({entry['score']})")


if __name__ == "__main__":
    main()
