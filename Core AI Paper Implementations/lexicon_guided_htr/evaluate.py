"""Compare the vision-only draft (pass 1) against the lexicon-guided final
transcription on CER/WER and lexicon entity hit-rate; for structured results
(pipeline.py --structured) additionally compute entity-level metrics against
a gold page in PAGE_SCHEMA shape.

Usage:
    # legacy free-text result against a ground-truth text file
    venv/bin/python evaluate.py --result result.json --truth page15_gt.txt

    # no ground truth: report mask resolution + entity-grounding stats only
    venv/bin/python evaluate.py --result result.json

    # structured result against a gold JSON page (PAGE_SCHEMA shape)
    venv/bin/python evaluate.py --result result.json --truth-json page15_gt.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import editdistance

from kg_retrieval import LexiconKG
from schema import FIELD_LEXICON_MAP, validate_page

MASK_RE = re.compile(r"⟦([^⟧]*)⟧")
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]+")

# Fields scored at entity level: the lexicon-constrained ones plus the
# unconstrained role/date so the whole record is covered.
PERSON_FIELDS = ("role", "given_name", "surname", "profession", "place")
STRUCT_FIELDS = tuple(FIELD_LEXICON_MAP) + ("role", "date")


def cer(hyp: str, ref: str) -> float:
    return editdistance.eval(hyp, ref) / max(1, len(ref))


def wer(hyp: str, ref: str) -> float:
    h, r = hyp.split(), ref.split()
    return editdistance.eval(h, r) / max(1, len(r))


def entity_hit_rate(text: str, lexicon_words: set[str]) -> tuple[float, int]:
    """Share of capitalised words in the transcription that are attested in
    the RTB lexicon — a grounding proxy when no ground truth exists."""
    caps = [w for w in WORD_RE.findall(text) if w[0].isupper() and len(w) > 2]
    if not caps:
        return 0.0, 0
    hits = sum(1 for w in caps if w in lexicon_words)
    return hits / len(caps), len(caps)


# ── structured-mode helpers ──────────────────────────────────────────────────

def strip_field(value: str) -> str:
    """Drop ⟦⟧ uncertainty markers (keeping the reading) and outer space."""
    return MASK_RE.sub(lambda m: m.group(1), value).strip()


def field_bags(page: dict) -> dict[str, Counter]:
    """Bag (multiset) of stripped values per STRUCT_FIELD over the page."""
    bags = {f: Counter() for f in STRUCT_FIELDS}
    for entry in page.get("entries") or []:
        date = entry.get("date")
        if isinstance(date, str) and strip_field(date):
            bags["date"][strip_field(date)] += 1
        for person in entry.get("persons") or []:
            for f in PERSON_FIELDS:
                v = person.get(f)
                if f in bags and isinstance(v, str) and strip_field(v):
                    bags[f][strip_field(v)] += 1
    return bags


def multiset_prf(pred: Counter, gold: Counter) -> tuple[float, float, float]:
    """Order-independent precision/recall/F1 over value multisets."""
    n_pred, n_gold = sum(pred.values()), sum(gold.values())
    if not n_pred and not n_gold:
        return 1.0, 1.0, 1.0
    overlap = sum((pred & gold).values())
    p = overlap / n_pred if n_pred else 0.0
    r = overlap / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def align_persons(pred_persons: list, gold_persons: list) -> list[tuple]:
    """Greedy 1:1 matching of predicted to gold persons, best (role, surname)
    agreement first; surname edit distance breaks ties."""
    pairs = []
    for gi, g in enumerate(gold_persons):
        g_role = strip_field(str(g.get("role", "")))
        g_sur = strip_field(str(g.get("surname", "")))
        for pi, p in enumerate(pred_persons):
            p_role = strip_field(str(p.get("role", "")))
            p_sur = strip_field(str(p.get("surname", "")))
            score = (2 * (p_sur == g_sur and bool(g_sur))
                     + (p_role == g_role and bool(g_role))
                     - 0.01 * editdistance.eval(p_sur, g_sur))
            pairs.append((score, gi, pi))
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_g, used_p, matched = set(), set(), []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched.append((pred_persons[pi], gold_persons[gi]))
    return matched


def entry_aligned_exact(pred: dict, gold: dict) -> dict[str, list[int]]:
    """Per-field [hits, gold-present total] with entries aligned by index and
    persons by best (role, surname) match. Caller checks len(entries) agree."""
    counts = {f: [0, 0] for f in STRUCT_FIELDS}
    for pe, ge in zip(pred.get("entries") or [], gold.get("entries") or []):
        g_date = ge.get("date")
        if isinstance(g_date, str) and strip_field(g_date):
            counts["date"][1] += 1
            counts["date"][0] += (strip_field(str(pe.get("date", "")))
                                  == strip_field(g_date))
        gold_persons = ge.get("persons") or []
        for g in gold_persons:  # unmatched gold persons still count as misses
            for f in PERSON_FIELDS:
                gv = g.get(f)
                if f in counts and isinstance(gv, str) and strip_field(gv):
                    counts[f][1] += 1
        for p, g in align_persons(pe.get("persons") or [], gold_persons):
            for f in PERSON_FIELDS:
                gv = g.get(f)
                if f in counts and isinstance(gv, str) and strip_field(gv):
                    counts[f][0] += (strip_field(str(p.get(f, "")))
                                     == strip_field(gv))
    return counts


def count_persons(page: dict) -> int:
    return sum(len(e.get("persons") or []) for e in page.get("entries") or [])


def count_unresolved(page: dict) -> int:
    """Number of field values on the page still containing a ⟦ marker."""
    n = 0
    for entry in page.get("entries") or []:
        for v in (entry.get("date"), entry.get("notes")):
            n += isinstance(v, str) and "⟦" in v
        for person in entry.get("persons") or []:
            n += sum(isinstance(v, str) and "⟦" in v
                     for v in person.values())
    return n


def canonical_text(page: dict) -> str:
    """Flatten a page to plain text: all values in document order,
    space-separated, ⟦⟧ markers stripped — the CER/WER fallback target."""
    parts = []
    if isinstance(page.get("register_type"), str):
        parts.append(page["register_type"])
    for entry in page.get("entries") or []:
        if isinstance(entry.get("date"), str):
            parts.append(entry["date"])
        for person in entry.get("persons") or []:
            parts.extend(person[f] for f in PERSON_FIELDS
                         if isinstance(person.get(f), str))
        if isinstance(entry.get("notes"), str):
            parts.append(entry["notes"])
    return " ".join(strip_field(p) for p in parts if strip_field(p))


def report_page(label: str, page: dict, gold: dict | None):
    problems = validate_page(page)
    schema_str = "valid" if not problems else f"{len(problems)} problems"
    counts = f"entries={len(page.get('entries') or [])}"
    persons = f"persons={count_persons(page)}"
    if gold is not None:
        counts += f" (gold {len(gold.get('entries') or [])})"
        persons += f" (gold {count_persons(gold)})"
    print(f"{label}: {counts}  {persons}  "
          f"unresolved-⟦⟧-fields={count_unresolved(page)}  schema={schema_str}")
    for prob in problems:
        print(f"  schema problem: {prob}")
    if gold is None:
        return
    pred_bags, gold_bags = field_bags(page), field_bags(gold)
    print(f"{label}: bag-of-values P/R/F1")
    for f in STRUCT_FIELDS:
        p, r, f1 = multiset_prf(pred_bags[f], gold_bags[f])
        print(f"  {f:<11} P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    n_pred = len(page.get("entries") or [])
    n_gold = len(gold.get("entries") or [])
    if n_pred == n_gold:
        print(f"{label}: entry-aligned exact match")
        for f, (hits, total) in entry_aligned_exact(page, gold).items():
            if total:
                print(f"  {f:<11} {hits}/{total} ({hits / total:.1%})")
    else:
        print(f"{label}: entry count mismatch "
              f"({n_pred} vs gold {n_gold}) — skipping entry-aligned match")


def evaluate_structured(result: dict, gold: dict | None):
    passes = result.get("passes", [])
    final_parsed = result.get("final_parsed")
    if final_parsed is None:
        parsed_passes = [p.get("parsed") for p in passes
                         if p.get("parsed") is not None]
        final_parsed = parsed_passes[-1] if parsed_passes else None

    if final_parsed is None:
        print("final: unparseable JSON — no structured metrics")
        if gold is not None:
            ref = canonical_text(gold)
            final = result.get("final", passes[-1]["text"] if passes else "")
            hyp = strip_field(final)
            print("final vs gold-as-text: "
                  f"CER={cer(hyp, ref):.4f}  WER={wer(hyp, ref):.4f}")
        return

    # Draft first, so the lexicon-guided improvement is attributable
    # (mirrors the draft/final CER/WER pattern in the free-text path).
    draft_parsed = passes[0].get("parsed") if passes else None
    if draft_parsed is not None and draft_parsed is not final_parsed:
        report_page("draft", draft_parsed, gold)
    report_page("final", final_parsed, gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, type=Path,
                    help="JSON produced by pipeline.py")
    ap.add_argument("--truth", type=Path, default=None,
                    help="plain-text ground-truth transcription")
    ap.add_argument("--truth-json", type=Path, default=None,
                    help="gold page in PAGE_SCHEMA shape (structured mode)")
    args = ap.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    print(f"image: {result.get('image')}  model: {result.get('model')}")

    structured = result.get("mode") == "structured" or args.truth_json
    if structured:
        gold = (json.loads(args.truth_json.read_text(encoding="utf-8"))
                if args.truth_json else None)
        evaluate_structured(result, gold)
        return

    passes = result.get("passes", [])
    if not passes:
        raise SystemExit("Result file contains no passes (dry-run output?)")
    draft = passes[0]["text"]
    final = result.get("final", passes[-1]["text"])
    strip = lambda t: MASK_RE.sub(lambda m: m.group(1), t)

    kg = LexiconKG.load()
    lex_words = (set(kg.lex["surnames"]) | set(kg.lex["given_names"])
                 | set(kg.lex["towns"]))

    print(f"masks: pass1={len(MASK_RE.findall(draft))} "
          f"final={len(MASK_RE.findall(final))}")
    for label, text in (("draft", draft), ("final", final)):
        rate, n = entity_hit_rate(strip(text), lex_words)
        print(f"{label}: RTB entity hit-rate {rate:.1%} over {n} cap-words")

    if args.truth:
        ref = args.truth.read_text(encoding="utf-8").strip()
        for label, text in (("draft", draft), ("final", final)):
            t = strip(text).strip()
            print(f"{label}: CER={cer(t, ref):.4f}  WER={wer(t, ref):.4f}")


if __name__ == "__main__":
    main()
