"""Lexicon-guided two-pass HTR for Austrian parish registers.

The decoding loop is modelled on ABINet ("Read Like Humans", CVPR 2021 —
PDF in this folder): a vision reading produces a draft with explicit
uncertainty masks; an external language prior (here: the Reichstelefonbuch
1942 lexicon scoped to the parish via the town↔surname KG, cf. MegaRAG)
proposes candidates for the masked spans; the model then *re-reads the
pixels* with the candidate lexicon in context, iterating until no masks
remain.  This is pre-OCR context injection at the second pass — the lexicon
shapes decoding, it is not a post-hoc spell-checker.

Usage:
    venv/bin/python pipeline.py \
        --image "../../Althofen_Marriage Book_input_images/Althofen_TrauungsbuchTomIV_1907_1936_00015.jpg" \
        --parish "Althofen (Kärnten)" \
        --out result.json

    # Use Qwen3-VL instead of Chandra:
    VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct" venv/bin/python pipeline.py --image …

    # Inspect prompts without a VLM endpoint:
    venv/bin/python pipeline.py --image … --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

import config
import schema
from kg_retrieval import LexiconKG, ScopedLexicon

MASK_RE = re.compile(r"⟦([^⟧]*)⟧")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

PASS1_PROMPT = """\
You are an expert reader of 19th/early-20th-century Austrian parish registers \
(Trauungsbuch, Taufbuch, Sterberegister) written in German Kurrent and Latin \
cursive script.

Transcribe this register page into Markdown, preserving the table structure \
and column headers exactly as printed (e.g. date columns, Ort, names, \
Krankheit und Todesart, witnesses/Beistände).

CRITICAL: do NOT guess. Wherever a word — especially a personal name, place \
name, or occupation — is smudged, faded, or ambiguous, wrap your best partial \
reading in ⟦double brackets⟧, e.g. ⟦Hu?er⟧ or ⟦...berger⟧. Use one ⟦⟧ per \
uncertain word. Transcribe everything else verbatim, keeping original German \
spelling and abbreviations.
"""

# Static auxiliary vocabulary: Austrian calendar terms and recurring register
# formulae. These are not in the RTB lexicon (it has no month names) but fix
# a frequent error class, e.g. Kurrent "Jänner" misread as "Feinnet".
AUXILIARY_VOCAB = (
    "Jänner, Feber, Februar, März, April, Mai, Juni, Juli, August, September, "
    "Oktober, November, Dezember, Uhr, Vormittag, Nachmittag, Mittags, Abends, "
    "Nachts, Früh, gestorben, begraben, beerdigt, getauft, getraut, geboren, "
    "ehelich, unehelich, ledig, verheiratet, verwitwet, Witwe, Witwer, Sohn, "
    "Tochter, Gattin, Gatte, Pfarrer, Kaplan, Pfarre, katholisch, evangelisch, "
    "Schlagfluss, Wassersucht, Lungenentzündung, Lungentuberkulose, "
    "Tuberkulose, Abzehrung, Altersschwäche, Fraisen, Herzschlag, Gehirnschlag, "
    "Krebs, Typhus, Scharlach, Diphtherie, Schwindsucht, Brand, Entzündung"
)

PASS2_PROMPT_TEMPLATE = """\
You are an expert reader of 19th/early-20th-century Austrian parish registers \
from the parish of {parish}. You previously produced the DRAFT \
transcription below; uncertain words are wrapped in ⟦double brackets⟧.

Re-examine the page image and resolve every ⟦masked⟧ word by matching the \
visible letter shapes against this verified vocabulary, extracted from the \
1942 Reichstelefonbuch (a ground-truth directory of real names in this \
region):

SURNAMES ATTESTED IN {town} ITSELF (most likely):
{town_surnames}

SURNAMES FROM THE SURROUNDING STATE ({state}):
{state_surnames}

COMMON GIVEN NAMES:
{given_names}

PLACE NAMES IN {state}:
{nearby_towns}

COMMON OCCUPATIONS:
{occupations}

CALENDAR TERMS AND REGISTER FORMULAE (Austrian German):
{auxiliary}
{per_token_block}
Rules:
1. A masked word may only be replaced by a vocabulary entry if the entry is \
visually consistent with the strokes on the page (same first letter unless \
that letter itself is ambiguous in Kurrent, compatible length and ascenders/\
descenders). The vocabulary guides reading — it must not override the pixels.
2. If no vocabulary entry is visually plausible, transcribe what you see and \
keep the ⟦brackets⟧.
3. Words that were already certain in the draft must be copied unchanged.
4. {output_rule}

DRAFT:
{draft}
"""

MARKDOWN_OUTPUT_RULE = ("Output ONLY the corrected full transcription in the "
                        "same Markdown layout.")
# Structured mode keeps the pass-2 loop identical; only the carrier format
# of draft and output changes (frozen contract in schema.py — see C3 note).
STRUCTURED_OUTPUT_RULE = ("Output ONLY the corrected JSON object with exactly "
                          "the same structure and fields as the DRAFT — no "
                          "Markdown fences, no commentary.")

PER_TOKEN_HEADER = "\nBEST CANDIDATES PER MASKED WORD (fuzzy-ranked):\n"


def parse_structured(text: str) -> tuple[dict | None, list[str]]:
    """Parse a structured-mode model reply into a PAGE_SCHEMA dict.

    Returns (parsed, problems); parsed is None when the reply is not valid
    JSON at all. Fences are tolerated even though the prompt forbids them.
    """
    cleaned = FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e}"]
    if not isinstance(obj, dict):
        return None, ["top-level JSON is not an object"]
    return obj, schema.validate_page(obj)


def build_pass2_prompt(draft: str, scoped: ScopedLexicon,
                       kg: LexiconKG, structured: bool = False) -> str:
    masked = MASK_RE.findall(draft)
    per_token = ""
    if masked:
        lines = []
        aux_pool = [w.strip() for w in AUXILIARY_VOCAB.split(",")]
        for tok in dict.fromkeys(masked):  # unique, keep order
            cands = kg.candidates_for(tok.replace("?", "").replace("...", ""),
                                      scoped, extra_pool=aux_pool)
            if cands:
                lines.append(f"  ⟦{tok}⟧ → " + ", ".join(c for c, _ in cands))
        if lines:
            per_token = PER_TOKEN_HEADER + "\n".join(lines) + "\n"
    return PASS2_PROMPT_TEMPLATE.format(
        parish=scoped.parish,
        town=scoped.matched_town or scoped.parish,
        state=scoped.state or "the region",
        town_surnames=", ".join(scoped.town_surnames) or "(none)",
        state_surnames=", ".join(scoped.state_surnames) or "(none)",
        given_names=", ".join(scoped.given_names),
        nearby_towns=", ".join(scoped.nearby_towns) or "(none)",
        occupations=", ".join(scoped.occupations),
        auxiliary=AUXILIARY_VOCAB,
        per_token_block=per_token,
        output_rule=STRUCTURED_OUTPUT_RULE if structured
        else MARKDOWN_OUTPUT_RULE,
        draft=draft,
    )


def run(image: Path, parish: str, iterations: int, model: str | None,
        dry_run: bool, structured: bool = False) -> dict:
    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon(parish)
    print(f"Parish {parish!r} resolved to town={scoped.matched_town!r}, "
          f"state={scoped.state!r} | 1-hop surnames={len(scoped.town_surnames)} "
          f"2-hop={len(scoped.state_surnames)}", file=sys.stderr)

    pass1_prompt = schema.STRUCTURED_PROMPT if structured else PASS1_PROMPT
    result = {"image": str(image), "parish": parish,
              "matched_town": scoped.matched_town, "state": scoped.state,
              "model": model or config.VLM_MODEL,
              "mode": "structured" if structured else "markdown",
              "schema_version": schema.SCHEMA_VERSION if structured else None,
              "passes": []}

    if dry_run:
        fake_draft = ('{"register_type": "Trauungsbuch", "entries": [{"date": '
                      '"⟦17.⟧ Jänner 1909", "persons": [{"role": "Bräutigam", '
                      '"given_name": "Johann", "surname": "⟦Hu6er⟧", '
                      '"profession": "⟦Bauerssohn⟧"}]}]}'
                      if structured else
                      "| 1909 | ⟦Hu6er⟧ Johann, ⟦Bauerssohn⟧ | ⟦Wieser⟧ Maria |")
        result["pass1_prompt"] = pass1_prompt
        result["pass2_prompt_example"] = build_pass2_prompt(
            fake_draft, scoped, kg, structured=structured)
        return result

    from vlm_client import VLMClient
    client = VLMClient(model=model)

    def record(pass_no: int, kind: str, text: str) -> dict:
        entry = {"pass": pass_no, "kind": kind,
                 "masks": len(MASK_RE.findall(text)), "text": text}
        if structured:
            parsed, problems = parse_structured(text)
            entry["parsed"] = parsed
            entry["schema_problems"] = problems
            if problems:
                print(f"Pass {pass_no}: schema problems: {problems}",
                      file=sys.stderr)
        result["passes"].append(entry)
        print(f"Pass {pass_no}: {entry['masks']} uncertain spans",
              file=sys.stderr)
        return entry

    current = client.transcribe(image, pass1_prompt)
    record(1, "vision-draft", current)

    for i in range(iterations):
        if not MASK_RE.search(current) and i > 0:
            break
        prompt = build_pass2_prompt(current, scoped, kg, structured=structured)
        current = client.transcribe(image, prompt)
        entry = record(i + 2, "lexicon-guided", current)
        if entry["masks"] == 0:
            break

    result["final"] = current
    if structured:
        result["final_parsed"] = result["passes"][-1]["parsed"]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--parish", default=config.DEFAULT_PARISH)
    ap.add_argument("--iterations", type=int, default=2,
                    help="max lexicon-guided refinement passes (ABINet-style)")
    ap.add_argument("--model", default=None,
                    help="override VLM_MODEL, e.g. Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--structured", action="store_true",
                    help="emit schema.py JSON (frozen C3 contract) instead of "
                         "Markdown; enables field-level constrained decoding "
                         "and entity-level evaluation downstream")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print prompts without calling the VLM")
    args = ap.parse_args()

    if not args.image.exists() and not args.dry_run:
        sys.exit(f"Image not found: {args.image}")

    result = run(args.image, args.parish, args.iterations, args.model,
                 args.dry_run, structured=args.structured)

    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(out, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(out)


if __name__ == "__main__":
    main()
