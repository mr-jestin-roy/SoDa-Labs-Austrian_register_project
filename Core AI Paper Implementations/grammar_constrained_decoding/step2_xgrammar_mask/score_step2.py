"""
score_step2.py -- score raw_outputs/*.json against the silver Pernegg rows (pernegg_rows.csv)
and, secondarily, against crop_read_reference.json (agent-typed from the crops, unverified).

Per run (grammar x input):
  cer_silver        Levenshtein(hyp, ref) / len(ref) on "\n".join(lines); ref = silver header + silver
                    entries for that region as `Phone_Number raw_transcribed_text`, in printed order
                    (order taken from the crop-read list; content untouched). n/a for the negative control.
  cer_cropread      same, against crop_read_reference.json (all printed entries incl. ones missing in CSV)
  names_recovered   silver entries whose Surname is a fuzzy prefix (rapidfuzz ratio>=90 on the first
                    len(Surname) chars, casefolded) of some output entry text (one-to-one greedy)
  ext_correct       silver entries recovered AND with the right extension, out of the silver entries
  foreign_lex       output entries starting (fuzzy prefix) with a Pernegg lexicon name that is NOT
                    printed in that crop (printed = fuzzy prefix of some crop-read entry)
                    (for the negative control: every Pernegg name it emits)
  header_ok_silver  output header == silver header (exact)
  longest_run       longest run of identical consecutive output lines
  grammar_valid     output (+EOS) accepted by the grammar via xgrammar GrammarMatcher (if available)
Normalisation: strip each line, drop empty lines and ``` fence lines. Nothing else.
"""
import csv, glob, json, os, re

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw_outputs")
GRAMMAR_FILES = {"G1_structure": "grammars/G1_structure.ebnf",
                 "G2_pernegg_lexicon": "grammars/G2_pernegg_lexicon.ebnf",
                 "G3_closed_oracle": "grammars/G3_closed_oracle.ebnf"}
CROP_REGION = {"pernegg_niederdonau_col2_sec04": "Niederdonau",
               "pernegg_steiermark_col2_sec05": "Steiermark",
               "negctrl_pernhofen_col2_sec07": None}


def lines_of(text):
    out = []
    for ln in (text or "").split("\n"):
        ln = ln.strip()
        if ln and not ln.startswith("```"):
            out.append(ln)
    return out


def parse_entry(ln):
    m = re.match(r"^(\d+)\s+(.*)$", ln)
    return (m.group(1), m.group(2)) if m else (None, ln)


def norm(s):
    return re.sub(r"\s+", " ", s).strip().casefold()


def starts_with_name(text, name, thr=90):
    """fuzzy prefix: the first len(name) chars of text match name with rapidfuzz ratio >= thr
    (so a correct 'Vorschußkasse' still counts for the silver 'Vorschubkasse')."""
    a, b = norm(text), norm(name)
    return fuzz.ratio(a[:len(b)], b) >= thr


def longest_run(ls):
    best = cur = 0
    prev = object()
    for ln in ls:
        cur = cur + 1 if ln == prev else 1
        best = max(best, cur)
        prev = ln
    return best


def cer(hyp_lines, ref_lines):
    ref = "\n".join(ref_lines)
    return Levenshtein.distance("\n".join(hyp_lines), ref) / max(1, len(ref))


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "pernegg_rows.csv"), encoding="utf-8")))
    entries = [r for r in rows if r["Surname"].strip()]
    lexicon = list(dict.fromkeys(e["Surname"] for e in entries))
    cropread = json.load(open(os.path.join(HERE, "crop_read_reference.json"), encoding="utf-8"))

    silver = {}
    for crop, reg in CROP_REGION.items():
        if reg is None:
            silver[crop] = None
            continue
        ents = [e for e in entries if reg.casefold() in e["region"].casefold()]
        header = next(e["region"] for e in rows if reg.casefold() in e["region"].casefold())
        cr = cropread[crop][1:]
        def pos(e):  # printed position = best fuzzy match in the crop-read list
            line = f'{e["Phone_Number"]} {e["raw_transcribed_text"]}'
            return max(range(len(cr)), key=lambda i: fuzz.ratio(line, cr[i]))
        ents.sort(key=pos)
        silver[crop] = {"header": header, "entries": ents,
                        "lines": [header] + [f'{e["Phone_Number"]} {e["raw_transcribed_text"]}' for e in ents]}

    # optional grammar-validity check
    matcher_for = None
    try:
        import xgrammar as xgr
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=151936)
        comp = xgr.GrammarCompiler(info)
        cgs = {k: comp.compile_grammar(open(os.path.join(HERE, p), encoding="utf-8").read())
               for k, p in GRAMMAR_FILES.items()}
        def matcher_for(g, text):
            if g not in cgs:
                return None
            m = xgr.GrammarMatcher(cgs[g])
            return bool(m.accept_string(text)) and bool(m.accept_token(info.stop_token_ids[0]))
    except Exception as ex:
        print("grammar validity check skipped:", ex)

    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "G*__*.json"))):
        r = json.load(open(f, encoding="utf-8"))
        crop, g = r["input"], r["grammar"]
        text = r.get("text") or ""
        ls = lines_of(text)
        hyp_entries = [parse_entry(x) for x in ls[1:]]
        sv = silver.get(crop)
        cr_texts = [parse_entry(x)[1] for x in cropread[crop][1:]]
        printed_names = {n for n in lexicon if any(starts_with_name(t, n) for t in cr_texts)}
        foreign = sum(1 for _, t in hyp_entries
                      if any(starts_with_name(t, n) for n in lexicon if n not in printed_names))
        rec = {"grammar": g, "input": crop, "finish_reason": r.get("finish_reason"),
               "completion_tokens": r.get("completion_tokens"), "latency_s": r.get("latency_s"),
               "n_out_lines": len(ls), "longest_run": longest_run(ls),
               "cer_cropread": round(cer(ls, cropread[crop]), 4),
               "n_cropread_entries": len(cropread[crop]) - 1,
               "foreign_lex": foreign, "error": r.get("error")}
        if sv:
            used, rec_n, ext_n = set(), 0, 0
            for e in sv["entries"]:
                for i, (ext, t) in enumerate(hyp_entries):
                    if i not in used and starts_with_name(t, e["Surname"]):
                        used.add(i)
                        rec_n += 1
                        ext_n += int(ext == e["Phone_Number"].strip())
                        break
            rec.update({"cer_silver": round(cer(ls, sv["lines"]), 4),
                        "names_recovered": f"{rec_n}/{len(sv['entries'])}",
                        "ext_correct": f"{ext_n}/{len(sv['entries'])}",
                        "header_ok_silver": bool(ls) and ls[0] == sv["header"]})
        else:
            rec.update({"cer_silver": None, "names_recovered": None, "ext_correct": None,
                        "header_ok_silver": None,
                        "header_is_pernegg": bool(ls) and ls[0].casefold().startswith("pernegg ")})
        rec["grammar_valid"] = matcher_for(g, text) if matcher_for else None
        out.append(rec)

    cols = ["grammar", "input", "cer_silver", "cer_cropread", "names_recovered", "ext_correct",
            "header_ok_silver", "header_is_pernegg", "foreign_lex", "longest_run", "n_out_lines",
            "n_cropread_entries", "finish_reason", "completion_tokens", "latency_s", "grammar_valid", "error"]
    with open(os.path.join(HERE, "results.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for rec in out:
            w.writerow({c: rec.get(c) for c in cols})
    json.dump({k: v and v["lines"] for k, v in silver.items()},
              open(os.path.join(HERE, "silver_reference_used.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    for rec in out:
        print({c: rec.get(c) for c in cols if c != "error"})


if __name__ == "__main__":
    main()
