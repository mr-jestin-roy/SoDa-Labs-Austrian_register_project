"""
make_grammars.py -- STEP 2: build the XGrammar EBNF masks from the Pernegg silver rows,
then compile each one on CPU with xgrammar against the Qwen3-VL-8B tokenizer and time it.

  G0  none (free decoding)                      -> no file
  G1  structure only                            -> grammars/G1_structure.ebnf
  G2  Pernegg lexicon (header + entry prefix)   -> grammars/G2_pernegg_lexicon.ebnf
  G3  closed oracle (exactly the ref entries)   -> grammars/G3_closed_oracle.ebnf   (LEAKS ANSWER)

Usage:  python make_grammars.py [--csv pernegg_rows.csv] [--out grammars] [--no-compile]
Writes grammars/compile_report.json.
"""
import argparse, csv, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_ENTRIES_OPEN = 60     # G1/G2 cap (a cell never has more than ~60 entries)
MAX_ENTRIES_ORACLE = 20   # G3 cap


def q(s: str) -> str:
    """EBNF double-quoted string literal (xgrammar dialect)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    regions = []
    for r in rows:
        if r["region"] not in regions:
            regions.append(r["region"])
    entries = [r for r in rows if r["Surname"].strip()]
    return rows, regions, entries


def alt(items):
    return "\n    | ".join(q(x) for x in items)


def build(regions, entries, rep_ok=True):
    def rep(sym, n):
        if rep_ok:
            return f"{sym}{{1,{n}}}"
        # unrolled fallback: sym (sym (sym ...)?)?
        out = sym
        for _ in range(n - 1):
            out = f"{sym} ({out})?"
        return out

    g1 = f"""# G1 structure-only: header line, then `extension SPACE text NEWLINE`, extension digits only.
root ::= header {rep('entry', MAX_ENTRIES_OPEN)}
header ::= [^\\n]+ "\\n"
entry ::= ext " " text "\\n"
ext ::= [0-9]{{1,5}}
text ::= [^\\n]+
"""
    names = []
    for e in entries:
        if e["Surname"] not in names:
            names.append(e["Surname"])
    g2 = f"""# G2 Pernegg lexicon: header must be one of the Pernegg region strings (both regions, incl.
# the lowercase variant as it occurs in the silver CSV); each entry text must START with one of
# the Pernegg Surname/organisation strings from pernegg_rows.csv, then free text to end of line.
root ::= header {rep('entry', MAX_ENTRIES_OPEN)}
header ::= region "\\n"
region ::= {alt(regions)}
entry ::= ext " " name rest "\\n"
ext ::= [0-9]{{1,5}}
name ::= {alt(names)}
rest ::= [^\\n]*
"""
    lines = []
    for e in entries:
        ln = f'{e["Phone_Number"].strip()} {e["raw_transcribed_text"].strip()}'
        if ln not in lines:
            lines.append(ln)
    g3 = f"""# G3 CLOSED ORACLE (leaks the answer; upper bound / proof the mask is applied):
# header = a Pernegg region string; every entry line must be exactly one of the reference entries
# `Phone_Number raw_transcribed_text` from pernegg_rows.csv.
root ::= header {rep('entry', MAX_ENTRIES_ORACLE)}
header ::= region "\\n"
region ::= {alt(regions)}
entry ::= ref "\\n"
ref ::= {alt(lines)}
"""
    return {"G1_structure": g1, "G2_pernegg_lexicon": g2, "G3_closed_oracle": g3}, names, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(HERE, "pernegg_rows.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "grammars"))
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, regions, entries = load_rows(a.csv)
    report = {"n_rows": len(rows), "n_entries": len(entries), "regions": regions}

    rep_ok = True
    if not a.no_compile:
        import xgrammar as xgr
        from importlib.metadata import version
        report["xgrammar_version"] = version("xgrammar")
        try:
            xgr.Grammar.from_ebnf('root ::= [0-9]{1,3} "x"')
            report["repetition_range_supported"] = True
        except Exception as ex:
            rep_ok = False
            report["repetition_range_supported"] = False
            report["repetition_range_error"] = str(ex)[:300]

    grammars, names, lines = build(regions, entries, rep_ok)
    report["n_lexicon_names"] = len(names)
    report["n_oracle_lines"] = len(lines)
    for k, g in grammars.items():
        with open(os.path.join(a.out, k + ".ebnf"), "w", encoding="utf-8") as f:
            f.write(g)

    if not a.no_compile:
        from transformers import AutoTokenizer, AutoConfig
        tok = AutoTokenizer.from_pretrained(MODEL)
        cfg = AutoConfig.from_pretrained(MODEL)
        vocab = getattr(getattr(cfg, "text_config", cfg), "vocab_size", None) or len(tok)
        t0 = time.perf_counter()
        info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab)
        compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=False)
        report["tokenizer_info_s"] = round(time.perf_counter() - t0, 3)
        report["vocab_size"] = vocab
        report["compile"] = {}
        for k, g in grammars.items():
            t0 = time.perf_counter()
            cg = compiler.compile_grammar(g)
            dt = time.perf_counter() - t0
            report["compile"][k] = {"ok": True, "compile_s": round(dt, 4),
                                    "ebnf_chars": len(g), "memory_bytes": getattr(cg, "memory_size_bytes", None)}
        # sanity: the full silver reference must be accepted by G2 and G3; a junk name must be rejected by G2
        def accepts(g, text):
            m = xgr.GrammarMatcher(compiler.compile_grammar(g))
            if not m.accept_string(text):
                return False
            return bool(m.accept_token(info.stop_token_ids[0]))  # grammar must allow EOS here
        st = [e for e in entries if "Steiermark" in e["region"] or "steiermark" in e["region"]]
        ref_st = "Pernegg (Steiermark)\n" + "".join(
            f'{e["Phone_Number"]} {e["raw_transcribed_text"]}\n' for e in st)
        bad = "Pernegg (Steiermark)\n6 Bahnhof\n"
        report["sanity"] = {
            "G1_accepts_ref": accepts(grammars["G1_structure"], ref_st),
            "G2_accepts_ref": accepts(grammars["G2_pernegg_lexicon"], ref_st),
            "G3_accepts_ref": accepts(grammars["G3_closed_oracle"], ref_st),
            "G2_rejects_'6 Bahnhof'": not accepts(grammars["G2_pernegg_lexicon"], bad),
            "G3_rejects_'6 Bahnhof'": not accepts(grammars["G3_closed_oracle"], bad),
        }
    with open(os.path.join(a.out, "compile_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
