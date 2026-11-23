"""score.py -- latency + entry-level accuracy from runs.jsonl -> summary.csv, per_page.csv, summary.json

Accuracy is page-level against the SILVER reference (earlier Chandra transcription, not
human-checked): reference entries = (phone_extension, transcribed_text) rows of that page;
predicted entries = output lines `ext text`. Greedy one-to-one matching per page by
rapidfuzz ratio on the text (threshold 80).
  recall    = matched reference entries / reference entries
  precision = matched predicted entries / predicted entries
  char_acc  = mean (1 - normalised Levenshtein) over matched pairs
  exact     = matched pairs whose text is identical after whitespace/dash normalisation
  ext_acc   = matched pairs with the same extension
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/work/bench_xgrammar")
REF_CSV = sys.argv[2] if len(sys.argv) > 2 else "/mnt/data/rtb4_p1266_1285_reference.csv"
ENTRY = re.compile(r"^\s*(\d{1,5})\s+(.+?)\s*$")


def norm(s):
    s = str(s).replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip()


def longest_run(lines):
    best = cur = 0; prev = None
    for l in lines:
        k = ENTRY.sub(r"\2", l).strip().lower()
        cur = cur + 1 if k == prev and k else 1
        prev = k; best = max(best, cur)
    return best


rows = [json.loads(l) for l in open(WORK / "runs.jsonl") if l.strip()]
rows = [r for r in rows if r["run"] != "warmup"]
ref = pd.read_csv(REF_CSV)
ref = ref[ref.phone_extension.notna() & ref.transcribed_text.notna()]
ref_by_page = {int(p): [(str(int(float(e))) if re.fullmatch(r"[\d.]+", str(e)) else str(e), norm(t))
                        for e, t in zip(g.phone_extension, g.transcribed_text)]
               for p, g in ref.groupby("page")}

# ---------- latency ----------
lat = []
for (run, cond), g in pd.DataFrame(rows).groupby(["run", "cond"]):
    ok = g[g.err.isna()]
    tok = ok.completion_tokens.fillna(0)
    dec = (ok.latency_s - ok.ttft_s.fillna(0)).clip(lower=1e-6)
    lat.append(dict(run=run, cond=cond, n=len(g), errors=int(g.err.notna().sum()),
                    hit_max_tokens=int((ok.finish == "length").sum()),
                    lat_mean_s=ok.latency_s.mean(), lat_p50_s=ok.latency_s.median(),
                    lat_p95_s=ok.latency_s.quantile(.95), ttft_mean_s=ok.ttft_s.mean(),
                    tok_mean=tok.mean(), tok_total=int(tok.sum()),
                    decode_tok_per_s=(tok / dec).replace([np.inf], np.nan).median(),
                    longest_run_max=max(longest_run(t.splitlines()) for t in ok.text)))
lat = pd.DataFrame(lat)

# ---------- accuracy (page level) ----------
per_page = []
for (run, cond, page), g in pd.DataFrame(rows).groupby(["run", "cond", "page"]):
    if int(page) not in ref_by_page:
        continue
    preds = []
    for t in g.text.fillna(""):
        for l in t.splitlines():
            m = ENTRY.match(l)
            if m:
                preds.append((m.group(1), norm(m.group(2))))
    refs = ref_by_page[int(page)]
    free = {i: p[1] for i, p in enumerate(preds)}
    match = []
    for re_ext, re_txt in refs:
        if not free:
            break
        hit = process.extractOne(re_txt, free, scorer=fuzz.ratio, score_cutoff=80)
        if hit:
            _, score, k = hit
            match.append((re_ext, re_txt, preds[k][0], preds[k][1])); del free[k]
    n_ref, n_pred, n_m = len(refs), len(preds), len(match)
    per_page.append(dict(run=run, cond=cond, page=int(page), n_ref=n_ref, n_pred=n_pred, matched=n_m,
                         recall=n_m / n_ref if n_ref else np.nan,
                         precision=n_m / n_pred if n_pred else np.nan,
                         char_acc=np.mean([1 - Levenshtein.normalized_distance(a, b) for _, a, _, b in match]) if match else np.nan,
                         exact=sum(a == b for _, a, _, b in match),
                         ext_ok=sum(x == y for x, _, y, _ in match)))
pp = pd.DataFrame(per_page)
acc = pp.groupby(["run", "cond"]).agg(pages=("page", "nunique"), n_ref=("n_ref", "sum"), n_pred=("n_pred", "sum"),
                                       matched=("matched", "sum"), exact=("exact", "sum"), ext_ok=("ext_ok", "sum"),
                                       char_acc=("char_acc", "mean")).reset_index()
acc["recall"] = acc.matched / acc.n_ref
acc["precision"] = acc.matched / acc.n_pred
acc["exact_rate"] = acc.exact / acc.n_ref
acc["ext_acc"] = acc.ext_ok / acc.matched

summary = lat.merge(acc, on=["run", "cond"], how="left")
wall = json.load(open(WORK / "wall.json")) if (WORK / "wall.json").exists() else {}
for c, s in wall.get("wall_conc8_s", {}).items():
    m = (summary.run == "conc") & (summary.cond == c)
    summary.loc[m, "wall_20pages_s"] = s
    summary.loc[m, "s_per_page"] = s / wall["n_pages"]
    summary.loc[m, "est_100pages_min"] = s / wall["n_pages"] * 100 / 60
summary.to_csv(WORK / "summary.csv", index=False)
pp.to_csv(WORK / "per_page.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(summary.round(3).to_string(index=False))
