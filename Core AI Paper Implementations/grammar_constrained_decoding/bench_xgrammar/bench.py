"""bench.py -- latency + accuracy of XGrammar-constrained vs free decoding at page scale.

Inputs : the SINGLE-cell request images that step 1 built for RTB vol. 4 pages 1266-1285
         (/mnt/work/step1_loops_attention/manifest.json) and the silver reference rows for
         those pages (/mnt/data/rtb4_p1266_1285_reference.csv, from the earlier Chandra run).
Model  : Qwen3-VL-8B on the shared vLLM server (localhost:8010, served name qwen3vl).

Conditions (same prompt, same images, temperature 0, max_tokens 2048):
  G0_free        no grammar
  G1_structure   XGrammar: one or more blocks of  header-line + `ext text` entry lines
  G2_pagelex     G1 + entry text must START with a name from that page's reference
                 (closed list, like the Pernegg G2 but built per page; ~a KG town-surname list)

Runs:
  seq  : concurrency 1 over pages 1266-1270 (5 pages)  -> clean per-request latency, TTFT
  conc : concurrency 8 over pages 1266-1285 (20 pages)  -> wall-clock throughput (pages/min)
Every request gets a random cache_salt so prefix caching cannot carry image prefill between
conditions. Output: runs.jsonl (one row per request).
"""
import json, os, random, re, sys, time, base64, uuid, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd, requests

URL = "http://localhost:8010/v1/chat/completions"
MODEL = "qwen3vl"
WORK = Path("/mnt/work/bench_xgrammar"); WORK.mkdir(parents=True, exist_ok=True)
MAN = json.load(open("/mnt/work/step1_loops_attention/manifest.json"))
REF = pd.read_csv("/mnt/data/rtb4_p1266_1285_reference.csv")
PROMPT = (
    "This image is one cell of the 1942 German Reichstelefonbuch (telephone directory). "
    "Transcribe it exactly as printed. For each town/region heading in the cell, first output the "
    "heading on its own line, then one line per directory entry in the form `extension text`, "
    "where extension is the telephone number printed at the left of the entry and text is the "
    "rest of the entry. If an entry wraps over several printed lines, join it into one line. "
    "Output only the transcription, no commentary."
)
MAX_TOKENS = 2048

# Lines are separated by \n and the final \n is OPTIONAL. (v1 required a trailing \n on every line;
# the model wants EOS right after the last character, EOS was masked, and it ran on to max_tokens.)
G1 = r'''root ::= item ("\n" item)* "\n"?
item ::= header | entry
header ::= [^0-9\n] [^\n]*
entry ::= ext " " text
ext ::= [0-9]{1,5}
text ::= [^\n]+
'''


def esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def page_names(page):
    names = []
    for t in REF.loc[REF.page == page, "transcribed_text"].dropna().astype(str):
        n = t.split(",")[0].strip()[:60]
        if len(n) >= 2 and n not in names:
            names.append(n)
    return names


def g2_for(page):
    alts = "\n    | ".join(f'"{esc(n)}"' for n in page_names(page))
    return G1.replace('entry ::= ext " " text', 'entry ::= ext " " name rest') + f"name ::= {alts}\nrest ::= [^\\n]*\n"


GRAMMARS = {"G0_free": lambda p: None, "G1_structure": lambda p: G1, "G2_pagelex": g2_for}


def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()


SALT_OK = None


def call(rec, cond, run):
    global SALT_OK
    g = GRAMMARS[cond](rec["page"])
    body = {"model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64(rec["image"])}},
                {"type": "text", "text": PROMPT}]}]}
    if SALT_OK is not False:
        body["cache_salt"] = uuid.uuid4().hex
    if g:
        body["structured_outputs"] = {"grammar": g}
    t0 = time.perf_counter(); ttft = None; text = []; usage = {}; fin = None; err = None
    try:
        with requests.post(URL, json=body, stream=True, timeout=1200) as r:
            if r.status_code != 200:
                err = r.text[:300]
                if SALT_OK is None and "cache_salt" in err:
                    SALT_OK = False
                    return call(rec, cond, run)
            else:
                for line in r.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    if line == b"data: [DONE]":
                        break
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        usage = d["usage"]
                    for ch in d.get("choices", []):
                        delta = ch.get("delta", {}).get("content")
                        if delta:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            text.append(delta)
                        fin = ch.get("finish_reason") or fin
    except Exception as e:
        err = repr(e)[:300]
    if SALT_OK is None and err is None:
        SALT_OK = True
    return dict(run=run, cond=cond, page=rec["page"], req_id=rec["req_id"], latency_s=time.perf_counter() - t0,
                ttft_s=ttft, completion_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"), finish=fin, err=err,
                grammar_chars=len(g) if g else 0, text="".join(text))


def main():
    single = [r for r in MAN if r["condition"] == "SINGLE"]
    out = open(WORK / "runs.jsonl", "a")
    lock = threading.Lock()

    def save(row):
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()

    # warm-up (not recorded): compile caches, CUDA graphs
    for c in GRAMMARS:
        call(single[0], c, "warmup")

    conds = list(GRAMMARS)
    # 1) sequential, pages 1266-1275, condition order rotated per cell
    seq = [r for r in single if 1266 <= r["page"] <= 1270]
    for i, rec in enumerate(seq):
        for c in conds[i % 3:] + conds[:i % 3]:
            save(call(rec, c, "seq"))
        print("seq", i + 1, "/", len(seq), flush=True)
    # 2) concurrency 8, all 20 pages, one condition at a time (wall clock per condition)
    wall = {}
    for c in conds:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as ex:
            for row in ex.map(lambda r: call(r, c, "conc"), single):
                save(row)
        wall[c] = time.perf_counter() - t0
        print("conc", c, round(wall[c], 1), "s", flush=True)
    json.dump({"wall_conc8_s": wall, "n_cells": len(single), "n_pages": 20,
               "seq_cells": len(seq), "cache_salt": SALT_OK}, open(WORK / "wall.json", "w"), indent=1)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
