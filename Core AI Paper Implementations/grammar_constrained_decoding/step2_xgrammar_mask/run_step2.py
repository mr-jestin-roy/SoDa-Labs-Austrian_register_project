"""
run_step2.py -- STEP 2 client: every grammar (G0..G3) x every input crop through the SHARED
vLLM server (Qwen3-VL-8B, served name `qwen3vl`, localhost:8010). Temperature 0, max_tokens 2048.

1. Probes which structured-output request parameter this vLLM accepts, with a text-only request
   whose grammar only allows the string PROBE_OK:
     a) extra_body={"structured_outputs": {"grammar": g}}     (newer vLLM)
     b) extra_body={"guided_grammar": g}                      (older vLLM)
   A style "works" only if the output is exactly PROBE_OK (an ignored parameter would give free text).
2. Runs the 4 x N matrix sequentially (one request at a time; the server is shared).
Raw responses -> raw_outputs/<grammar>__<input>.json ; summary -> raw_outputs/run_summary.json
"""
import base64, glob, json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.getenv("VLLM_URL", "http://localhost:8010")
MODEL = os.getenv("VLLM_MODEL", "qwen3vl")
OUT = os.path.join(HERE, "raw_outputs")
CROPS = os.path.join(HERE, "crops")
GRAMMARS = {
    "G0_none": None,
    "G1_structure": "grammars/G1_structure.ebnf",
    "G2_pernegg_lexicon": "grammars/G2_pernegg_lexicon.ebnf",
    "G3_closed_oracle": "grammars/G3_closed_oracle.ebnf",
}
PROMPT = (
    "This image is one cell of the 1942 German Reichstelefonbuch (telephone directory). "
    "Transcribe it exactly as printed. First output the town/region header line. "
    "Then output one line per directory entry in the form `extension text`, where extension is "
    "the telephone number printed at the left of the entry and text is the rest of the entry. "
    "If an entry wraps over several printed lines, join it into one line. "
    "Output only the transcription, no commentary."
)


def post(path, payload, timeout=900):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path):
    with urllib.request.urlopen(URL + path, timeout=30) as r:
        return r.read().decode()


def grammar_fields(style, g):
    if g is None:
        return {}
    if style == "structured_outputs":
        return {"structured_outputs": {"grammar": g}}
    if style == "guided_grammar":
        return {"guided_grammar": g}
    raise ValueError(style)


def probe():
    g = 'root ::= "PROBE_OK"'
    res = {}
    for style in ("structured_outputs", "guided_grammar"):
        body = {"model": MODEL, "temperature": 0, "max_tokens": 16,
                "messages": [{"role": "user", "content": "Say hello."}], **grammar_fields(style, g)}
        try:
            r = post("/v1/chat/completions", body, timeout=120)
            txt = r["choices"][0]["message"]["content"]
            res[style] = {"output": txt, "works": txt.strip() == "PROBE_OK"}
        except Exception as ex:
            res[style] = {"error": str(ex)[:500], "works": False}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {"url": URL, "model": MODEL}
    try:
        meta["vllm_version"] = get("/version")
    except Exception as ex:
        meta["vllm_version"] = f"unavailable: {ex}"
    meta["probe"] = probe()
    style = next((s for s in ("structured_outputs", "guided_grammar") if meta["probe"][s]["works"]), None)
    meta["style_used"] = style
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if style is None:
        json.dump(meta, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, ensure_ascii=False)
        raise SystemExit("no structured-output parameter style works; aborting grammar runs")

    inputs = sorted(glob.glob(os.path.join(CROPS, "*.png")))
    runs = []
    for gname, gpath in GRAMMARS.items():
        g = open(os.path.join(HERE, gpath), encoding="utf-8").read() if gpath else None
        for img in inputs:
            iname = os.path.splitext(os.path.basename(img))[0]
            b64 = base64.b64encode(open(img, "rb").read()).decode()
            body = {
                "model": MODEL, "temperature": 0, "max_tokens": 2048,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": PROMPT}]}],
                **grammar_fields(style, g),
            }
            t0 = time.perf_counter()
            try:
                r = post("/v1/chat/completions", body)
                err = None
            except Exception as ex:
                r, err = None, str(ex)[:1000]
            dt = time.perf_counter() - t0
            rec = {"grammar": gname, "input": iname, "latency_s": round(dt, 2), "error": err,
                   "request_params": {k: v for k, v in body.items() if k != "messages"} | {"prompt": PROMPT},
                   "response": r}
            if r:
                c = r["choices"][0]
                rec["text"] = c["message"]["content"]
                rec["finish_reason"] = c.get("finish_reason")
                rec["completion_tokens"] = r.get("usage", {}).get("completion_tokens")
                rec["prompt_tokens"] = r.get("usage", {}).get("prompt_tokens")
            json.dump(rec, open(os.path.join(OUT, f"{gname}__{iname}.json"), "w"), indent=2, ensure_ascii=False)
            print(f"{gname:22s} {iname:36s} {dt:7.1f}s fin={rec.get('finish_reason')} "
                  f"tok={rec.get('completion_tokens')} err={err}")
            runs.append({k: rec.get(k) for k in ("grammar", "input", "latency_s", "finish_reason",
                                                 "completion_tokens", "prompt_tokens", "error")})
    meta["runs"] = runs
    json.dump(meta, open(os.path.join(OUT, "run_summary.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
