"""Task A: free decoding with Qwen3-VL-8B via the shared vLLM server (localhost:8010, model 'qwen3vl').
temperature 0, max_tokens 3000, no grammar. Exact generated token ids are kept
(logprobs + return_tokens_as_token_ids) so Task B can teacher-force the identical sequence.
Resumable: skips req_ids already in raw_outputs.jsonl.
"""
import json, base64, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

WORK = Path("/mnt/work/step1_loops_attention")
URL = "http://localhost:8010/v1/chat/completions"
MAX_TOKENS = 3000
WORKERS = 4  # shared server has max-num-seqs 8; leave room for the other agent

PROMPT = (
    "This image shows one or more cells cut from a page of the 1942 German telephone directory "
    "(Reichstelefonbuch). Transcribe the text. For each cell, first write its region heading "
    "(the bold town name) on its own line, then write one directory entry per line in the form "
    "`extension text`, where extension is the telephone number printed before the entry. "
    "Output plain text only."
)


def call(rec):
    b64 = base64.b64encode(Path(rec["image"]).read_bytes()).decode()
    payload = {
        "model": "qwen3vl",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": PROMPT}]}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
    }
    t0 = time.time()
    for attempt in range(3):
        try:
            r = requests.post(URL, json=payload, timeout=1800)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                return dict(rec, error=str(e))
            time.sleep(10)
    j = r.json()
    ch = j["choices"][0]
    toks = [int(t["token"].split(":", 1)[1]) for t in (ch.get("logprobs") or {}).get("content", [])]
    return dict(rec, text=ch["message"]["content"], finish_reason=ch["finish_reason"],
                completion_tokens=j["usage"]["completion_tokens"], prompt_tokens=j["usage"]["prompt_tokens"],
                gen_token_ids=toks, seconds=round(time.time() - t0, 1), prompt=PROMPT)


def main():
    manifest = json.load(open(WORK / "manifest.json"))
    out = WORK / "raw_outputs.jsonl"
    done = set()
    if out.exists():
        for line in open(out):
            d = json.loads(line)
            if "error" not in d:
                done.add(d["req_id"])
    todo = [m for m in manifest if m["req_id"] not in done]
    # interleave conditions so a partial run still covers both
    todo.sort(key=lambda m: (m["condition"] != "GRID8", m["page"]))
    print(f"{len(todo)} to do, {len(done)} done", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex, open(out, "a") as f:
        futs = [ex.submit(call, m) for m in todo]
        for i, fu in enumerate(as_completed(futs)):
            d = fu.result()
            f.write(json.dumps(d, ensure_ascii=False) + "\n"); f.flush()
            print(i + 1, d["req_id"], d.get("finish_reason"), d.get("completion_tokens"), d.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
