"""Task A metrics from raw_outputs.jsonl -> loop_metrics.csv, loop_rate_by_condition.csv, per_page_table.csv"""
import json, re, bisect
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

WORK = Path("/mnt/work/step1_loops_attention")
LOOP_RUN = 5
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
LEAD_NUM = re.compile(r"^\s*[\d][\d\s/\-–.,]*")


def norm(line):
    return LEAD_NUM.sub("", line).strip().lower()


def char_to_token(ids, char_off):
    """smallest k such that decode(ids[:k]) covers char_off (binary search over prefix lengths)."""
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid + 1], skip_special_tokens=True)) > char_off:
            hi = mid
        else:
            lo = mid + 1
    return lo


def metrics(d):
    text = d["text"] or ""
    lines, offs, pos = [], [], 0
    for raw in text.split("\n"):
        s = raw.strip()
        if s and not s.startswith("```"):
            lines.append(s); offs.append(pos + (len(raw) - len(raw.lstrip())))
        pos += len(raw) + 1
    n = [norm(l) for l in lines]
    best, best_start, run, start = (1 if n else 0), 0, 1, 0
    for i in range(1, len(n)):
        if n[i] == n[i - 1]:
            run += 1
        else:
            run, start = 1, i
        if run > best:
            best, best_start = run, start
    # block (period 1..4) repetition, to catch cycles of several lines
    blk_rep, blk_p, blk_start = 1, 1, 0
    for p in range(1, 5):
        r = 0
        for i in range(p, len(n)):
            r = r + 1 if n[i] == n[i - p] else 0
            reps = r // p + 1
            if reps > blk_rep:
                blk_rep, blk_p, blk_start = reps, p, i - r - p + 1
    looped = best >= LOOP_RUN
    out = dict(req_id=d["req_id"], page=d["page"], condition=d["condition"], n_cells=len(d["cells"]),
               img_w=d["w"], img_h=d["h"], prompt_tokens=d["prompt_tokens"],
               finish_reason=d["finish_reason"], completion_tokens=d["completion_tokens"],
               hit_max_tokens=d["finish_reason"] == "length", n_lines=len(lines),
               n_distinct_texts=len(set(n)), longest_run=best, loop=looped,
               loop_line_text=lines[best_start] if looped else "",
               block_max_repeats=blk_rep, block_period=blk_p, block_loop=blk_rep >= LOOP_RUN,
               loop_run_start_token=None, loop_onset_token=None)
    ids = d["gen_token_ids"]
    if looped:
        out["loop_run_start_token"] = char_to_token(ids, offs[best_start])
        out["loop_onset_token"] = char_to_token(ids, offs[best_start + 1])  # first repeated line
    if blk_rep >= LOOP_RUN:
        out["block_loop_onset_token"] = char_to_token(ids, offs[blk_start + blk_p])
    return out


def main():
    recs = {}
    for line in open(WORK / "raw_outputs.jsonl"):
        d = json.loads(line)
        if "error" not in d:
            recs[d["req_id"]] = d
    df = pd.DataFrame([metrics(d) for d in recs.values()]).sort_values(["condition", "page", "req_id"])
    df.to_csv(WORK / "loop_metrics.csv", index=False)
    ref = pd.read_csv("/mnt/data/rtb4_p1266_1285_reference.csv")
    ref_n = ref.groupby(ref.batch.str.extract(r"page_(\d+)")[0].astype(int)).size().rename("ref_entries")
    g = df.groupby("condition")
    summ = pd.DataFrame(dict(
        n_outputs=g.size(), n_cells=g.n_cells.sum(), loop_rate=g.loop.mean(), n_loop=g.loop.sum(),
        block_loop_rate=g.block_loop.mean(), max_tokens_rate=g.hit_max_tokens.mean(),
        n_max_tokens=g.hit_max_tokens.sum(), mean_completion_tokens=g.completion_tokens.mean(),
        total_lines=g.n_lines.sum(), mean_prompt_tokens=g.prompt_tokens.mean()))
    # loop rate per cell covered (a GRID8 output covers ~8 cells)
    summ["loops_per_100_cells"] = 100 * summ.n_loop / summ.n_cells
    summ.to_csv(WORK / "loop_rate_by_condition.csv")
    pp = df.groupby(["page", "condition"]).agg(outputs=("req_id", "size"), loops=("loop", "sum"),
                                               max_tok=("hit_max_tokens", "sum"), lines=("n_lines", "sum"),
                                               distinct=("n_distinct_texts", "sum"),
                                               max_run=("longest_run", "max")).unstack("condition")
    pp.columns = [f"{c}_{a}" for a, c in pp.columns]
    pp = pp.join(ref_n)
    pp.to_csv(WORK / "per_page_table.csv")
    print(summ.to_string()); print(pp.to_string())


if __name__ == "__main__":
    main()
