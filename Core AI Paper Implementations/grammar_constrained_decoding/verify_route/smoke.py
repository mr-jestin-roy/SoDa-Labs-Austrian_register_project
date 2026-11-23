"""smoke.py -- one-arm-at-a-time smoke tests for the verify/route layer (Qwen3-VL around Chandra OCR-2).

Arms (run one per invocation):
  A  Chandra OCR-2 alone      : transcribe the cell                       (port 8011, served 'chandra')
  B  Qwen3-VL-8B alone        : transcribe the cell                       (port 8010, served 'qwen3vl')
  C  Chandra -> Qwen verifies : per output line, P(Yes) that the cell image shows exactly that line
  D  Route tall cells         : Qwen classifies the cell; tall cells are cut into full-resolution strips
                                (no 900-px downscale) and Chandra reads each strip
  E  Pick between readings    : where A and B disagree on a line, Qwen picks A or B from the image
Usage: python smoke.py ARM [cell_id ...]      (defaults: 3 cells from page 1266: medium, tall, medium)
Output: /mnt/work/verify_route/smoke_<ARM>.jsonl and a printed side-by-side with the silver reference.
Note: the silver reference was itself produced by an earlier Chandra run, so it favours arm A.
"""
import base64, html, io, json, math, re, sys, time, uuid
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd, requests
from PIL import Image

WORK = Path("/mnt/work/verify_route"); WORK.mkdir(parents=True, exist_ok=True)
STEP1 = Path("/mnt/work/step1_loops_attention")
MAN = {r["req_id"]: r for r in json.load(open(STEP1 / "manifest.json"))}
REF = pd.read_csv("/mnt/data/rtb4_p1266_1285_reference.csv")
EP = {"chandra": "http://localhost:8011/v1/chat/completions", "qwen3vl": "http://localhost:8010/v1/chat/completions"}
PROMPT = (
    "This image is one cell of the 1942 German Reichstelefonbuch (telephone directory). "
    "Transcribe it exactly as printed. For each town/region heading in the cell, first output the "
    "heading on its own line, then one line per directory entry in the form `extension text`, "
    "where extension is the telephone number printed at the left of the entry and text is the "
    "rest of the entry. If an entry wraps over several printed lines, join it into one line. "
    "Output only the transcription, no commentary."
)
DEFAULT = ["p1266_single_col3_sec05", "p1266_single_col2_sec02", "p1266_single_col4_sec02"]
MAX_TOKENS = 2048
STRIP_H, STRIP_OVERLAP = 700, 80          # arm D: full-res strip height and overlap (px)


def b64img(img):
    if isinstance(img, (str, Path)):
        return base64.b64encode(open(img, "rb").read()).decode()
    buf = io.BytesIO(); img.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def chat(model, img, text, max_tokens=MAX_TOKENS, logprobs=False, salt=True):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64img(img)}"}},
                {"type": "text", "text": text}]}]}
    if logprobs:
        body.update(logprobs=True, top_logprobs=20)
    if salt:
        body["cache_salt"] = uuid.uuid4().hex
    t = time.time(); r = requests.post(EP[model], json=body, timeout=900); r.raise_for_status()
    j = r.json(); c = j["choices"][0]
    return c, j["usage"], time.time() - t


def transcribe(model, img):
    c, u, dt = chat(model, img, PROMPT)
    return {"text": c["message"]["content"], "finish": c["finish_reason"], "tok": u["completion_tokens"],
            "prompt_tok": u["prompt_tokens"], "s": round(dt, 2)}


def lines(text):
    """One entry per line. Chandra OCR-2 answers in its own HTML (ul/li, table/tr/td, h2) whatever the
    prompt says, so flatten that: each li / tr / heading / paragraph is a line, td cells joined by a space,
    and a table row whose first cell is empty is a wrapped continuation of the previous entry."""
    text = text or ""
    if "<" not in text:
        return [l.strip() for l in text.splitlines() if l.strip()]
    out = []
    for row in re.split(r"</li>|</tr>|</h\d>|</p>|<br\s*/?>|\n", text):
        cont = bool(re.match(r"\s*(<[^>]+>\s*)*<tr>\s*<td>\s*</td>", row))
        row = re.sub(r"</td>\s*<td>", " ", row)
        row = html.unescape(re.sub(r"<[^>]+>", " ", row))
        row = re.sub(r"\s+", " ", row).strip()
        if not row:
            continue
        if cont and out:
            out[-1] += " " + row
        else:
            out.append(row)
    return out


def p_choice(c, options):
    """Normalised probability over option tokens from the first generated token's top_logprobs."""
    top = c["logprobs"]["content"][0]["top_logprobs"]
    p = {o: 0.0 for o in options}
    for t in top:
        k = t["token"].strip().strip(".").upper()
        for o in options:
            if k == o.upper():
                p[o] += math.exp(t["logprob"])
    s = sum(p.values())
    return {o: (v / s if s else float("nan")) for o, v in p.items()}, s


def verify_line(img, line):
    q = ("Look at this cell of a 1942 German telephone directory. Is the following entry printed in the "
         f"image EXACTLY as written (same spelling, same number)?\n\n{line}\n\nAnswer Yes or No.")
    c, u, dt = chat("qwen3vl", img, q, max_tokens=1, logprobs=True, salt=False)   # image prefix cache reused
    p, mass = p_choice(c, ["Yes", "No"])
    return {"line": line, "p_yes": round(p["Yes"], 4), "mass": round(mass, 3), "s": round(dt, 3)}


def pick(img, a, b):
    q = ("Look at this cell of a 1942 German telephone directory. Two OCR systems read one entry differently.\n"
         f"A: {a}\nB: {b}\nWhich reading matches the image exactly? Answer A or B.")
    c, u, dt = chat("qwen3vl", img, q, max_tokens=1, logprobs=True, salt=False)
    p, mass = p_choice(c, ["A", "B"])
    return {"A": a, "B": b, "pA": round(p["A"], 4), "mass": round(mass, 3), "s": round(dt, 3)}


def route(img):
    q = ("This is one cell cut from a page of a 1942 German telephone directory. Classify it. "
         "A: blank or only a rule/border. B: an advertisement or decorative block. "
         "C: directory entries, short (under about 25 printed lines). "
         "D: directory entries, long (25 or more printed lines). Answer A, B, C or D.")
    c, u, dt = chat("qwen3vl", img, q, max_tokens=1, logprobs=True)
    p, mass = p_choice(c, list("ABCD"))
    return max(p, key=p.get), p, round(dt, 3)


def strips(cell_png):
    im = Image.open(cell_png).convert("RGB"); W, H = im.size; out, y = [], 0
    while True:
        out.append(im.crop((0, y, W, min(H, y + STRIP_H))))
        if y + STRIP_H >= H: break
        y += STRIP_H - STRIP_OVERLAP
    return out, (W, H)


def merge_strip_lines(chunks):
    """Concatenate strip transcriptions, dropping lines repeated across the overlap seam."""
    out = []
    for ch in chunks:
        ls = lines(ch)
        k = 0
        for n in range(min(6, len(ls), len(out)), 0, -1):
            if all(SequenceMatcher(None, out[-n + i], ls[i]).ratio() > 0.85 for i in range(n)):
                k = n; break
        out += ls[k:]
    return out


def ref_lines(rid):
    """Whole-page reference: the reference's `cells` names come from an older segmentation run and do
    not line up with step 1's cells, so (like score.py) a prediction counts if it matches any line on the page."""
    r = MAN[rid]
    df = REF[REF.page == r["page"]]
    return [f"{int(e) if pd.notna(e) and str(e).replace('.0','').isdigit() else e} {t}".strip()
            for e, t in zip(df.phone_extension, df.transcribed_text.fillna(""))]


def best(l, cands):
    return max((SequenceMatcher(None, l, c).ratio() for c in cands), default=0.0)


def show(tag, pred, ref):
    m = sum(best(l, ref) >= 0.8 for l in pred)
    print(f"  [{tag}] lines={len(pred)} matched_to_page_ref(>=0.8)={m} ({m/max(1,len(pred)):.0%}) longest_repeat={longest_repeat(pred)}")


def longest_repeat(ls):
    b = cur = 0
    for i, l in enumerate(ls):
        cur = cur + 1 if i and l == ls[i - 1] else 1; b = max(b, cur)
    return b


PARISH_Q = ("This is a page of a handwritten Austrian Catholic parish register (Pfarre Pernegg an der Mur). "
            "An OCR system transcribed one table row as follows (cells separated by | ):\n\n{row}\n\n"
            "Is this a correct transcription of a row on this page (names, places, dates and numbers as written)? "
            "Answer Yes or No.")


def verify_html(img, html_in, html_out, threshold=0.5):
    """Arm V (JEV-style check for parish tables): ask Qwen3-VL Yes/No for every non-empty body row of the
    OCR table, then append two columns -- JEV_p_yes and JEV_review ('check' when p_yes < threshold) -- so the
    existing convert_html_to_xlsx.py carries them into the spreadsheet unchanged."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(open(html_in, encoding="utf-8").read(), "html.parser")
    stats = {"rows": 0, "checked": 0, "flagged": 0, "s": 0.0}
    for table in soup.find_all("table"):
        head = [tr for tr in table.find_all("tr") if tr.find("th")]
        if head:
            for name in ("JEV_p_yes", "JEV_review"):
                th = soup.new_tag("th", rowspan=str(len(head))); th.string = name; head[0].append(th)
        for tr in table.find_all("tr"):
            if tr.find("th"):
                continue
            stats["rows"] += 1
            cells = [re.sub(r"\s+", " ", td.get_text(" ")).strip() for td in tr.find_all("td")]
            row = " | ".join(c for c in cells if c)
            p, flag = "", ""
            if row:
                c, u, dt = chat("qwen3vl", img, PARISH_Q.format(row=row[:1500]), max_tokens=1, logprobs=True, salt=False)
                pv, mass = p_choice(c, ["Yes", "No"])
                p = f"{pv['Yes']:.3f}"; flag = "check" if pv["Yes"] < threshold else ""
                stats["checked"] += 1; stats["flagged"] += bool(flag); stats["s"] += dt
            for v in (p, flag):
                td = soup.new_tag("td"); td.string = v; tr.append(td)
    Path(html_out).parent.mkdir(parents=True, exist_ok=True)
    open(html_out, "w", encoding="utf-8").write(str(soup))
    stats["s"] = round(stats["s"], 2)
    return stats


def main():
    arm = sys.argv[1].upper(); ids = sys.argv[2:] or DEFAULT
    if arm == "V":                     # python smoke.py V <image> <html_in> <html_out>
        print(json.dumps(verify_html(*ids[:3])))
        return
    out = open(WORK / f"smoke_{arm}.jsonl", "w")
    for rid in ids:
        r = MAN[rid]; img = r["image"]; ref = ref_lines(rid)
        cell_png = STEP1 / "pages" / f"p{r['page']}" / "cells" / r["cells"][0]
        print(f"\n== {arm} {rid}  req_img={r['w']}x{r['h']}  full_cell={Image.open(cell_png).size}")
        rec = {"arm": arm, "req_id": rid}
        if arm in ("A", "B"):
            res = transcribe("chandra" if arm == "A" else "qwen3vl", img); rec.update(res)
            print(f"  finish={res['finish']} tok={res['tok']} prompt_tok={res['prompt_tok']} {res['s']}s")
            show(arm, lines(res["text"]), ref)
            for l in lines(res["text"]): print(f"   {'ok' if best(l, ref) >= 0.8 else '--'} |", l[:90])
        elif arm == "C":
            a = transcribe("chandra", img); rec["chandra"] = a
            t = time.time(); v = [verify_line(img, l) for l in lines(a["text"])]
            rec["verify"] = v; rec["verify_s"] = round(time.time() - t, 2)
            print(f"  chandra {a['tok']} tok {a['s']}s | verify {len(v)} lines in {rec['verify_s']}s")
            for x in v:
                ok = best(x["line"], ref) >= 0.8
                print(f"   p_yes={x['p_yes']:.3f} mass={x['mass']:.2f} ref_match={'Y' if ok else 'n'} | {x['line'][:70]}")
        elif arm == "D":
            cls, p, dt = route(img); rec.update(route=cls, route_p={k: round(v, 3) for k, v in p.items()})
            print(f"  route={cls} p={rec['route_p']} ({dt}s)")
            if cls == "D" or r["h"] >= 900:
                parts, size = strips(cell_png); rec["strips"] = len(parts)
                res = [transcribe("chandra", s) for s in parts]
                merged = merge_strip_lines([x["text"] for x in res])
                rec.update(strip_runs=res, text="\n".join(merged))
                print(f"  {len(parts)} strips of {size[0]}x<= {STRIP_H}: finish={[x['finish'] for x in res]} "
                      f"tok={[x['tok'] for x in res]} {sum(x['s'] for x in res):.1f}s")
            else:
                res = transcribe("chandra", img); rec.update(res); merged = lines(res["text"])
                print(f"  single pass: finish={res['finish']} tok={res['tok']} {res['s']}s")
            show("D", merged, ref)
        elif arm == "E":
            a, b = transcribe("chandra", img), transcribe("qwen3vl", img)
            la, lb = lines(a["text"]), lines(b["text"])
            sm = SequenceMatcher(None, la, lb, autojunk=False); picks = []
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "replace":
                    for x, y in zip(la[i1:i2], lb[j1:j2]):
                        if SequenceMatcher(None, x, y).ratio() > 0.6:      # same entry, different reading
                            picks.append(pick(img, x, y))
            rec.update(chandra=a, qwen=b, picks=picks)
            print(f"  chandra {len(la)} lines, qwen {len(lb)} lines, {len(picks)} disagreeing pairs judged")
            for x in picks:
                win = "A" if x["pA"] >= 0.5 else "B"
                ra, rb = best(x["A"], ref), best(x["B"], ref)
                print(f"   pA={x['pA']:.2f} pick={win} refsim A={ra:.2f} B={rb:.2f}\n      A| {x['A'][:70]}\n      B| {x['B'][:70]}")

        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()


if __name__ == "__main__":
    main()
