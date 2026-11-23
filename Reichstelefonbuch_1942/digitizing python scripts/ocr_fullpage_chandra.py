"""
ocr_fullpage_chandra.py
───────────────────────
Sends the WHOLE annotated page (cv_segment_lines.png) to the Chandra vLLM
endpoint in one shot. The page is scaled to fit within MAX_PAGE_W × MAX_PAGE_H
before sending. Chandra is asked to extract ALL regions it can see on the page.

Run:
    python ocr_fullpage_chandra.py

    # Use a different input image:
    INPUT_IMAGE=page_749-0749.png python ocr_fullpage_chandra.py

    # Override the vLLM URL:
    VLLM_URL=http://my-server:8010/v1/chat/completions python ocr_fullpage_chandra.py

Output:
    ocr_results_fullpage_chandra.json   — all regions extracted from the page
    ocr_fullpage_scaled.png             — the scaled image that was sent to Chandra
"""

import os, sys, json, base64, textwrap, time
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent   # Reichstelefonbuch_1942/
INPUT_IMAGE = BASE / os.getenv("INPUT_IMAGE", "cv_segment_lines.png")
OUT_JSON    = BASE / "ocr_results_fullpage_chandra.json"
OUT_SCALED  = BASE / "ocr_fullpage_scaled.png"

VLLM_URL    = os.getenv("VLLM_URL", "http://localhost:8010/v1/chat/completions")
MODEL       = "chandra"

MAX_PAGE_W  = None   # set to an int (e.g. 4096) to cap width; None = no scaling
MAX_PAGE_H  = None   # set to an int (e.g. 4096) to cap height; None = no scaling

PROMPT = textwrap.dedent("""\
    This is a full page from a 1942 German telephone directory (Reichstelefonbuch).
    The page contains multiple sections, each covering a specific town or region
    in Austria/Germany.

    For EVERY section visible on the page, extract the following as structured data:

    1. region_name  — the bold town/region heading at the top of each section
                      (e.g. "Mattsee", "Maulbronn", "Mauerbach")
    2. entries      — a list of telephone entries, each with:
         - extension : the number that appears before the name/text
                       (these are the local telephone extension numbers)
         - text      : the full name, business, or description that follows
                       the extension number on the same line

    Rules:
    - If a line has no extension number, it is a continuation of the previous
      entry or a sub-heading — include it in the text of the nearest entry above.
    - Preserve German characters (ä, ö, ü, ß).
    - If a section contains an advertisement or non-entry block, set entries to []
      and describe it briefly in region_name.
    - Extract ALL sections visible — do not stop early.

    Return ONLY valid JSON — no markdown fences, no extra commentary:
    [
      {
        "region_name": "...",
        "entries": [
          {"extension": "12", "text": "Müller, Hans, Bäckerei"},
          {"extension": "34", "text": "Gasthof zum Adler"}
        ]
      }
    ]
""")

# ── Load & scale image ────────────────────────────────────────────────────────
if not INPUT_IMAGE.exists():
    sys.exit(f"Image not found: {INPUT_IMAGE}")

img = cv2.imread(str(INPUT_IMAGE))
if img is None:
    sys.exit(f"cv2 could not read: {INPUT_IMAGE}")

h, w = img.shape[:2]
print(f"Input image  : {INPUT_IMAGE.name}  ({w}×{h} px)")

if MAX_PAGE_W or MAX_PAGE_H:
    cap_w = MAX_PAGE_W or w
    cap_h = MAX_PAGE_H or h
    scale = min(cap_w / w, cap_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Scaled to    : {new_w}×{new_h} px  (scale={scale:.3f})")
    else:
        img_scaled = img
        new_w, new_h = w, h
        print(f"No scaling needed")
else:
    img_scaled = img
    new_w, new_h = w, h
    print(f"Sending at full resolution: {w}×{h} px")

cv2.imwrite(str(OUT_SCALED), img_scaled)
print(f"Image saved  → {OUT_SCALED}")

# ── Encode & send ─────────────────────────────────────────────────────────────
_, buf = cv2.imencode(".png", img_scaled)
b64 = base64.standard_b64encode(buf.tobytes()).decode()

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ],
    "max_tokens": 16000,
    "temperature": 0.1,
}

print(f"\nSending to Chandra at {VLLM_URL} …")
start = time.time()
resp  = requests.post(VLLM_URL, json=payload, timeout=600)
if not resp.ok:
    sys.exit(f"HTTP {resp.status_code} — server rejected the request:\n{resp.text}")
elapsed = time.time() - start

raw = resp.json()["choices"][0]["message"]["content"]
print(f"Response     : {len(raw)} chars  ({elapsed:.1f}s)")

# ── Parse JSON ────────────────────────────────────────────────────────────────
clean = (raw
         .removeprefix("```json")
         .removeprefix("```")
         .removesuffix("```")
         .strip())
try:
    parsed = json.loads(clean)
    n_regions = len(parsed) if isinstance(parsed, list) else "?"
    print(f"Regions extracted: {n_regions}")
except json.JSONDecodeError as e:
    print(f"WARNING: JSON parse failed ({e}) — storing raw response")
    parsed = {"raw": raw}

# ── Save results ──────────────────────────────────────────────────────────────
result = {
    "input_image":  INPUT_IMAGE.name,
    "scaled_image": OUT_SCALED.name,
    "scaled_size":  f"{new_w}×{new_h}",
    "elapsed_s":    round(elapsed, 1),
    "data":         parsed,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"Results saved → {OUT_JSON}")
