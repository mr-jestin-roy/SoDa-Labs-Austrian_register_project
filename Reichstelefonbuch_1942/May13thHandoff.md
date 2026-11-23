# May 13th Handoff — OCR Pipeline Issues & Mitigations

**Project:** Reichstelefonbuch 1942 Digitisation  
**Date:** 2026-05-13  
**Scripts affected:** `ocr_cells_chandra.py`, `cv_segment_test.py`

---

## Issue 1: Vision Token Overflow / Mid-stream JSON Truncation

### What is happening

`call_chandra()` hardcodes `"max_tokens": 8192`. vLLM's total context budget is:

```
max_model_len = input_tokens (image + prompt) + output_tokens
```

A 3800×3800 composite grid (the current `GRID_3x3_MAX_AREA` ceiling) at 600 DPI encodes to
4,000–8,000+ vision tokens depending on Chandra's patch size. Combined with the prompt (~250
tokens) and JSON output for 9 cells (~200–400 tokens each = up to 3,600 tokens), the total
easily exceeds a 16K context window — causing generation to be cut mid-stream, producing
invalid/truncated JSON.

### Mitigations

#### a) Cap composite image size before base64 encoding — `ocr_cells_chandra.py`

Highest leverage. Reduces vision token count by 4–8× with negligible quality loss for text OCR.

```python
MAX_SEND_DIM = 1600  # px — cap longest edge before encoding

def img_to_b64(img):
    h, w = img.shape[:2]
    scale = min(MAX_SEND_DIM / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return base64.standard_b64encode(buf.tobytes()).decode()
```

#### b) Dynamic `max_tokens` scaled to batch size — `ocr_cells_chandra.py`

Replace the hardcoded value in `call_chandra()` with a per-batch estimate:

```python
# ~400 output tokens per cell; cap at 8192
max_output_tokens = min(8192, 400 * len(batch))

payload = {
    ...
    "max_tokens": max_output_tokens,
    ...
}
```

#### c) vLLM server-side hyperparameters

Align server launch flags to accommodate large vision inputs:

```bash
vllm serve chandra \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8
```

> Key insight: `max_tokens` in the API request is silently capped by
> `max_model_len − input_tokens`. If the image consumes 12K tokens and
> `max_model_len=16K`, the effective output budget is only 4K regardless of
> what the client requests.

#### d) Detect truncation and retry with a smaller batch — `ocr_cells_chandra.py`

```python
def is_truncated(raw: str) -> bool:
    stripped = raw.strip()
    return not (stripped.endswith("]") or stripped.endswith("}"))

# In the main batch loop — after call_chandra():
if is_truncated(raw):
    # split batch in half and re-submit each half separately
```

---

## Issue 2: Hallucinations on Heavy Pages (Oversized Unbroken Solid Columns)

### What is happening

`cv_segment_test.py` detects horizontal rules with `H_FRAC_OF_COL = 0.05`. When a column has
**no printed horizontal dividers** (a dense, unbroken block of entries), `row_splits` is empty
and `y_bounds = [0, H]` — the full page height (~7,500 px at 600 DPI) becomes a single cell.

When that giant cell is batched into a composite grid and sent to Chandra:
- The model cannot attend reliably to entries near the bottom (attention dilution over long
  visual sequences).
- It begins hallucinating: repeated entries, merged town names, or collapsed rows — classic
  "copy drift" in VLMs overwhelmed by oversized context.

### Mitigations

#### a) Hard cap on cell height with overlap splitting — `cv_segment_test.py`

Add a subdivision step after building `y_bounds` for each column. The overlap prevents cutting
an entry mid-line and losing content at a boundary.

```python
MAX_CELL_H_PX = 1200  # at 600 DPI ≈ 2 inches of text; tune as needed

def subdivide_tall_cells(y_bounds, max_h, overlap=50):
    """Split y-intervals exceeding max_h into overlapping sub-intervals."""
    new_bounds = [y_bounds[0]]
    for i in range(len(y_bounds) - 1):
        y1, y2 = y_bounds[i], y_bounds[i + 1]
        if (y2 - y1) > max_h:
            pos = y1
            while pos < y2:
                new_bounds.append(min(pos + max_h, y2))
                pos += max_h - overlap
        else:
            new_bounds.append(y2)
    return sorted(set(new_bounds))

# Usage — replace direct y_bounds construction in the per-column loop:
y_bounds = subdivide_tall_cells([0] + row_splits + [H], MAX_CELL_H_PX)
```

#### b) Horizontal projection profile as fallback splitter — `cv_segment_test.py`

When no printed rules are detected, use text density (row sums of the binary image) to locate
natural whitespace gaps between town blocks:

```python
def find_whitespace_splits(col_thresh, cluster_gap):
    """Fallback: find inter-entry whitespace when no printed rules exist."""
    row_sums = col_thresh.sum(axis=1)
    smooth = np.convolve(row_sums, np.ones(5) / 5, mode='same')
    gaps = np.where(smooth < smooth.mean() * 0.1)[0]
    return cluster(gaps.tolist(), cluster_gap)

# In the per-column loop — use as fallback when row_splits is empty:
if not row_splits:
    row_splits = find_whitespace_splits(col_thresh, CLUSTER_GAP)
```

#### c) Route oversized cells to solo 1-per-request batches — `ocr_cells_chandra.py`

Pre-flight check before batching. Oversized cells are sent individually with the same prompt,
eliminating hallucination pressure from crowded grids:

```python
MAX_SOLO_AREA = 1000 * 1600  # px² threshold — tune to your typical cell size

oversized = [(n, im) for n, im in cells if im.shape[0] * im.shape[1] > MAX_SOLO_AREA]
normal    = [(n, im) for n, im in cells if im.shape[0] * im.shape[1] <= MAX_SOLO_AREA]

# Process oversized cells individually, then batch normal cells as usual
```

---

## Summary

| Issue | Root Cause | Recommended Fix | Where |
|-------|-----------|-----------------|-------|
| JSON truncation mid-stream | Large composite image consumes vision token budget, leaving insufficient room for output | Cap composite image longest edge at 1600 px before encoding | `ocr_cells_chandra.py` → `img_to_b64()` |
| `max_tokens` not scaling with batch | Fixed 8192 output cap is too low for 9-cell batches with many entries | Set `max_tokens = min(8192, 400 × n_cells)` dynamically | `ocr_cells_chandra.py` → `call_chandra()` |
| vLLM context budget mismatch | `max_model_len` not sized for large vision inputs | Launch vLLM with `--max-model-len 32768 --max-num-batched-tokens 32768` | vLLM server startup |
| Truncation not caught/retried | No detection of incomplete JSON responses | Add `is_truncated()` check and halve-batch retry | `ocr_cells_chandra.py` → main loop |
| Hallucination on solid columns | No H-rules detected → full page height becomes one cell | Hard cap `MAX_CELL_H_PX=1200` with 50 px overlap subdivision | `cv_segment_test.py` → per-column loop |
| Attention dilution on large cells | Oversized single cells overwhelm cross-attention in VLM | Route cells exceeding `MAX_SOLO_AREA` to individual requests | `ocr_cells_chandra.py` → batch builder |
| No fallback when rules absent | Pipeline relies solely on printed dividers for splitting | Horizontal projection profile as whitespace-gap fallback | `cv_segment_test.py` → per-column loop |
