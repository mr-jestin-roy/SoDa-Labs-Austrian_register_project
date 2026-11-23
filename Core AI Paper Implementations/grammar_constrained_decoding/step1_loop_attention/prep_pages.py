"""Step 1 prep: render RTB vol.4 pages 1266-1285 at 600 DPI, segment with cv_segment_test.py,
then build the request images for both conditions:
  SINGLE : each cell scaled with the old pipeline's scale_to_tile (max 560x900) -> one image
  GRID8  : cells (sorted by name, like the old pipeline) tiled 8 per image with build_grid/choose_cols
Same per-cell resolution in both conditions, so the only difference is cells-per-request.
All request images are padded (white) up to multiples of 32 px so the Qwen3-VL processor does no
further resizing and vLLM and HF see identical pixels.
"""
import os, sys, json, subprocess, importlib.util
from pathlib import Path
import cv2, numpy as np, fitz

WORK = Path("/mnt/work/step1_loops_attention")
DATA = Path("/mnt/data")
PAGES = range(1266, 1286)
GRID_N = 8

spec = importlib.util.spec_from_file_location("occ", DATA / "ocr_cells_chandra.py")
occ = importlib.util.module_from_spec(spec); spec.loader.exec_module(occ)


def pad32(img):
    h, w = img.shape[:2]
    H, W = -(-h // 32) * 32, -(-w // 32) * 32
    out = np.full((H, W, 3), 255, np.uint8)
    out[:h, :w] = img
    return out


def main():
    doc = fitz.open(DATA / "RTB_4_Karbow-Quittainen.pdf")
    manifest = []
    for p in PAGES:
        pdir = WORK / "pages" / f"p{p}"
        cells_dir = pdir / "cells"
        req_dir = pdir / "req"
        req_dir.mkdir(parents=True, exist_ok=True)
        png = pdir / f"page_{p}.png"
        if not png.exists():
            pix = doc[p - 1].get_pixmap(matrix=fitz.Matrix(600 / 72, 600 / 72))
            pix.pil_save(str(png), dpi=(600, 600))
        if not cells_dir.exists() or not any(cells_dir.glob("*.png")):
            env = dict(os.environ, INPUT_IMAGE=str(png), OUT_LINES=str(pdir / "lines.png"),
                       OUT_COLS=str(pdir / "cols.png"), OUT_CELLS=str(cells_dir))
            r = subprocess.run([sys.executable, str(DATA / "cv_segment_test.py")], env=env,
                               capture_output=True, text=True)
            (pdir / "segment.log").write_text(r.stdout + r.stderr)
            if r.returncode:
                print("segment failed", p, r.stderr[-500:]); continue
        cell_files = sorted(cells_dir.glob("*.png"))
        cells = [(f.name, cv2.imread(str(f))) for f in cell_files]
        for name, im in cells:
            t = pad32(occ.scale_to_tile(im, occ.MAX_TILE_W, occ.MAX_TILE_H))
            out = req_dir / f"single_{name}"
            cv2.imwrite(str(out), t)
            manifest.append(dict(page=p, condition="SINGLE", req_id=f"p{p}_single_{name[:-4]}",
                                 cells=[name], image=str(out), w=t.shape[1], h=t.shape[0]))
        for bi in range(0, len(cells), GRID_N):
            batch = cells[bi:bi + GRID_N]
            ims = [im for _, im in batch]
            cols = occ.choose_cols(ims)
            g = pad32(occ.build_grid(ims, cols, occ.PADDING))
            out = req_dir / f"grid8_b{bi // GRID_N + 1:02d}.png"
            cv2.imwrite(str(out), g)
            manifest.append(dict(page=p, condition="GRID8", req_id=f"p{p}_grid8_b{bi // GRID_N + 1:02d}",
                                 cells=[n for n, _ in batch], image=str(out), w=g.shape[1], h=g.shape[0],
                                 grid_cols=cols))
        print(p, "cells", len(cells), flush=True)
    with open(WORK / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print("requests:", len(manifest))


if __name__ == "__main__":
    main()
