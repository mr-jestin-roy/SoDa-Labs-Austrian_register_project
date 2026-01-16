#!/usr/bin/env python3
"""
Batch Chandra OCR processor for folders of images.
Uses Surya for layout detection + Chandra for table recognition & OCR.

Pipeline:
  1. Surya: Layout Detection → Identifies table bounding boxes
  2. Chandra: Table Recognition → Understands row/column structure
  3. Chandra: OCR/Transcription → Converts visual text to HTML/Markdown
  4. Post-processing: HTML → Clean Text → XLSX (optional)

Usage:
    python batch_chandra_ocr.py --input-dir ./inputs --output-dir ./output --method hf --to-xlsx
"""
import subprocess
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List
import json
import logging
from datetime import datetime
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
import threading
import time

# Restore the clear, structured logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chandra_batch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing a single image through the Surya+Chandra pipeline."""
    image_path: str
    status: str
    html_path: Optional[str] = None
    md_path: Optional[str] = None
    clean_txt_path: Optional[str] = None
    xlsx_path: Optional[str] = None
    error_msg: Optional[str] = None
    processing_time: float = 0.0
    peak_vram_mib: int = 0

class ChandraBatchProcessor:
    def __init__(self, input_dir: str, output_dir: str, method: str = "vllm", vllm_url: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.method = method
        self.vllm_url = vllm_url or "http://localhost:8010"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

        # Log initialization like your favorite version
        logger.info(f"Initialized processor: input={self.input_dir}, output={self.output_dir}, method={self.method}, vllm_url={self.vllm_url}")

    def get_image_files(self) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        images = sorted([f for f in self.input_dir.rglob("*") if f.suffix.lower() in extensions])
        logger.info(f"Found {len(images)} images in {self.input_dir}")
        return images

    def _monitor_vram(self, stop_event, vram_log):
        peak = 0
        while not stop_event.is_set():
            try:
                cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                current_usage = int(output.strip().split('\n')[0])
                if current_usage > peak:
                    peak = current_usage
            except Exception:
                pass
            time.sleep(1)
        vram_log.append(peak)

    def process_single_image(self, image_path: Path) -> ProcessingResult:
        """
        Run Chandra OCR on a single image with VRAM monitoring.
        Simplified version: No XLSX or TXT conversion.
        """
        start_time = time.time()
        vram_log = []
        stop_event = threading.Event()

        # Start VRAM monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_vram, args=(stop_event, vram_log))
        monitor_thread.start()

        try:
            basename = image_path.stem
            image_output_dir = self.output_dir / basename
            image_output_dir.mkdir(parents=True, exist_ok=True)

            # Standard Chandra CLI command
            command = [
                "chandra",
                str(image_path),
                str(image_output_dir),
                "--method", self.method,
                "--save-html"
            ]

            logger.info(f"Processing: {image_path.name}")

            env = os.environ.copy()
            if self.method == "vllm":
                env["VLLM_API_SERVER"] = self.vllm_url
                # Injecting your specific prompt details into the environment
                # Note: This depends on whether the 'chandra' CLI reads this specific env var.
                env["CHANDRA_CUSTOM_PROMPT"] = (
                    "Transcribe this table, which contains sterbRegister of Austrian Parish Records "
                    "of Pfarramt Pernegg a.d Mur, detect table using layout detection, table detection, "
                    "table recognition."
                )
                logger.info(f"   Using vLLM endpoint: {self.vllm_url}")

            # Run the OCR subprocess
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute allowance
                env=env
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return ProcessingResult(str(image_path), "failed", error_msg=error_msg)

            # Locate core outputs to confirm success
            html_path = image_output_dir / f"{basename}.html"
            md_path = image_output_dir / f"{basename}.md"

            if not html_path.exists():
                return ProcessingResult(str(image_path), "failed", error_msg="HTML output not generated")

            return ProcessingResult(
                image_path=str(image_path),
                status="success",
                html_path=str(html_path),
                md_path=str(md_path) if md_path.exists() else None,
                processing_time=time.time() - start_time
            )

        except subprocess.TimeoutExpired:
            return ProcessingResult(str(image_path), "failed", error_msg="Timeout (>900s)")
        except Exception as e:
            return ProcessingResult(str(image_path), "failed", error_msg=str(e))
        finally:
            # Signal monitor thread to stop and collect peak VRAM
            stop_event.set()
            monitor_thread.join()
            peak_val = vram_log[0] if vram_log else 0
            logger.info(f"📊 {image_path.name} | Peak VRAM: {peak_val} MiB")


    def process_batch(self, max_workers: int = 1):
        images = self.get_image_files()
        if not images: return

        logger.info(f"Starting batch processing with {max_workers} worker(s)...")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_single_image, img): img for img in images}
            for i, future in enumerate(as_completed(futures), 1):
                res = future.result()
                results.append(res)
                if res.status == "success":
                    logger.info(f"[{i}/{len(images)}] ✅ {Path(res.image_path).name} ({res.processing_time:.1f}s)")
                else:
                    logger.error(f"[{i}/{len(images)}] ❌ {Path(res.image_path).name}: {res.error_msg}")

        self._save_summary(results)

    def _save_summary(self, results):
        summary = {"timestamp": datetime.now().isoformat(), "results": [asdict(r) for r in results]}
        with open(self.output_dir / "processing_summary.json", "w") as f:
            json.dump(summary, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="./chandra_output")
    parser.add_argument("--method", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--to-xlsx", action="store_true")
    parser.add_argument("--vllm-url", default="http://localhost:8010")
    args = parser.parse_args()

    proc = ChandraBatchProcessor(args.input_dir, args.output_dir, method=args.method, vllm_url=args.vllm_url)
    proc.process_batch(max_workers=args.workers)

if __name__ == "__main__":
    main()
