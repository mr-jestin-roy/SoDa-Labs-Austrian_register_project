"""Central configuration for the lexicon-guided HTR pipeline.

Everything can be overridden by environment variables so the same code runs
on the local Mac (mlx / API endpoint) and on the Nectar HPC vLLM server.
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).parent
PROJECT_ROOT = PKG_DIR.parent.parent  # SoDa-Labs-Austrian_register_project

# ── Corpus (Reichstelefonbuch 1942 digitisation output) ──────────────────────
RTB_MASTER_DIR = Path(os.getenv(
    "RTB_MASTER_DIR",
    PROJECT_ROOT / "Reichstelefonbuch_1942" / "RTB_results_1_2_3_4",
))
RTB_MASTER_GLOB = "master_RTB_*_clean.xlsx"

# ── Artifacts produced by build_lexicon.py ────────────────────────────────────
ARTIFACT_DIR = Path(os.getenv("LEXICON_ARTIFACT_DIR", PKG_DIR / "artifacts"))
LEXICON_PATH = ARTIFACT_DIR / "lexicon.json.gz"          # category -> {term: count}
TOWN_SURNAME_KG_PATH = ARTIFACT_DIR / "kg_town_surnames.json.gz"  # town -> {surname: count}

# ── VLM endpoint (OpenAI-compatible, e.g. vLLM on Nectar) ────────────────────
# Works for both `chandra` and Qwen3-VL — just change VLM_MODEL / VLM_BASE_URL.
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://localhost:8010/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "chandra")            # e.g. "Qwen/Qwen3-VL-8B-Instruct"
VLM_API_KEY = os.getenv("VLM_API_KEY", "EMPTY")          # vLLM ignores it
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "4096"))
VLM_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))
MAX_SEND_DIM = int(os.getenv("MAX_SEND_DIM", "1600"))    # px, see May13thHandoff.md

# ── Lexicon building heuristics ──────────────────────────────────────────────
MIN_SURNAME_FREQ = 2          # drop hapax OCR noise from the surname lexicon
MIN_GIVEN_NAME_FREQ = 3
MIN_OCCUPATION_FREQ = 3

# ── Retrieval defaults ───────────────────────────────────────────────────────
DEFAULT_PARISH = "Althofen (Kärnten)"
TOP_TOWN_SURNAMES = 150       # 1-hop: surnames attested in the parish town itself
TOP_STATE_SURNAMES = 250      # 2-hop: surnames aggregated over the same state
TOP_GLOBAL_GIVEN_NAMES = 120
TOP_GLOBAL_OCCUPATIONS = 80
FUZZY_SCORE_CUTOFF = 78.0     # rapidfuzz WRatio cutoff for draft-token matching
FUZZY_TOP_K = 5               # candidates kept per uncertain draft token
