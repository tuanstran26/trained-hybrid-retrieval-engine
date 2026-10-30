import os
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"

BI_MODEL_PATH = Path(os.getenv("BI_MODEL_PATH", MODELS_DIR / "bi-encoder-hard-neg-v2"))
CE_MODEL_PATH = Path(os.getenv("CE_MODEL_PATH", MODELS_DIR / "cross-encoder-stackoverflow-v2"))
CACHE_FILE = Path(os.getenv("CACHE_FILE", DATA_DIR / "id_text_lookup_question_body_cache.json"))
DAY3_RES_FILE = Path(os.getenv("DAY3_RES_FILE", DATA_DIR / "day3_final_reranked_results.json"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 256
MAX_QUERY_CHARS = 360
MAX_PASSAGE_CHARS = 900
RERANK_TOP_K_CANDIDATES = 30
DENSE_SEARCH_K = 100
BM25_SEARCH_K = 100

RRF_DENSE_WEIGHT = 0.88
RRF_BM25_WEIGHT = 0.12
RRF_K = 60.0
SCORE_BI_WEIGHT = 0.65
SCORE_CE_WEIGHT = 0.35
BM25_BONUS_WEIGHT = 0.04
