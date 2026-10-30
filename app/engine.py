import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    BI_MODEL_PATH,
    BM25_BONUS_WEIGHT,
    BM25_SEARCH_K,
    CACHE_FILE,
    CE_MODEL_PATH,
    DAY3_RES_FILE,
    DENSE_SEARCH_K,
    DEVICE,
    MAX_PASSAGE_CHARS,
    MAX_QUERY_CHARS,
    MAX_SEQ_LENGTH,
    RERANK_TOP_K_CANDIDATES,
    RRF_BM25_WEIGHT,
    RRF_DENSE_WEIGHT,
    RRF_K,
    SCORE_BI_WEIGHT,
    SCORE_CE_WEIGHT,
)


class HybridSearchEngine:
    """In-memory BM25 and dense retrieval with cross-encoder reranking."""

    def __init__(self) -> None:
        self.device = DEVICE
        self.bi_model = None
        self.ce_model = None
        self.ce_tokenizer = None
        self.faiss_index = None
        self.bm25 = None
        self.corpus_ids: List[int] = []
        self.corpus_texts: List[str] = []
        self.id_to_index: Dict[int, int] = {}
        self.corpus_embeddings = None
        self.ready = False

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"Required retrieval artifact not found: {path}")
        with path.open(encoding="utf-8") as file:
            return json.load(file)

    def _load_corpus(self) -> None:
        results = self._read_json(DAY3_RES_FILE)
        cache = self._read_json(CACHE_FILE)
        passage_lookup = {int(key): text for key, text in cache["p_lookup"].items()}

        candidate_ids = set(results["test_pos_ids"])
        for row in results["hybrid_top100_ids"]:
            candidate_ids.update(row[:RERANK_TOP_K_CANDIDATES])

        self.corpus_ids = sorted(candidate_ids)
        missing_ids = [pid for pid in self.corpus_ids if pid not in passage_lookup]
        if missing_ids:
            raise ValueError(
                f"Passage cache is missing {len(missing_ids)} IDs; first missing ID: {missing_ids[0]}"
            )

        self.corpus_texts = [passage_lookup[pid] for pid in self.corpus_ids]
        if not self.corpus_texts:
            raise ValueError("Retrieval corpus is empty")
        self.id_to_index = {pid: index for index, pid in enumerate(self.corpus_ids)}
        self.bm25 = BM25Okapi([text.lower().split() for text in self.corpus_texts])

    def _load_models(self) -> None:
        self.bi_model = SentenceTransformer(str(BI_MODEL_PATH), device=self.device)
        self.bi_model.max_seq_length = MAX_SEQ_LENGTH
        self.corpus_embeddings = self.bi_model.encode(
            self.corpus_texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        self.faiss_index = faiss.IndexFlatIP(self.corpus_embeddings.shape[1])
        self.faiss_index.add(self.corpus_embeddings)

        self.ce_tokenizer = AutoTokenizer.from_pretrained(str(CE_MODEL_PATH))
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(str(CE_MODEL_PATH)).to(self.device)
        self.ce_model.eval()
        if self.device == "cuda":
            self.ce_model.half()

    def initialize(self) -> None:
        started = time.perf_counter()
        print(f"[*] Initializing search engine on {self.device.upper()}...")
        self._load_corpus()
        self._load_models()
        self.ready = True
        elapsed = time.perf_counter() - started
        print(f"[*] Engine ready ({len(self.corpus_ids):,} passages) in {elapsed:.2f}s.")

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        score_range = np.ptp(scores)
        if score_range <= 1e-6:
            return np.zeros_like(scores, dtype=np.float32)
        return ((scores - np.min(scores)) / score_range).astype(np.float32)

    @staticmethod
    def _fuse_candidates(dense_ids: List[int], bm25_ids: List[int]) -> List[int]:
        fused_scores: Dict[int, float] = {}
        for weight, ranked_ids in ((RRF_DENSE_WEIGHT, dense_ids), (RRF_BM25_WEIGHT, bm25_ids)):
            for rank, passage_id in enumerate(ranked_ids, start=1):
                fused_scores[passage_id] = fused_scores.get(passage_id, 0.0) + weight / (RRF_K + rank)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return ranked_ids[:RERANK_TOP_K_CANDIDATES]

    def _rerank(
        self,
        query: str,
        query_embedding: np.ndarray,
        candidates: List[int],
        bm25_ids: List[int],
    ) -> np.ndarray:
        short_query = query[:MAX_QUERY_CHARS].strip()
        passages = [
            self.corpus_texts[self.id_to_index[pid]][:MAX_PASSAGE_CHARS].strip()
            for pid in candidates
        ]
        with torch.inference_mode():
            inputs = self.ce_tokenizer(
                [short_query] * len(candidates),
                passages,
                padding=True,
                truncation="only_second",
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            logits = self.ce_model(**inputs).logits.squeeze(-1)
            ce_scores = torch.sigmoid(logits).float().cpu().numpy()

        corpus_indices = [self.id_to_index[pid] for pid in candidates]
        bi_scores = np.dot(self.corpus_embeddings[corpus_indices], query_embedding[0])
        bm25_top_ids = set(bm25_ids[:15])
        bm25_bonus = np.array(
            [BM25_BONUS_WEIGHT if pid in bm25_top_ids else 0.0 for pid in candidates],
            dtype=np.float32,
        )
        return (
            SCORE_BI_WEIGHT * self._normalize(bi_scores)
            + SCORE_CE_WEIGHT * self._normalize(ce_scores)
            + bm25_bonus
        )

    def search(
        self, query: str, top_k: int = 10
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        if not self.ready:
            raise RuntimeError("Search engine is not initialized")
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        started = time.perf_counter()
        stage_started = time.perf_counter()
        query_embedding = self.bi_model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
        dense_count = min(DENSE_SEARCH_K, len(self.corpus_ids))
        _, dense_indices = self.faiss_index.search(query_embedding, dense_count)
        dense_ids = [self.corpus_ids[index] for index in dense_indices[0] if index >= 0]
        dense_ms = (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_count = min(BM25_SEARCH_K, len(bm25_scores))
        bm25_indices = np.argsort(bm25_scores)[-bm25_count:][::-1]
        bm25_ids = [self.corpus_ids[index] for index in bm25_indices]
        bm25_ms = (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        candidates = self._fuse_candidates(dense_ids, bm25_ids)
        fusion_ms = (time.perf_counter() - stage_started) * 1000

        stage_started = time.perf_counter()
        final_scores = self._rerank(query, query_embedding, candidates, bm25_ids)
        rerank_ms = (time.perf_counter() - stage_started) * 1000
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        results = [
            {
                "rank": rank,
                "passage_id": candidates[index],
                "score": float(final_scores[index]),
                "text": self.corpus_texts[self.id_to_index[candidates[index]]],
            }
            for rank, index in enumerate(ranked_indices, start=1)
        ]
        latency = {
            "dense_encode_and_search_ms": round(dense_ms, 2),
            "bm25_search_ms": round(bm25_ms, 2),
            "rrf_fusion_ms": round(fusion_ms, 2),
            "cross_encoder_rerank_ms": round(rerank_ms, 2),
            "total_service_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return results, latency
