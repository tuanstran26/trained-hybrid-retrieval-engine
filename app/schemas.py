from typing import List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Technical search query")
    top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Number of results")


class SearchResultItem(BaseModel):
    rank: int
    passage_id: int
    score: float
    text: str


class LatencyBreakdown(BaseModel):
    dense_encode_and_search_ms: float
    bm25_search_ms: float
    rrf_fusion_ms: float
    cross_encoder_rerank_ms: float
    total_service_ms: float


class SearchResponse(BaseModel):
    query: str
    top_k: int
    latency: LatencyBreakdown
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_name: Optional[str] = None
