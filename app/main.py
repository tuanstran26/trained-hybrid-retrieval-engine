from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from app.config import DEVICE
from app.engine import HybridSearchEngine
from app.schemas import HealthResponse, SearchRequest, SearchResponse

engine = HybridSearchEngine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.initialize()
    try:
        yield
    finally:
        engine.ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="Hybrid Retrieval & Two-Stage Reranking Engine",
    description="BM25 and dense retrieval with weighted RRF and cross-encoder reranking.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest) -> SearchResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    top_k = request.top_k if request.top_k is not None else 10
    try:
        results, latency = engine.search(query=request.query, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(
        query=request.query,
        top_k=top_k,
        latency=latency,
        results=results,
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return HealthResponse(
        status="healthy" if engine.ready else "starting",
        device=DEVICE,
        gpu_name=gpu_name,
    )
