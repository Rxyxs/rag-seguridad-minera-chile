"""Servicio FastAPI: clasificación de severidad de incidentes + consultas RAG al DS 132.

Ejecutar desde la raíz del repositorio con:
    uvicorn src.api.main:app --reload

Variables de entorno:
    RAG_LLM_BACKEND   "auto" (default) | "ollama" | "openai" | "extractive"
    OPENAI_API_KEY    requerido solo si se usa el backend "openai"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.nlp.severity_classifier import MODELS_DIR, get_embedder, predict_severity
from src.rag.pipeline import RAGPipeline

LLM_BACKEND_PREFERENCE = os.environ.get("RAG_LLM_BACKEND", "auto")

app = FastAPI(
    title="RAG Seguridad Minera Chile API",
    description=(
        "Clasificación de severidad de incidentes de seguridad minera y consultas RAG "
        "sobre el Reglamento de Seguridad Minera (DS 132, SERNAGEOMIN)."
    ),
    version="1.0.0",
)

_embedder = None
_severity_model = None
_rag_pipeline: RAGPipeline | None = None


def _get_severity_components():
    global _embedder, _severity_model
    if _severity_model is None:
        import joblib

        model_path = MODELS_DIR / "severity_classifier.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró {model_path}. Corre primero: python -m src.nlp.severity_classifier"
            )
        _severity_model = joblib.load(model_path)
        _embedder = get_embedder()
    return _embedder, _severity_model


def _get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(llm_preference=LLM_BACKEND_PREFERENCE)
    return _rag_pipeline


class IncidentClassificationRequest(BaseModel):
    narrative_text: str = Field(min_length=10, description="Narrativa del incidente a clasificar")


class IncidentClassificationResponse(BaseModel):
    narrative_text: str
    predicted_severity: str
    confidence: float
    probabilities: dict[str, float]


class RAGQueryRequest(BaseModel):
    question: str = Field(min_length=5, description="Pregunta sobre el Reglamento de Seguridad Minera")


class RAGSource(BaseModel):
    article_number: str
    section: str


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    backend: str
    sources: list[RAGSource]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "severity_model_loaded": _severity_model is not None,
        "rag_pipeline_loaded": _rag_pipeline is not None,
    }


@app.post("/classify-incident", response_model=IncidentClassificationResponse)
def classify_incident(payload: IncidentClassificationRequest) -> IncidentClassificationResponse:
    embedder, model = _get_severity_components()
    prediction = predict_severity(payload.narrative_text, embedder, model)
    return IncidentClassificationResponse(
        narrative_text=payload.narrative_text,
        predicted_severity=prediction["predicted_severity"],
        confidence=prediction["confidence"],
        probabilities=prediction["probabilities"],
    )


@app.post("/rag-query", response_model=RAGQueryResponse)
def rag_query(payload: RAGQueryRequest) -> RAGQueryResponse:
    pipeline = _get_rag_pipeline()
    result = pipeline.query(payload.question)
    return RAGQueryResponse(
        question=result["question"],
        answer=result["answer"],
        backend=result["backend"],
        sources=[RAGSource(**s) for s in result["sources"]],
    )
