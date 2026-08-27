"""Dataset de evaluación ampliado (27 consultas, una por cada artículo del
extracto curado del DS 132) y métricas de recuperación (MRR, NDCG@k, Context
Relevance) y de generación (Citation Faithfulness).

Faithfulness / Context Relevance sin LLM-juez: las definiciones "estándar" de
estas métricas (ej. RAGAS) usan un LLM para juzgar si la respuesta está
soportada por el contexto. Este proyecto no depende de tener un LLM externo
configurado (el backend extractivo es 100% funcional sin uno, ver
`pipeline.py`), así que ambas métricas se definen acá de forma determinista y
reproducible sin juez externo:

- **Context Relevance@k**: de los k documentos recuperados, qué fracción
  tiene un `article_number` que efectivamente está en el set de artículos
  relevantes etiquetado a mano para esa consulta -- una precision@k directa
  contra la verdad etiquetada.
- **Citation Faithfulness**: de los números de artículo que la respuesta
  *generada* cita explícitamente ("Artículo N"), qué fracción corresponde a
  artículos que de verdad estaban en el contexto recuperado -- una cita a un
  artículo que nunca se recuperó es una alucinación de cita, algo
  particularmente grave en un dominio legal/normativo. En modo extractivo
  (sin LLM) esta métrica es trivialmente 1.0 por construcción (el modo
  extractivo solo cita lo que efectivamente recuperó) -- se documenta esto
  explícitamente en vez de presentarlo como un resultado sorprendente; la
  métrica se vuelve real y no trivial en cuanto se usa un backend con LLM
  (Ollama/OpenAI), donde sí es posible que el modelo invente una cita.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from langchain_core.documents import Document

CITATION_RE = re.compile(r"[Aa]rt[íi]culo\s+(\d+)")


@dataclass
class EvalQuery:
    question: str
    relevant_articles: frozenset[str]
    topic: str


# Una consulta por artículo del extracto curado (27 artículos, ver
# data/ds132_sernageomin.txt), agrupadas por tema -- cubre investigación de
# accidentes, emergencias/rescate, reglamentos internos, fortificación y
# acuñadura, ventilación, tránsito de equipos de gran tonelaje, tronadura y
# explosivos, y sanciones.
QUERY_DATASET: list[EvalQuery] = [
    EvalQuery("¿Qué debe hacer la empresa minera cuando ocurre un accidente con lesiones a un trabajador?", frozenset({"76"}), "accidentes"),
    EvalQuery("¿Qué lesiones graves deben informarse inmediatamente a la Dirección Regional del Servicio?", frozenset({"77"}), "accidentes"),
    EvalQuery("¿Qué debe disponer la empresa minera respecto a trabajadores capacitados en primeros auxilios?", frozenset({"73"}), "emergencias"),
    EvalQuery("¿Cuántos vehículos de rescate debe haber dentro de un radio de cinco kilómetros de la faena?", frozenset({"74"}), "emergencias"),
    EvalQuery("¿Qué son las Brigadas de Rescate Minero y cuándo son obligatorias?", frozenset({"75"}), "emergencias"),
    EvalQuery("¿Qué reglamentos internos específicos debe elaborar la empresa minera?", frozenset({"78"}), "reglamentos_internos"),
    EvalQuery("¿En qué condiciones se puede dejar un sector subterráneo sin fortificación?", frozenset({"157"}), "fortificacion"),
    EvalQuery("¿Cada cuánto se debe inspeccionar una galería que no está fortificada?", frozenset({"158"}), "fortificacion"),
    EvalQuery("¿Qué se debe hacer en los piques de tránsito de personal que no están fortificados?", frozenset({"160"}), "fortificacion"),
    EvalQuery("¿Está permitido acceder a un lugar de la mina sin fortificar antes de acuñar?", frozenset({"161"}), "fortificacion"),
    EvalQuery("¿Qué debe consignar el procedimiento de acuñadura permanente de la Administración?", frozenset({"162"}), "fortificacion"),
    EvalQuery("¿Qué precaución se debe tomar al acuñar cerca de conductores eléctricos?", frozenset({"163"}), "fortificacion"),
    EvalQuery("¿Qué elementos de protección personal se requieren para trabajos desde una plataforma suspendida?", frozenset({"174"}), "fortificacion"),
    EvalQuery("¿Cuál es el caudal mínimo de aire fresco por persona en una mina subterránea?", frozenset({"138"}), "ventilacion"),
    EvalQuery("¿Con qué frecuencia se debe hacer el aforo de ventilación de la mina?", frozenset({"139"}), "ventilacion"),
    EvalQuery("¿Qué distancia máxima puede tener la tubería de ventilación auxiliar respecto a la frente de trabajo?", frozenset({"141"}), "ventilacion"),
    EvalQuery("¿Cuál es la concentración mínima de oxígeno permitida en el interior de una mina subterránea?", frozenset({"144"}), "ventilacion"),
    EvalQuery("¿Todas las minas subterráneas deben tener circuitos de ventilación?", frozenset({"137"}), "ventilacion"),
    EvalQuery("¿Qué elementos de alta visibilidad debe portar una persona que transita por el rajo?", frozenset({"246"}), "transito_caex"),
    EvalQuery("¿Qué distancia mínima debe tener la pértiga de un vehículo liviano que transita junto a equipos CAEX?", frozenset({"247"}), "transito_caex"),
    EvalQuery("¿Cómo se debe tapar un hoyo cargado con explosivos?", frozenset({"250"}), "tronadura"),
    EvalQuery("¿A qué distancia mínima debe trabajar el equipo mecanizado de los equipos de carguío de explosivos?", frozenset({"251"}), "tronadura"),
    EvalQuery("¿Qué se debe hacer ante una tormenta eléctrica durante el carguío de explosivos?", frozenset({"252"}), "tronadura"),
    EvalQuery("¿Con qué tipo de luz se puede realizar la tronadura?", frozenset({"253"}), "tronadura"),
    EvalQuery("¿Cuál es la multa por infracciones al Reglamento de Seguridad Minera?", frozenset({"590"}), "sanciones"),
    EvalQuery("¿Quién impone las multas por infracciones al Reglamento de Seguridad Minera?", frozenset({"591"}), "sanciones"),
    EvalQuery("¿Qué pasa en caso de reincidencia en infracciones graves al reglamento?", frozenset({"592"}), "sanciones"),
]


def _article_numbers(documents: list[Document]) -> list[str]:
    return [d.metadata["article_number"] for d in documents]


def reciprocal_rank(ranked_articles: list[str], relevant: frozenset[str]) -> float:
    for i, article in enumerate(ranked_articles, start=1):
        if article in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_articles: list[str], relevant: frozenset[str], k: int) -> float:
    top_k = ranked_articles[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, art in enumerate(top_k, start=1) if art in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def context_relevance_at_k(ranked_articles: list[str], relevant: frozenset[str], k: int) -> float:
    top_k = ranked_articles[:k]
    if not top_k:
        return 0.0
    return sum(1 for art in top_k if art in relevant) / len(top_k)


def citation_faithfulness(answer: str, retrieved_articles: list[str]) -> float | None:
    """None si la respuesta no cita ningún artículo explícitamente (no hay
    nada que evaluar); en caso contrario, la fracción de citas que
    corresponden a artículos efectivamente presentes en el contexto
    recuperado."""
    cited = CITATION_RE.findall(answer)
    if not cited:
        return None
    grounded = sum(1 for art in cited if art in retrieved_articles)
    return grounded / len(cited)


def evaluate_retrieval(
    retrieve_fn: Callable[[str], list[Document]],
    dataset: list[EvalQuery] = QUERY_DATASET,
    k: int = 4,
) -> pd.DataFrame:
    """Evalúa una función de recuperación (p.ej. `pipeline.retriever.invoke`
    para la primera etapa sola, o `pipeline.retrieve` para el pipeline
    completo con re-ranking) sobre el dataset etiquetado."""
    rows = []
    for item in dataset:
        docs = retrieve_fn(item.question)
        ranked_articles = _article_numbers(docs)
        rows.append({
            "question": item.question,
            "topic": item.topic,
            "reciprocal_rank": reciprocal_rank(ranked_articles, item.relevant_articles),
            "ndcg_at_k": ndcg_at_k(ranked_articles, item.relevant_articles, k),
            "context_relevance_at_k": context_relevance_at_k(ranked_articles, item.relevant_articles, k),
            "retrieved_articles": ranked_articles,
        })
    return pd.DataFrame(rows)


def evaluate_faithfulness(pipeline, dataset: list[EvalQuery] = QUERY_DATASET) -> pd.DataFrame:
    """Corre el pipeline completo (recuperación + síntesis) sobre el dataset
    y mide citation faithfulness de la respuesta generada."""
    rows = []
    for item in dataset:
        result = pipeline.query(item.question)
        retrieved_articles = [s["article_number"] for s in result["sources"]]
        rows.append({
            "question": item.question,
            "topic": item.topic,
            "backend": result["backend"],
            "citation_faithfulness": citation_faithfulness(result["answer"], retrieved_articles),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    summary = {}
    if "reciprocal_rank" in df:
        summary["MRR"] = float(df["reciprocal_rank"].mean())
    if "ndcg_at_k" in df:
        summary["NDCG@k"] = float(df["ndcg_at_k"].mean())
    if "context_relevance_at_k" in df:
        summary["Context Relevance@k"] = float(df["context_relevance_at_k"].mean())
    if "citation_faithfulness" in df:
        valid = df["citation_faithfulness"].dropna()
        summary["Citation Faithfulness"] = float(valid.mean()) if len(valid) else None
        summary["n_no_citation"] = int(df["citation_faithfulness"].isna().sum())
    return summary
