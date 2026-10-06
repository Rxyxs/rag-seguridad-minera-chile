"""Metricas de evaluacion RAG sin LLM-juez ni API keys: Faithfulness, Answer Relevance y Context Precision.

Las definiciones "estandar" (p.ej. RAGAS) piden un LLM que juzgue cada afirmacion o genere preguntas a
partir de la respuesta. Aqui las tres se definen de forma determinista y reproducible, para que puedan correr
en CI y sobre el backend extractivo sin servicios externos. Todas devuelven un float en [0, 1].

- **Faithfulness**: fraccion de las afirmaciones de la respuesta (una por oracion) que el contexto recuperado
  soporta. Una afirmacion queda soportada si al menos ``support_threshold`` de sus palabras de contenido
  aparecen en el contexto Y todas las cifras que menciona aparecen tambien en el contexto: inventar un plazo o
  una multa es la alucinacion que mas importa en un reglamento. Respuesta vacia: 0.0.
- **Answer Relevance**: similitud coseno entre la pregunta y la respuesta en un espacio de embeddings. Por
  defecto un embedder lexico determinista (``HashingEmbedder``); se puede pasar cualquier funcion
  ``textos -> matriz`` (p.ej. sentence-transformers) y la similitud negativa se trunca a 0.
- **Context Precision**: precision promedio ponderada por rango (como RAGAS) de los fragmentos recuperados:
  ``sum_k(precision@k * relevante_k) / n_relevantes``. Premia traer los fragmentos relevantes primero.
  La relevancia viene de etiquetas (``relevant_flags``) o, sin etiquetas, de la similitud con una referencia.

Limites: son metricas lexicas. Detectan respuestas que se apartan del contexto o de la pregunta y cifras
inventadas, pero no una parafrasis que invierta el sentido ("debe" por "no debe"). Para eso hace falta un juez
semantico (LLM o NLI); estas metricas son el piso verificable que corre sin red.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Callable, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]

STOPWORDS = frozenset("""
a al algo ante aquel aquella aquellos as asi aun bajo cada como con contra cual cuales cuando de del desde donde
durante e el ella ellas ello ellos en entre era es esa esas ese eso esos esta estan estar este esto estos ha han
hasta hay la las le les lo los mas me mi mis muy ni no nos nuestro o os otra otro para pero poco por porque que
quien se segun ser si sin sobre su sus tal tambien tanto te tiene tienen todo todos tu tus un una uno unos y ya
""".split())

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+|\s[—–]\s")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_STEM_LEN = 5


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _normalize_number(token: str) -> str:
    return token.replace(",", ".").rstrip(".")


def tokenize(text: str) -> list[str]:
    """Palabras de contenido en minuscula, sin acentos ni stopwords, truncadas a una raiz de 5 letras.

    Las cifras se conservan completas (``250``, ``1.5``): comparar raices con numeros no tiene sentido.
    """
    out = []
    for tok in _TOKEN_RE.findall(_strip_accents(text.lower())):
        if tok[0].isdigit():
            out.append(_normalize_number(tok))
        elif tok not in STOPWORDS and len(tok) > 1:
            out.append(tok[:_STEM_LEN])
    return out


def numbers_in(text: str) -> set[str]:
    return {_normalize_number(t) for t in _TOKEN_RE.findall(_strip_accents(text.lower())) if t[0].isdigit()}


class HashingEmbedder:
    """Bolsa de raices con truco de hashing, frecuencia sublineal y norma L2. Determinista (crc32, no ``hash``).

    Los vectores son no negativos, asi que el coseno entre dos textos queda en [0, 1].
    """

    def __init__(self, dim: int = 8192) -> None:
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                out[i, zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0
        out = np.where(out > 0, 1.0 + np.log(np.maximum(out, 1.0)), 0.0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


DEFAULT_EMBEDDER: Embedder = HashingEmbedder()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno de dos vectores, truncado a [0, 1]; 0.0 si alguno es nulo."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(min(1.0, max(0.0, float(np.dot(a, b)) / (na * nb))))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ------------------------------------------------------------------ faithfulness
def split_claims(answer: str, min_content_tokens: int = 3) -> list[str]:
    """Una afirmacion por oracion. Se descartan las marcas entre corchetes (p.ej. el aviso del modo extractivo),
    las oraciones que terminan en ``:`` (introducen contenido, no afirman nada) y los fragmentos con menos de
    ``min_content_tokens`` palabras de contenido (encabezados, citas sueltas)."""
    claims = []
    for piece in _SENTENCE_SPLIT_RE.split(_BRACKET_RE.sub(" ", answer)):
        piece = piece.strip()
        if not piece.endswith(":") and len(tokenize(piece)) >= min_content_tokens:
            claims.append(piece)
    return claims


def claim_support(claim: str, context: str) -> float:
    """Fraccion de las palabras de contenido de la afirmacion presentes en el contexto, o 0.0 si la afirmacion
    menciona alguna cifra que el contexto no contiene."""
    tokens = tokenize(claim)
    if not tokens:
        return 0.0
    if not numbers_in(claim) <= numbers_in(context):
        return 0.0
    ctx = set(tokenize(context))
    return sum(1 for t in tokens if t in ctx) / len(tokens)


def faithfulness_detail(answer: str, contexts: Sequence[str], support_threshold: float = 0.7) -> list[dict]:
    context = "\n".join(contexts)
    return [{"claim": c, "support": round(s, 4), "supported": s >= support_threshold}
            for c in split_claims(answer) for s in [claim_support(c, context)]]


def faithfulness(answer: str, contexts: Sequence[str], support_threshold: float = 0.7) -> float:
    """Fraccion de afirmaciones de ``answer`` soportadas por ``contexts``. 0.0 si no hay afirmaciones que evaluar."""
    detail = faithfulness_detail(answer, contexts, support_threshold)
    if not detail:
        return 0.0
    return _clip01(sum(d["supported"] for d in detail) / len(detail))


# ------------------------------------------------------------------ answer relevance
def answer_relevance(question: str, answer: str, embed: Embedder = DEFAULT_EMBEDDER) -> float:
    """Similitud coseno pregunta-respuesta, en [0, 1]."""
    if not question.strip() or not answer.strip():
        return 0.0
    vecs = np.asarray(embed([question, answer]), dtype=float)
    return cosine(vecs[0], vecs[1])


# ------------------------------------------------------------------ context precision
def context_precision_from_flags(relevant_flags: Sequence[bool]) -> float:
    """Precision promedio ponderada por rango: ``sum_k(precision@k * v_k) / sum_k(v_k)``; 0.0 si no hay relevantes."""
    hits, total = 0, 0.0
    for k, flag in enumerate(relevant_flags, start=1):
        if flag:
            hits += 1
            total += hits / k
    return _clip01(total / hits) if hits else 0.0


def relevance_flags_by_similarity(
    reference: str, contexts: Sequence[str], threshold: float = 0.2, embed: Embedder = DEFAULT_EMBEDDER
) -> list[bool]:
    """Sin etiquetas: un fragmento es relevante si su coseno con la referencia (pregunta o respuesta de
    referencia) alcanza ``threshold``."""
    if not contexts:
        return []
    vecs = np.asarray(embed([reference, *contexts]), dtype=float)
    return [cosine(vecs[0], v) >= threshold for v in vecs[1:]]


def context_precision(
    contexts: Sequence[str],
    relevant_flags: Sequence[bool] | None = None,
    reference: str | None = None,
    threshold: float = 0.2,
    embed: Embedder = DEFAULT_EMBEDDER,
) -> float:
    """Context Precision de la lista ordenada ``contexts``.

    Con ``relevant_flags`` (una por fragmento, p.ej. del etiquetado manual) usa esas etiquetas; si no, las infiere
    por similitud con ``reference``. Lista vacia: 0.0.
    """
    if relevant_flags is None:
        if reference is None:
            raise ValueError("indica relevant_flags o reference")
        relevant_flags = relevance_flags_by_similarity(reference, contexts, threshold, embed)
    if len(relevant_flags) != len(contexts):
        raise ValueError(f"{len(relevant_flags)} etiquetas para {len(contexts)} fragmentos")
    return context_precision_from_flags(relevant_flags)


def mean_and_std(values: Sequence[float]) -> dict:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


__all__ = [
    "HashingEmbedder", "DEFAULT_EMBEDDER", "tokenize", "numbers_in", "cosine", "split_claims", "claim_support",
    "faithfulness", "faithfulness_detail", "answer_relevance", "context_precision", "context_precision_from_flags",
    "relevance_flags_by_similarity", "mean_and_std",
]
