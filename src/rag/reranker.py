"""Segunda etapa de re-ranking con Cross-Encoder, sobre los candidatos que
devuelve el retriever híbrido (primera etapa, ver `pipeline.py`).

Por qué dos etapas: el retriever híbrido (denso + BM25) está optimizado para
**recall** -- traer un conjunto amplio de candidatos razonablemente
relevantes, rápido, sobre miles de documentos. Un Cross-Encoder es mucho más
preciso para **ranking** (evalúa la pregunta y cada documento *juntos* en una
sola pasada por el modelo, en vez de comparar embeddings pre-calculados por
separado), pero es demasiado costoso para correr sobre toda la colección --
por eso se aplica solo sobre los ~10 candidatos que ya trajo la primera etapa,
no sobre los artículos completos del reglamento.

Nota sobre el nombre del modelo: el identificador
`sentence-transformers/ms-marco-MiniLM-L-6-v2` (una forma común de referirse
a este modelo) no existe en el Hub de HuggingFace -- devuelve
`RepositoryNotFoundError` (verificado directamente antes de escribir este
módulo). El modelo real, publicado por el mismo equipo que entrena los
cross-encoders de sentence-transformers pero bajo la organización
`cross-encoder`, es `cross-encoder/ms-marco-MiniLM-L-6-v2`, y es el que se usa
acá.
"""
from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-ordena una lista de documentos candidatos por relevancia real a la
    pregunta, usando un Cross-Encoder entrenado sobre MS MARCO (pares
    pregunta-pasaje con relevancia etiquetada)."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        # Carga perezosa: instanciar `CrossEncoderReranker` no debería
        # descargar ~90MB de pesos si el llamador nunca termina usándolo
        # (p.ej. `RAGPipeline(use_reranker=False)` en un test).
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[Document], top_k: int | None = None) -> list[Document]:
        """Re-ordena `documents` de mayor a menor relevancia real a `query`.
        Si `top_k` se especifica, trunca a los primeros `top_k` tras
        re-ordenar (no antes -- el punto de re-rankear es justamente decidir
        cuáles de los candidatos originales sobreviven al corte final)."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = [doc for _, doc in sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)]
        return ranked[:top_k] if top_k is not None else ranked

    def rerank_with_scores(self, query: str, documents: list[Document], top_k: int | None = None) -> list[tuple[Document, float]]:
        """Igual que `rerank`, pero devuelve también el score del
        Cross-Encoder por documento -- útil para depuración y para el
        notebook de evaluación."""
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k] if top_k is not None else ranked
