"""Pipeline RAG modular para el DS 132 (Reglamento de Seguridad Minera).

Diseño:
1. Chunking por artículo: el reglamento ya viene estructurado con
   encabezados "Artículo N" y secciones delimitadas por "====" -- en vez
   de trocear por caracteres (que puede cortar un artículo a la mitad),
   se explota esa estructura para que cada chunk sea exactamente un
   artículo completo, con su sección como metadata. Esto es clave para
   un dominio legal/normativo: la cita debe ser precisa.
2. Retriever híbrido: `langchain.retrievers.EnsembleRetriever` combina un
   retriever denso (embeddings HuggingFace + ChromaDB) con uno disperso
   (BM25 por palabras clave) -- captura tanto similitud semántica como
   coincidencias léxicas exactas (números de artículo, términos técnicos
   como "acuñadura" o "pértiga" que un embedding puede diluir).
3. Backend de síntesis intercambiable: Ollama (local) -> OpenAI (API) ->
   modo extractivo (sin LLM, siempre disponible) como fallback. El
   fallback extractivo hace que el pipeline sea 100% funcional y testeable
   sin depender de un LLM externo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHROMA_PERSIST_DIR = ROOT_DIR / "data" / "chroma_db"

REGULATION_PATH = DATA_DIR / "ds132_sernageomin.txt"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "ds132_articles"
DEFAULT_K = 4

ARTICLE_HEADER_RE = re.compile(r"^Artículo (\d+)$")
SECTION_DELIMITER_RE = re.compile(r"^=+$")

RAG_SYSTEM_PROMPT = """Eres un asistente de prevención de riesgos que ayuda a ingenieros de faena a \
consultar el Reglamento de Seguridad Minera de Chile (DS 132). Responde SOLO en base a los \
artículos entregados como contexto. Si el contexto no es suficiente para responder con certeza, dilo \
explícitamente y recomienda consultar el texto completo y vigente del DS 132 o a un experto en \
prevención de riesgos -- no inventes artículos, cifras ni sanciones que no aparezcan en el contexto.

Contexto (extracto del DS 132, Reglamento de Seguridad Minera):
{context}

Pregunta del usuario:
{question}

Responde en español, citando el número de artículo de cada afirmación que hagas."""


def load_regulation_text(path: Path | None = None) -> str:
    path = path or REGULATION_PATH
    return path.read_text(encoding="utf-8")


def chunk_regulation(text: str) -> list[dict]:
    """Divide el texto en un chunk por artículo, con su sección como contexto."""
    lines = text.split("\n")
    chunks: list[dict] = []

    current_section = "Disposiciones generales"
    current_article_num: str | None = None
    current_article_lines: list[str] = []
    in_header_block = False
    header_lines: list[str] = []

    def flush_article() -> None:
        nonlocal current_article_num, current_article_lines
        if current_article_num is not None:
            body = "\n".join(current_article_lines).strip()
            if body:
                chunks.append(
                    {
                        "article_number": current_article_num,
                        "section": current_section,
                        "text": f"Artículo {current_article_num}\n{body}",
                    }
                )
        current_article_num = None
        current_article_lines = []

    for line in lines:
        stripped = line.strip()
        if SECTION_DELIMITER_RE.match(stripped):
            if not in_header_block:
                flush_article()
                header_lines = []
                in_header_block = True
            else:
                current_section = " — ".join(h.strip() for h in header_lines if h.strip())
                in_header_block = False
            continue
        if in_header_block:
            header_lines.append(line)
            continue

        match = ARTICLE_HEADER_RE.match(stripped)
        if match:
            flush_article()
            current_article_num = match.group(1)
            continue

        if current_article_num is not None:
            current_article_lines.append(line)

    flush_article()
    return chunks


def build_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["text"],
            metadata={"article_number": c["article_number"], "section": c["section"]},
        )
        for c in chunks
    ]


def build_dense_retriever(documents: list[Document], persist_dir: Path, k: int = DEFAULT_K):
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruccion idempotente: se borra la coleccion via el cliente de Chroma
    # (no el directorio con shutil.rmtree) porque en Windows un cliente previo
    # dentro del mismo proceso (p.ej. otro RAGPipeline en la misma sesion de
    # tests) puede dejar los archivos del indice bloqueados, y rmtree falla
    # con PermissionError. Borrar la coleccion por nombre evita el lock de OS.
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=client,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def build_sparse_retriever(documents: list[Document], k: int = DEFAULT_K) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k
    return retriever


def build_hybrid_retriever(
    documents: list[Document],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    k: int = DEFAULT_K,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> EnsembleRetriever:
    dense = build_dense_retriever(documents, persist_dir, k=k)
    sparse = build_sparse_retriever(documents, k=k)
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[dense_weight, sparse_weight])


# --- Backends de sintesis (Ollama -> OpenAI -> extractivo) -----------------


def _ollama_available(model: str) -> bool:
    try:
        import ollama

        available = {m["model"] for m in ollama.list().get("models", [])}
        return any(model in name for name in available) or bool(available)
    except Exception:
        return False


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[Artículo {d.metadata['article_number']}] {d.page_content}" for d in docs)


def _extractive_answer(question: str, docs: list[Document]) -> str:
    lines = [
        "[Modo extractivo -- sin LLM configurado] Artículos del DS 132 más relevantes "
        f"para tu consulta:\n"
    ]
    for doc in docs:
        art = doc.metadata.get("article_number")
        section = doc.metadata.get("section")
        body = doc.page_content.split("\n", 1)[-1].strip()
        preview = body[:350] + ("..." if len(body) > 350 else "")
        lines.append(f"— Artículo {art} ({section}):\n{preview}\n")
    return "\n".join(lines)


def _ollama_answer(question: str, docs: list[Document], model: str = "llama3.1") -> str:
    import ollama

    prompt = RAG_SYSTEM_PROMPT.format(context=_format_context(docs), question=question)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def _openai_answer(question: str, docs: list[Document], model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI()
    prompt = RAG_SYSTEM_PROMPT.format(context=_format_context(docs), question=question)
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def get_llm_backend(preference: str = "auto", ollama_model: str = "llama3.1"):
    """Devuelve (nombre_backend, funcion_respuesta(question, docs) -> str)."""
    if preference in ("auto", "ollama") and _ollama_available(ollama_model):
        return "ollama", lambda q, d: _ollama_answer(q, d, model=ollama_model)
    if preference in ("auto", "openai") and os.environ.get("OPENAI_API_KEY"):
        return "openai", _openai_answer
    return "extractive", _extractive_answer


class RAGPipeline:
    """Pipeline RAG completo: ingesta -> retriever híbrido -> síntesis."""

    def __init__(
        self,
        regulation_path: Path | None = None,
        persist_dir: Path = CHROMA_PERSIST_DIR,
        llm_preference: str = "auto",
        k: int = DEFAULT_K,
    ) -> None:
        text = load_regulation_text(regulation_path)
        chunks = chunk_regulation(text)
        if not chunks:
            raise ValueError("No se extrajo ningún artículo del texto normativo. Revisa el formato del archivo.")

        self.documents = build_documents(chunks)
        self.retriever = build_hybrid_retriever(self.documents, persist_dir=persist_dir, k=k)
        self.backend_name, self._answer_fn = get_llm_backend(llm_preference)

    def retrieve(self, question: str) -> list[Document]:
        return self.retriever.invoke(question)

    def query(self, question: str) -> dict:
        docs = self.retrieve(question)
        answer = self._answer_fn(question, docs)
        return {
            "question": question,
            "answer": answer,
            "backend": self.backend_name,
            "sources": [
                {"article_number": d.metadata["article_number"], "section": d.metadata["section"]}
                for d in docs
            ],
        }


def main() -> None:
    print("Construyendo pipeline RAG (chunking + retriever híbrido)...")
    pipeline = RAGPipeline()
    print(f"Artículos indexados: {len(pipeline.documents)}")
    print(f"Backend de síntesis activo: {pipeline.backend_name}")

    demo_question = "¿Qué debo hacer si un trabajador queda atrapado en el chancador?"
    result = pipeline.query(demo_question)

    print(f"\nPregunta: {result['question']}")
    print(f"Fuentes recuperadas: {result['sources']}")
    print(f"\nRespuesta:\n{result['answer']}")


if __name__ == "__main__":
    main()
