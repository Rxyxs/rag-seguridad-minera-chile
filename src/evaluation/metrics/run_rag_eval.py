"""Ejecutable de evaluacion RAG en modo local: sin API keys, sin red, sin descargar modelos.

    python -m src.evaluation.metrics.run_rag_eval            # modo mock (por defecto), escribe tmp_agent_b/eval_results.json
    python -m src.evaluation.metrics.run_rag_eval --real     # evalua el RAGPipeline real (requiere sus dependencias)

Modo mock: recupera con un TF-IDF coseno propio sobre los 27 articulos del extracto del DS 132 y responde en
el formato del backend extractivo del pipeline. Eso valida el codigo de las metricas y fija una referencia
reproducible, pero NO mide la calidad del retriever hibrido ni de un LLM. Como una respuesta extractiva copia el
contexto, su fidelidad es alta por construccion; por eso el informe incluye controles negativos (respuesta de
otra pregunta, cifras alteradas, contexto al azar) que demuestran que las metricas distinguen lo malo de lo bueno.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.evaluation.metrics import rag_evaluator as M

ROOT = Path(__file__).resolve().parents[3]
REGULATION_PATH = ROOT / "data" / "ds132_sernageomin.txt"
EVALUATION_SOURCE = ROOT / "src" / "rag" / "evaluation.py"
OUT_PATH = ROOT / "tmp_agent_b" / "eval_results.json"

K = 4
SEED = 7
ARTICLE_RE = re.compile(r"^Art[ií]culo (\d+)$")
DELIMITER_RE = re.compile(r"^=+$")
PREVIEW_CHARS = 350  # el backend extractivo muestra solo el comienzo de cada articulo
_FIGURE_RE = re.compile(r"(?<!Artículo )(?<!artículo )\b\d+(?:[.,]\d+)?\b")


@dataclass(frozen=True)
class Chunk:
    article: str
    section: str
    text: str  # "Artículo N\n<cuerpo>", igual que el chunking del pipeline

    @property
    def context(self) -> str:
        return f"{self.section}\n{self.text}"


@dataclass(frozen=True)
class Query:
    question: str
    relevant: frozenset
    topic: str


@dataclass(frozen=True)
class Output:
    answer: str
    contexts: list[str]
    articles: list[str]


# ------------------------------------------------------------------ datos
def load_chunks(path: Path = REGULATION_PATH) -> list[Chunk]:
    """Un chunk por articulo con su seccion, con la misma regla de ``src.rag.pipeline.chunk_regulation``
    (reimplementada aqui para no importar langchain/chroma, que el modo mock no necesita)."""
    chunks, section, header, in_header = [], "Disposiciones generales", [], False
    number, body = None, []

    def flush() -> None:
        nonlocal number, body
        text = "\n".join(body).strip()
        if number is not None and text:
            chunks.append(Chunk(number, section, f"Artículo {number}\n{text}"))
        number, body = None, []

    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.strip()
        if DELIMITER_RE.match(stripped):
            if not in_header:
                flush()
                header, in_header = [], True
            else:
                section, in_header = " — ".join(h.strip() for h in header if h.strip()), False
            continue
        if in_header:
            header.append(line)
            continue
        match = ARTICLE_RE.match(stripped)
        if match:
            flush()
            number = match.group(1)
        elif number is not None:
            body.append(line)
    flush()
    return chunks


def _literal(node: ast.AST):
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "frozenset":
        return frozenset(ast.literal_eval(node.args[0]))
    return ast.literal_eval(node)


def load_queries(source: Path = EVALUATION_SOURCE) -> list[Query]:
    """Las 27 consultas etiquetadas de ``src/rag/evaluation.py``, leidas del codigo fuente con ``ast`` para
    tener una unica fuente de verdad sin importar ese modulo (que arrastra langchain)."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = node.target if isinstance(node, ast.AnnAssign) else (node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == "QUERY_DATASET":
            return [Query(*[_literal(a) for a in call.args]) for call in node.value.elts]
    raise ValueError(f"no se encontro QUERY_DATASET en {source}")


# ------------------------------------------------------------------ recuperador y respuesta mock
class LexicalRetriever:
    """TF-IDF (idf suavizado) con coseno sobre las raices de ``rag_evaluator.tokenize``. Determinista."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        docs = [M.tokenize(c.context) for c in chunks]
        self.vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
        df = np.zeros(len(self.vocab))
        for d in docs:
            for t in set(d):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(chunks)) / (1 + df)) + 1.0
        self.matrix = self._normalize(np.array([self._tf(d) for d in docs]) * self.idf)

    def _tf(self, tokens: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        return np.where(v > 0, 1.0 + np.log(np.maximum(v, 1.0)), 0.0)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)

    def retrieve(self, question: str, k: int = K) -> list[Chunk]:
        q = self._normalize(self._tf(M.tokenize(question)) * self.idf)
        scores = self.matrix @ q
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))  # empates por orden de articulo
        return [self.chunks[i] for i in order[:k]]


def extractive_answer(chunks: list[Chunk]) -> str:
    """Mismo formato que ``_extractive_answer`` del pipeline: cada articulo con el comienzo de su texto."""
    lines = ["[Modo extractivo -- sin LLM configurado] Artículos del DS 132 más relevantes para tu consulta:\n"]
    for c in chunks:
        body = c.text.split("\n", 1)[-1].strip()
        lines.append(f"— Artículo {c.article} ({c.section}):\n{body[:PREVIEW_CHARS]}{'...' if len(body) > PREVIEW_CHARS else ''}\n")
    return "\n".join(lines)


def mock_system(chunks: list[Chunk]) -> Callable[[str], Output]:
    retriever = LexicalRetriever(chunks)

    def run(question: str) -> Output:
        got = retriever.retrieve(question)
        return Output(extractive_answer(got), [c.context for c in got], [c.article for c in got])

    return run


def adapt_rag_pipeline(pipeline) -> Callable[[str], Output]:
    """Adapta un ``RAGPipeline`` real (``retrieve`` y ``query``) a la interfaz de este evaluador."""

    def run(question: str) -> Output:
        docs = pipeline.retrieve(question)
        answer = pipeline.query(question)["answer"]
        return Output(answer, [f"{d.metadata.get('section', '')}\n{d.page_content}" for d in docs],
                      [d.metadata["article_number"] for d in docs])

    return run


# ------------------------------------------------------------------ evaluacion
def alter_figures(answer: str) -> str:
    """Cambia toda cifra que no sea un numero de articulo (n -> 3n + 1): simula un plazo o una multa inventados.
    El indicador ``figures_changed`` de la evaluacion solo cuenta cifras dentro de las afirmaciones."""
    return _FIGURE_RE.sub(lambda m: str(3 * int(float(m.group().replace(",", "."))) + 1), answer)


def _row(question: str, answer: str, contexts: list[str], flags: list[bool] | None) -> dict:
    row = {"faithfulness": M.faithfulness(answer, contexts), "answer_relevance": M.answer_relevance(question, answer)}
    if flags is not None:
        row["context_precision_labeled"] = M.context_precision(contexts, relevant_flags=flags)
    if contexts:
        row["context_precision_unlabeled"] = M.context_precision(contexts, reference=question)
    return row


def evaluate(system: Callable[[str], Output], queries: list[Query], chunks: list[Chunk], k: int = K, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    outs = [system(q.question) for q in queries]
    gold = [[a in q.relevant for a in o.articles] for q, o in zip(queries, outs)]
    shift = len(queries) // 2  # la respuesta de otra pregunta, a media tabla de distancia

    conditions: dict[str, list[dict]] = {n: [] for n in ("system", "answer_vs_random_context", "mismatched_answer", "fabricated_figures")}
    rows = []
    for i, (q, o) in enumerate(zip(queries, outs)):
        rand = [chunks[j] for j in rng.choice(len(chunks), size=k, replace=False)]
        rand_ctx = [c.context for c in rand]
        rand_flags = [c.article in q.relevant for c in rand]
        altered = alter_figures(o.answer)
        results = {
            "system": _row(q.question, o.answer, o.contexts, gold[i]),
            "answer_vs_random_context": _row(q.question, o.answer, rand_ctx, rand_flags),
            "mismatched_answer": _row(q.question, outs[(i + shift) % len(outs)].answer, o.contexts, gold[i]),
            "fabricated_figures": {**_row(q.question, altered, o.contexts, gold[i]), "figures_changed": bool(M.numbers_in(" ".join(M.split_claims(o.answer))))},
        }
        for name, r in results.items():
            conditions[name].append(r)
        ranks = [j for j, a in enumerate(o.articles, 1) if a in q.relevant]
        rows.append({"question": q.question, "topic": q.topic, "retrieved_articles": o.articles,
                     "first_relevant_rank": ranks[0] if ranks else None, **{f"system_{m}": v for m, v in results["system"].items()}})

    metric_names = ("faithfulness", "answer_relevance", "context_precision_labeled", "context_precision_unlabeled")
    summary = {name: {m: M.mean_and_std([r[m] for r in rs if m in r]) for m in metric_names if any(m in r for r in rs)}
               for name, rs in conditions.items()}
    changed = [r["faithfulness"] for r in conditions["fabricated_figures"] if r["figures_changed"]]
    unchanged_base = [b["faithfulness"] for b, a in zip(conditions["system"], conditions["fabricated_figures"]) if a["figures_changed"]]
    summary["fabricated_figures"]["on_answers_where_figures_changed"] = {
        "n": len(changed), "faithfulness_after": M.mean_and_std(changed), "faithfulness_before": M.mean_and_std(unchanged_base)}

    all_values = [v for rs in conditions.values() for r in rs for key, v in r.items() if key != "figures_changed"]
    first = [r["first_relevant_rank"] for r in rows]
    sys_ = summary["system"]
    return {
        "retrieval": {"k": k, "n_questions": len(queries), "mrr": float(np.mean([1 / f if f else 0.0 for f in first])),
                      "hit_at_k": float(np.mean([f is not None for f in first]))},
        "conditions": summary,
        "separation": {
            "faithfulness_system_minus_random_context": sys_["faithfulness"]["mean"] - summary["answer_vs_random_context"]["faithfulness"]["mean"],
            "faithfulness_system_minus_fabricated_figures": sys_["faithfulness"]["mean"] - summary["fabricated_figures"]["faithfulness"]["mean"],
            "answer_relevance_system_minus_mismatched": sys_["answer_relevance"]["mean"] - summary["mismatched_answer"]["answer_relevance"]["mean"],
            "context_precision_labeled_system_minus_random": sys_["context_precision_labeled"]["mean"] - summary["answer_vs_random_context"]["context_precision_labeled"]["mean"],
            "share_questions_system_faithfulness_above_random_context": float(np.mean(
                [a["faithfulness"] > b["faithfulness"] for a, b in zip(conditions["system"], conditions["answer_vs_random_context"])])),
            "share_questions_system_relevance_above_mismatched": float(np.mean(
                [a["answer_relevance"] > b["answer_relevance"] for a, b in zip(conditions["system"], conditions["mismatched_answer"])])),
        },
        "bounds_check": {"n_values": len(all_values), "min": float(min(all_values)), "max": float(max(all_values)),
                         "all_within_0_1": bool(all(0.0 <= v <= 1.0 for v in all_values))},
        "per_question": rows,
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real", action="store_true", help="evalua el RAGPipeline real en vez del mock (requiere langchain, chroma y modelos)")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    chunks, queries = load_chunks(), load_queries()
    if args.real:
        from src.rag.pipeline import RAGPipeline  # solo aqui: arrastra langchain/chroma y modelos de HuggingFace

        pipeline = RAGPipeline()
        system, mode, backend = adapt_rag_pipeline(pipeline), "real", pipeline.backend_name
    else:
        system, mode, backend = mock_system(chunks), "mock", "extractive (sin LLM)"
    results = {
        "mode": mode, "backend": backend, "seed": SEED, "n_articles": len(chunks),
        "metric_config": {"embedder": "HashingEmbedder(dim=8192), raices de 5 letras, sin stopwords", "support_threshold": 0.7,
                          "unlabeled_relevance_threshold": 0.2, "claim_min_content_tokens": 3},
        "notes": ["Modo mock: valida las metricas y fija una referencia; no mide la calidad del retriever hibrido ni de un LLM.",
                  "La fidelidad del backend extractivo es alta por construccion (copia el contexto): lo informativo son los controles negativos.",
                  "La fidelidad se mide por afirmacion: alterar las pocas cifras de una respuesta larga la baja poco en promedio (es la fraccion de afirmaciones afectadas).",
                  "Metricas lexicas: no detectan una parafrasis que invierta el sentido."],
        **evaluate(system, queries, chunks),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    c = results["conditions"]
    print(f"modo {mode}: {len(queries)} consultas, k={K}; MRR {results['retrieval']['mrr']:.3f}")
    for name, m in c.items():
        print(f"  {name:26s} " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in m.items() if isinstance(v, dict) and "mean" in v))
    print(f"valores acotados en [0,1]: {results['bounds_check']['all_within_0_1']} -> {args.out}")
    return results


if __name__ == "__main__":
    main()
