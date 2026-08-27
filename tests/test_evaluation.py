import pytest

from src.rag.evaluation import (
    QUERY_DATASET,
    citation_faithfulness,
    context_relevance_at_k,
    ndcg_at_k,
    reciprocal_rank,
    summarize,
)
from src.rag.pipeline import chunk_regulation, load_regulation_text
import pandas as pd


def test_reciprocal_rank_found_at_second_position():
    assert reciprocal_rank(["a", "b", "c"], frozenset({"b"})) == pytest.approx(0.5)


def test_reciprocal_rank_not_found_is_zero():
    assert reciprocal_rank(["x", "y"], frozenset({"z"})) == 0.0


def test_reciprocal_rank_found_first_is_one():
    assert reciprocal_rank(["b", "a"], frozenset({"b"})) == pytest.approx(1.0)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], frozenset({"a"}), k=2) == pytest.approx(1.0)


def test_ndcg_relevant_item_ranked_second():
    # DCG = 1/log2(3) (relevant at position 2), IDCG = 1/log2(2) = 1.0
    import math
    expected = (1 / math.log2(3)) / 1.0
    assert ndcg_at_k(["b", "a"], frozenset({"a"}), k=2) == pytest.approx(expected)


def test_ndcg_no_relevant_within_k_is_zero():
    assert ndcg_at_k(["x", "y", "a"], frozenset({"a"}), k=2) == 0.0


def test_ndcg_no_relevant_articles_at_all_is_zero():
    assert ndcg_at_k(["x", "y"], frozenset(), k=2) == 0.0


def test_context_relevance_at_k():
    ranked = ["a", "b", "c", "d"]
    relevant = frozenset({"a", "c"})
    assert context_relevance_at_k(ranked, relevant, k=4) == pytest.approx(0.5)
    assert context_relevance_at_k(ranked, relevant, k=2) == pytest.approx(0.5)
    assert context_relevance_at_k(ranked, relevant, k=1) == pytest.approx(1.0)


def test_context_relevance_empty_top_k_is_zero():
    assert context_relevance_at_k([], frozenset({"a"}), k=4) == 0.0


def test_citation_faithfulness_partial_grounding():
    answer = "Según el Artículo 144 y el Artículo 999, se debe controlar el oxígeno."
    result = citation_faithfulness(answer, retrieved_articles=["144", "162"])
    assert result == pytest.approx(0.5)


def test_citation_faithfulness_fully_grounded():
    answer = "El Artículo 144 establece la concentración mínima de oxígeno."
    result = citation_faithfulness(answer, retrieved_articles=["144", "162"])
    assert result == pytest.approx(1.0)


def test_citation_faithfulness_no_citations_is_none():
    answer = "No tengo información suficiente para responder con certeza."
    assert citation_faithfulness(answer, retrieved_articles=["144"]) is None


def test_summarize_reports_mean_metrics_and_no_citation_count():
    df = pd.DataFrame({
        "reciprocal_rank": [1.0, 0.5],
        "ndcg_at_k": [1.0, 0.0],
        "context_relevance_at_k": [0.5, 0.5],
        "citation_faithfulness": [1.0, None],
    })
    summary = summarize(df)
    assert summary["MRR"] == pytest.approx(0.75)
    assert summary["NDCG@k"] == pytest.approx(0.5)
    assert summary["Context Relevance@k"] == pytest.approx(0.5)
    assert summary["Citation Faithfulness"] == pytest.approx(1.0)
    assert summary["n_no_citation"] == 1


def test_query_dataset_has_27_queries_one_per_curated_article():
    assert len(QUERY_DATASET) == 27


def test_query_dataset_ground_truth_articles_actually_exist_in_the_regulation():
    """Every article number named as "relevant" in the eval dataset must
    actually be one of the articles the curated DS 132 excerpt contains --
    otherwise the eval set would be scoring against ground truth that no
    retriever could ever satisfy."""
    chunks = chunk_regulation(load_regulation_text())
    real_article_numbers = {c["article_number"] for c in chunks}

    for item in QUERY_DATASET:
        for article in item.relevant_articles:
            assert article in real_article_numbers, f"Articulo {article} (query: {item.question!r}) no existe en el reglamento"


def test_query_dataset_covers_every_article_in_the_regulation():
    """The eval dataset should have at least one query per real article, not
    just a subset -- otherwise "27 queries" would overstate the coverage."""
    chunks = chunk_regulation(load_regulation_text())
    real_article_numbers = {c["article_number"] for c in chunks}

    covered = set()
    for item in QUERY_DATASET:
        covered |= item.relevant_articles

    missing = real_article_numbers - covered
    assert not missing, f"Articulos sin ninguna consulta de evaluacion: {missing}"
