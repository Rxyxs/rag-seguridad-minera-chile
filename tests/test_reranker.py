from langchain_core.documents import Document

from src.rag.reranker import CrossEncoderReranker


def test_reranker_returns_empty_list_for_no_candidates():
    reranker = CrossEncoderReranker()
    assert reranker.rerank("cualquier pregunta", []) == []


def test_reranker_orders_the_more_relevant_passage_first():
    """A hand-picked, unambiguous case: one passage directly answers the
    question, the other is on a completely unrelated topic. The
    hybrid retriever's first-stage order is deliberately reversed here (the
    irrelevant one first) so this test actually exercises the re-ranking,
    not just "whatever order it was already in"."""
    reranker = CrossEncoderReranker()
    query = "¿Cuál es la concentración mínima de oxígeno permitida en una mina subterránea?"
    irrelevant = Document(page_content="Artículo 590\nLas contravenciones al reglamento se sancionan con multas de 20 a 50 UTM.", metadata={"article_number": "590", "section": "Sanciones"})
    relevant = Document(page_content="Artículo 144\nNo se permitirá trabajar con concentración de oxígeno inferior a 19,5%.", metadata={"article_number": "144", "section": "Ventilación"})

    ranked = reranker.rerank(query, [irrelevant, relevant])
    assert ranked[0].metadata["article_number"] == "144"


def test_reranker_top_k_truncates_after_reordering():
    reranker = CrossEncoderReranker()
    query = "¿Qué distancia mínima debe tener la pértiga?"
    docs = [
        Document(page_content="Artículo 590\nMultas por incumplimiento.", metadata={"article_number": "590", "section": "Sanciones"}),
        Document(page_content="Artículo 247\nLa pértiga tendrá una altura mínima de tres metros.", metadata={"article_number": "247", "section": "Tránsito"}),
        Document(page_content="Artículo 139\nAforo de ventilación trimestral.", metadata={"article_number": "139", "section": "Ventilación"}),
    ]

    ranked = reranker.rerank(query, docs, top_k=1)
    assert len(ranked) == 1
    assert ranked[0].metadata["article_number"] == "247"


def test_rerank_with_scores_returns_descending_scores():
    reranker = CrossEncoderReranker()
    query = "¿Qué distancia mínima debe tener la pértiga?"
    docs = [
        Document(page_content="Artículo 590\nMultas por incumplimiento.", metadata={"article_number": "590", "section": "Sanciones"}),
        Document(page_content="Artículo 247\nLa pértiga tendrá una altura mínima de tres metros.", metadata={"article_number": "247", "section": "Tránsito"}),
    ]

    ranked = reranker.rerank_with_scores(query, docs)
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)
