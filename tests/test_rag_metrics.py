"""Pruebas de las metricas RAG (``src/evaluation/metrics``): valores acotados en [0, 1], casos calculados a mano y
comportamiento ante entradas degeneradas. No necesitan red, API keys, langchain ni modelos descargados."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.metrics import rag_evaluator as M
from src.evaluation.metrics import run_rag_eval as R

CONTEXT = [
    "Artículo 138\nEn toda mina subterránea el caudal mínimo de aire fresco será de 3 metros cúbicos por minuto por persona.",
    "Artículo 139\nEl aforo de ventilación se realizará cada 6 meses y quedará registrado en la mina.",
]


# ----------------------------------------------------------------------------- acotamiento en [0, 1]
def _random_text(rng: random.Random, n_words: int) -> str:
    vocab = ["mina", "aire", "fresco", "Artículo", "138", "3", "250.5", "persona", "tronadura", ":", ".", "\n", "[x]", "—", "ñandú", "", " "]
    return " ".join(rng.choice(vocab) for _ in range(n_words))


@pytest.mark.parametrize("seed", range(40))
def test_all_metrics_stay_within_0_1_on_random_inputs(seed):
    rng = random.Random(seed)
    question, answer = _random_text(rng, rng.randint(0, 25)), _random_text(rng, rng.randint(0, 60))
    contexts = [_random_text(rng, rng.randint(0, 40)) for _ in range(rng.randint(0, 5))]
    flags = [rng.random() < 0.5 for _ in contexts]
    values = [
        M.faithfulness(answer, contexts),
        M.answer_relevance(question, answer),
        M.context_precision(contexts, relevant_flags=flags),
        M.context_precision(contexts, reference=question),
    ]
    assert all(0.0 <= v <= 1.0 and np.isfinite(v) for v in values), values


@pytest.mark.parametrize("answer,contexts", [("", []), ("", CONTEXT), ("texto", []), ("   \n ", CONTEXT), ("a", ["b"])])
def test_degenerate_inputs_return_a_bounded_number_instead_of_raising(answer, contexts):
    assert 0.0 <= M.faithfulness(answer, contexts) <= 1.0
    assert 0.0 <= M.answer_relevance(answer, answer) <= 1.0


def test_embedding_cosine_is_bounded_even_if_a_custom_embedder_returns_negative_or_large_values():
    assert M.answer_relevance("q", "a", embed=lambda texts: np.array([[1.0, 0.0], [-1.0, 0.0]])) == 0.0
    assert M.answer_relevance("q", "a", embed=lambda texts: np.array([[3.0, 4.0], [3.0, 4.0]])) == 1.0
    assert M.cosine(np.array([1.0, 1.0]), np.array([1.0, 1.0 + 1e-12])) <= 1.0


# ----------------------------------------------------------------------------- faithfulness
def test_an_answer_copied_from_the_context_is_fully_faithful():
    answer = "El caudal mínimo de aire fresco será de 3 metros cúbicos por minuto por persona. El aforo de ventilación se realizará cada 6 meses."
    assert M.faithfulness(answer, CONTEXT) == 1.0


def test_an_answer_about_something_else_is_unfaithful():
    assert M.faithfulness("Los explosivos deben almacenarse en polvorines autorizados por la autoridad.", CONTEXT) == 0.0


def test_half_supported_answer_scores_one_half():
    answer = "El caudal mínimo de aire fresco será de 3 metros cúbicos por minuto por persona. Los explosivos deben almacenarse en polvorines autorizados."
    assert M.faithfulness(answer, CONTEXT) == 0.5


def test_an_invented_figure_makes_an_otherwise_supported_claim_unfaithful():
    true_claim = "El aforo de ventilación se realizará cada 6 meses en la mina."
    fake_claim = "El aforo de ventilación se realizará cada 9 meses en la mina."
    assert M.faithfulness(true_claim, CONTEXT) == 1.0
    assert M.faithfulness(fake_claim, CONTEXT) == 0.0


def test_faithfulness_ignores_case_and_accents():
    assert M.faithfulness("EL AFORO DE VENTILACION SE REALIZARA CADA 6 MESES EN LA MINA", CONTEXT) == 1.0


def test_decimal_figures_with_comma_or_dot_are_the_same_figure():
    assert M.numbers_in("multa de 1,5 UTM") == M.numbers_in("multa de 1.5 UTM") == {"1.5"}


def test_boilerplate_brackets_and_colon_lead_ins_are_not_claims():
    answer = "[Modo extractivo -- sin LLM configurado] Artículos más relevantes para tu consulta:\n El aforo de ventilación se realizará cada 6 meses en la mina."
    claims = M.split_claims(answer)
    assert len(claims) == 1 and "aforo" in claims[0]
    assert M.faithfulness(answer, CONTEXT) == 1.0


def test_empty_answer_or_empty_context_scores_zero():
    assert M.faithfulness("", CONTEXT) == 0.0
    assert M.faithfulness("El aforo de ventilación se realizará cada 6 meses.", []) == 0.0


def test_faithfulness_detail_reports_each_claim_and_agrees_with_the_score():
    answer = "El aforo de ventilación se realizará cada 6 meses en la mina. Los explosivos deben almacenarse en polvorines autorizados."
    detail = M.faithfulness_detail(answer, CONTEXT)
    assert [d["supported"] for d in detail] == [True, False]
    assert M.faithfulness(answer, CONTEXT) == sum(d["supported"] for d in detail) / len(detail)


def test_a_higher_support_threshold_can_only_lower_the_score():
    answer = "El aforo de ventilación se realizará cada 6 meses y lo supervisa un ingeniero titulado externo."
    assert M.faithfulness(answer, CONTEXT, support_threshold=0.9) <= M.faithfulness(answer, CONTEXT, support_threshold=0.3)


# ----------------------------------------------------------------------------- answer relevance
def test_identical_question_and_answer_are_maximally_relevant():
    assert M.answer_relevance("caudal mínimo de aire fresco por persona", "caudal mínimo de aire fresco por persona") == pytest.approx(1.0)


def test_unrelated_answer_has_zero_relevance_and_related_one_scores_higher():
    q = "¿Cuál es el caudal mínimo de aire fresco por persona en una mina subterránea?"
    related = "El caudal mínimo de aire fresco será de 3 metros cúbicos por minuto por persona."
    unrelated = "Las multas por infracciones las impone el Servicio de Geología y Minería."
    assert M.answer_relevance(q, unrelated) == 0.0
    assert M.answer_relevance(q, related) > M.answer_relevance(q, unrelated)


def test_answer_relevance_is_symmetric_and_empty_inputs_score_zero():
    a, b = "ventilación de la mina subterránea", "aforo de ventilación cada seis meses"
    assert M.answer_relevance(a, b) == pytest.approx(M.answer_relevance(b, a))
    assert M.answer_relevance("", b) == 0.0 and M.answer_relevance(a, "  ") == 0.0


def test_stopwords_alone_do_not_create_relevance():
    assert M.answer_relevance("el de la que", "el de la que") == 0.0


# ----------------------------------------------------------------------------- context precision
@pytest.mark.parametrize("flags,expected", [
    ([True, True, True], 1.0),
    ([False, False], 0.0),
    ([True, False, False], 1.0),
    ([False, True], 0.5),
    ([True, False, True], (1 + 2 / 3) / 2),
    ([False, False, True], 1 / 3),
    ([], 0.0),
])
def test_context_precision_matches_the_hand_computed_average_precision(flags, expected):
    assert M.context_precision_from_flags(flags) == pytest.approx(expected)


def test_relevant_fragments_ranked_first_score_higher_than_ranked_last():
    assert M.context_precision_from_flags([True, True, False, False]) > M.context_precision_from_flags([False, False, True, True])


def test_context_precision_with_labels_uses_them_and_checks_the_length():
    assert M.context_precision(CONTEXT, relevant_flags=[True, False]) == 1.0
    with pytest.raises(ValueError):
        M.context_precision(CONTEXT, relevant_flags=[True])
    with pytest.raises(ValueError):
        M.context_precision(CONTEXT)


def test_context_precision_without_labels_infers_relevance_from_the_reference():
    ref = "caudal mínimo de aire fresco por persona en una mina subterránea"
    assert M.relevance_flags_by_similarity(ref, CONTEXT) == [True, False]
    assert M.context_precision(CONTEXT, reference=ref) == 1.0
    assert M.context_precision(list(reversed(CONTEXT)), reference=ref) == 0.5
    assert M.context_precision([], reference=ref) == 0.0


# ----------------------------------------------------------------------------- embedder y texto
def test_hashing_embedder_is_deterministic_non_negative_and_normalized():
    a, b = M.HashingEmbedder()(["aire fresco en la mina", "tronadura"]), M.HashingEmbedder()(["aire fresco en la mina", "tronadura"])
    assert np.array_equal(a, b) and (a >= 0).all()
    assert np.linalg.norm(a, axis=1) == pytest.approx([1.0, 1.0])
    assert M.HashingEmbedder()([""]).sum() == 0.0


def test_tokenize_strips_accents_stopwords_and_truncates_to_stems_but_keeps_figures():
    assert M.tokenize("El Acuñadura de la mina, 3 metros y 1,5 UTM") == ["acuna", "mina", "3", "metro", "1.5", "utm"]
    assert M.tokenize("fortificado") == M.tokenize("fortificación")


def test_mean_and_std_handle_empty_single_and_regular_inputs():
    assert M.mean_and_std([])["n"] == 0
    assert M.mean_and_std([0.5])["std"] == 0.0
    s = M.mean_and_std([0.0, 1.0])
    assert (s["mean"], s["min"], s["max"], s["n"]) == (0.5, 0.0, 1.0, 2)


# ----------------------------------------------------------------------------- ejecutable (modo mock)
@pytest.fixture(scope="module")
def chunks():
    return R.load_chunks()


@pytest.fixture(scope="module")
def queries():
    return R.load_queries()


@pytest.fixture(scope="module")
def results(chunks, queries):
    return R.evaluate(R.mock_system(chunks), queries, chunks)


def test_the_regulation_extract_is_split_into_27_article_chunks_with_sections(chunks):
    assert len(chunks) == 27 and len({c.article for c in chunks}) == 27
    assert all(c.text.startswith(f"Artículo {c.article}\n") and c.section for c in chunks)


def test_every_labeled_query_points_to_articles_that_exist(chunks, queries):
    assert len(queries) == 27
    articles = {c.article for c in chunks}
    assert all(q.relevant and q.relevant <= articles and q.question.strip() for q in queries)


def test_the_local_retriever_is_deterministic_and_finds_the_labeled_article_most_of_the_time(chunks, queries):
    retriever = R.LexicalRetriever(chunks)
    assert [c.article for c in retriever.retrieve(queries[0].question)] == [c.article for c in retriever.retrieve(queries[0].question)]
    hits = [any(c.article in q.relevant for c in retriever.retrieve(q.question, k=R.K)) for q in queries]
    assert sum(hits) / len(hits) >= 0.9


def test_every_reported_value_is_within_0_1(results):
    assert results["bounds_check"]["all_within_0_1"] is True
    assert 0.0 <= results["bounds_check"]["min"] <= results["bounds_check"]["max"] <= 1.0
    for cond in results["conditions"].values():
        for stats in cond.values():
            if "mean" in stats:
                assert 0.0 <= stats["min"] <= stats["mean"] <= stats["max"] <= 1.0


def test_the_metrics_separate_the_system_from_the_negative_controls(results):
    s = results["separation"]
    assert s["faithfulness_system_minus_random_context"] > 0.5
    assert s["answer_relevance_system_minus_mismatched"] > 0.1
    assert s["context_precision_labeled_system_minus_random"] > 0.5
    assert s["faithfulness_system_minus_fabricated_figures"] > 0.0
    assert s["share_questions_system_faithfulness_above_random_context"] == 1.0


def test_altering_figures_never_touches_article_numbers_and_changes_the_rest():
    out = R.alter_figures("Artículo 76: plazo de 24 horas y 5 días según el artículo 77.")
    assert "Artículo 76" in out and "artículo 77" in out
    assert "24" not in out and "73" in out and "16" in out


def test_evaluation_is_reproducible(chunks, queries, results):
    again = R.evaluate(R.mock_system(chunks), queries, chunks)
    assert json.dumps(again, sort_keys=True) == json.dumps(results, sort_keys=True)


def test_a_real_pipeline_can_be_plugged_in_through_the_adapter(chunks, queries):
    doc = lambda c: SimpleNamespace(page_content=c.text, metadata={"article_number": c.article, "section": c.section})
    fake = SimpleNamespace(retrieve=lambda q: [doc(c) for c in chunks[:4]],
                           query=lambda q: {"answer": R.extractive_answer(chunks[:4])})
    out = R.adapt_rag_pipeline(fake)(queries[0].question)
    assert out.articles == [c.article for c in chunks[:4]] and len(out.contexts) == 4
    res = R.evaluate(R.adapt_rag_pipeline(fake), queries[:5], chunks)
    assert res["bounds_check"]["all_within_0_1"] and res["conditions"]["system"]["faithfulness"]["mean"] == 1.0


def test_main_writes_strict_json_with_the_expected_sections(tmp_path):
    out = tmp_path / "eval_results.json"
    R.main(["--out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert data["mode"] == "mock" and data["n_articles"] == 27 and data["bounds_check"]["all_within_0_1"] is True
    assert set(data["conditions"]) == {"system", "answer_vs_random_context", "mismatched_answer", "fabricated_figures"}
    assert len(data["per_question"]) == 27 and set(data["retrieval"]) == {"k", "n_questions", "mrr", "hit_at_k"}
