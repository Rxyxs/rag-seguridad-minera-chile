from src.rag.pipeline import RAGPipeline, chunk_regulation, load_regulation_text


def test_chunk_regulation_produces_one_chunk_per_article():
    text = load_regulation_text()
    chunks = chunk_regulation(text)
    assert len(chunks) >= 20  # el extracto curado tiene ~30 articulos reales
    article_numbers = [c["article_number"] for c in chunks]
    assert len(article_numbers) == len(set(article_numbers)), "hay articulos duplicados"


def test_chunk_regulation_captures_known_article():
    text = load_regulation_text()
    chunks = chunk_regulation(text)
    art_162 = next(c for c in chunks if c["article_number"] == "162")
    assert "acuñadura" in art_162["text"].lower()
    assert "Fortificación" in art_162["section"]


def test_chunk_regulation_captures_sanctions_article():
    text = load_regulation_text()
    chunks = chunk_regulation(text)
    art_590 = next(c for c in chunks if c["article_number"] == "590")
    assert "Unidades Tributarias" in art_590["text"]


def test_pipeline_extractive_mode_retrieves_relevant_articles():
    pipeline = RAGPipeline(llm_preference="extractive")
    assert pipeline.backend_name == "extractive"

    result = pipeline.query("¿Qué distancia mínima debe tener la pértiga de un vehículo liviano?")
    retrieved_articles = {s["article_number"] for s in result["sources"]}
    assert "247" in retrieved_articles


def test_pipeline_retrieves_fortification_articles_for_acunadura_question():
    pipeline = RAGPipeline(llm_preference="extractive")
    result = pipeline.query("¿Cada cuánto se debe revisar la fortificación de un pique?")
    retrieved_articles = {s["article_number"] for s in result["sources"]}
    assert retrieved_articles & {"157", "158", "159", "160", "162"}
