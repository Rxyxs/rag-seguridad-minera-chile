import pytest

from src.nlp.dataset_generator import generate_dataset
from src.nlp.severity_classifier import (
    SEVERITY_LEVELS,
    embed_texts,
    evaluate_classifier,
    get_embedder,
    predict_severity,
    train_severity_classifier,
)


@pytest.fixture(scope="module")
def trained_classifier():
    records = generate_dataset(n_per_type=10, seed=7)
    texts = [r["narrative_text"] for r in records]
    labels = [r["severity"] for r in records]

    embedder = get_embedder()
    X = embed_texts(texts, embedder)

    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = labels[:split], labels[split:]

    model = train_severity_classifier(X_train, y_train)
    return embedder, model, X_test, y_test


def test_classifier_predicts_valid_labels(trained_classifier):
    embedder, model, X_test, y_test = trained_classifier
    preds = model.predict(X_test)
    assert all(p in SEVERITY_LEVELS for p in preds)


def test_evaluate_classifier_metrics_in_valid_range(trained_classifier):
    _, model, X_test, y_test = trained_classifier
    metrics = evaluate_classifier(model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_predict_severity_returns_probabilities_summing_to_one(trained_classifier):
    embedder, model, _, _ = trained_classifier
    result = predict_severity("Un trabajador sufrió una caída de rocas en el frente de avance.", embedder, model)
    assert result["predicted_severity"] in SEVERITY_LEVELS
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
