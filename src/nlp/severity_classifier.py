"""Clasificador de severidad de incidentes (LEVE / GRAVE / FATAL).

Enfoque: embeddings de oraciones (Sentence-Transformers, modelo
multilingüe con soporte de español) + un clasificador lineal clásico
(regresión logística) sobre esos embeddings. Es liviano de entrenar y
servir comparado con fine-tunear un transformer completo, y funciona bien
con dataset chicos como el sintético de este proyecto.

El split train/test es a nivel de incidente (aleatorio estratificado por
severidad) -- no hay dimensión temporal aquí, a diferencia de otros
proyectos de esta serie.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "data" / "models"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SEVERITY_LEVELS = ["LEVE", "GRAVE", "FATAL"]


def load_incidents(path: Path | None = None) -> list[dict]:
    path = path or (DATA_DIR / "raw_incidents.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Carga (y cachea en proceso) el modelo de embeddings."""
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def embed_texts(texts: list[str], embedder: SentenceTransformer) -> np.ndarray:
    return embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def train_severity_classifier(
    X_train: np.ndarray, y_train: list[str], seed: int = 42
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model: LogisticRegression, X_test: np.ndarray, y_test: list[str]) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(
            y_test, y_pred, labels=SEVERITY_LEVELS, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=SEVERITY_LEVELS).tolist(),
    }


def predict_severity(narrative_text: str, embedder: SentenceTransformer, model: LogisticRegression) -> dict:
    embedding = embed_texts([narrative_text], embedder)
    probs = model.predict_proba(embedding)[0]
    classes = model.classes_
    proba_map = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
    predicted = classes[int(np.argmax(probs))]
    return {
        "predicted_severity": predicted,
        "confidence": round(float(max(probs)), 4),
        "probabilities": proba_map,
    }


def main() -> None:
    incidents = load_incidents()
    texts = [r["narrative_text"] for r in incidents]
    labels = [r["severity"] for r in incidents]

    print(f"Incidentes cargados: {len(incidents)}")
    print(f"Cargando modelo de embeddings: {EMBEDDING_MODEL_NAME} (puede tardar la primera vez)...")
    embedder = get_embedder()

    print("Calculando embeddings...")
    X = embed_texts(texts, embedder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"Train: {len(y_train)}  Test: {len(y_test)}")

    model = train_severity_classifier(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)

    print(f"\nAccuracy: {metrics['accuracy']:.3f}")
    print(f"F1-macro: {metrics['f1_macro']:.3f}")
    print("Matriz de confusión (filas=real, columnas=predicho, orden LEVE/GRAVE/FATAL):")
    for row in metrics["confusion_matrix"]:
        print(" ", row)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "severity_classifier.joblib")
    with open(MODELS_DIR / "severity_classifier_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"embedding_model": EMBEDDING_MODEL_NAME, **metrics}, f, indent=2, ensure_ascii=False
        )

    print(f"\nModelo guardado en: {MODELS_DIR / 'severity_classifier.joblib'}")

    example = texts[0]
    prediction = predict_severity(example, embedder, model)
    print(f"\nEjemplo de predicción sobre: '{example[:80]}...'")
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
