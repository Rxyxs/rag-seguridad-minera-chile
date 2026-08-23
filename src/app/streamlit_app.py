"""Dashboard interactivo para ingenieros de prevención de riesgos.

Ejecutar desde la raíz del repositorio con:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import streamlit as st

from src.nlp.dataset_generator import DATA_DIR
from src.nlp.severity_classifier import MODELS_DIR, get_embedder, predict_severity
from src.rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Seguridad Minera Chile", layout="wide")

SEVERITY_COLORS = {"LEVE": "#2ca02c", "GRAVE": "#ff7f0e", "FATAL": "#d62728"}


@st.cache_resource
def load_severity_model():
    model_path = MODELS_DIR / "severity_classifier.joblib"
    if not model_path.exists():
        return None, None
    return get_embedder(), joblib.load(model_path)


@st.cache_resource
def load_rag_pipeline() -> RAGPipeline | None:
    try:
        return RAGPipeline()
    except Exception as exc:  # noqa: BLE001 -- se muestra en la UI, no debe tumbar la app
        st.session_state["rag_load_error"] = str(exc)
        return None


@st.cache_data
def load_incidents() -> pd.DataFrame:
    path = DATA_DIR / "raw_incidents.json"
    if not path.exists():
        return pd.DataFrame()
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def render_rag_answer(pipeline: RAGPipeline, question: str) -> None:
    with st.spinner(f"Consultando el DS 132 (backend: {pipeline.backend_name})..."):
        result = pipeline.query(question)

    st.markdown(f"**Backend de síntesis:** `{result['backend']}`")
    st.markdown(result["answer"])
    with st.expander(f"Artículos del DS 132 citados ({len(result['sources'])})"):
        for source in result["sources"]:
            st.write(f"- Artículo {source['article_number']} — {source['section']}")


def main() -> None:
    st.title("⛑️ RAG Seguridad Minera Chile")
    st.caption(
        "Clasificación de severidad de incidentes + consulta normativa al Reglamento de "
        "Seguridad Minera (DS 132, SERNAGEOMIN)"
    )

    tab_classify, tab_query, tab_dataset = st.tabs(
        ["🔎 Clasificar incidente", "📖 Consulta normativa", "📊 Dataset de incidentes"]
    )

    with tab_classify:
        st.subheader("Clasificar narrativa de incidente")
        embedder, severity_model = load_severity_model()

        if severity_model is None:
            st.warning(
                "No se encontró el clasificador entrenado. Corre "
                "`python -m src.nlp.severity_classifier` primero."
            )
        else:
            example_df = load_incidents()
            example_text = ""
            if not example_df.empty:
                if st.checkbox("Cargar un ejemplo del dataset sintético"):
                    example_row = example_df.sample(1, random_state=None).iloc[0]
                    example_text = example_row["narrative_text"]

            narrative = st.text_area(
                "Narrativa del incidente", value=example_text, height=140,
                placeholder="Ej: Durante el turno noche, un operador de CAEX...",
            )

            if st.button("Clasificar y consultar DS 132", type="primary") and narrative.strip():
                prediction = predict_severity(narrative, embedder, severity_model)
                severity = prediction["predicted_severity"]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Severidad predicha", severity)
                    st.metric("Confianza", f"{prediction['confidence']:.0%}")
                    st.bar_chart(pd.Series(prediction["probabilities"]))
                with col2:
                    st.markdown("**Protocolos y normativa aplicable (RAG sobre DS 132):**")
                    pipeline = load_rag_pipeline()
                    if pipeline is None:
                        st.error(
                            "No se pudo cargar el pipeline RAG: "
                            f"{st.session_state.get('rag_load_error', 'error desconocido')}"
                        )
                    else:
                        rag_question = (
                            f"Un incidente fue clasificado como severidad {severity}. Narrativa: "
                            f"'{narrative}'. ¿Qué protocolos de acción inmediata, medidas de "
                            f"control y sanciones normativas aplican según el DS 132?"
                        )
                        render_rag_answer(pipeline, rag_question)

    with tab_query:
        st.subheader("Consulta libre al Reglamento de Seguridad Minera")
        pipeline = load_rag_pipeline()
        if pipeline is None:
            st.error(
                "No se pudo cargar el pipeline RAG: "
                f"{st.session_state.get('rag_load_error', 'error desconocido')}"
            )
        else:
            st.info(f"Backend de síntesis activo: **{pipeline.backend_name}**")
            question = st.text_input(
                "Tu pregunta",
                placeholder="¿Cada cuánto tiempo debe revisarse la fortificación de un pique?",
            )
            if st.button("Consultar") and question.strip():
                render_rag_answer(pipeline, question)

    with tab_dataset:
        st.subheader("Dataset sintético de incidentes")
        df = load_incidents()
        if df.empty:
            st.warning(
                "No se encontró el dataset. Corre `python -m src.nlp.dataset_generator` primero."
            )
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Incidentes totales", len(df))
            col2.metric("Tipos de incidente", df["incident_type"].nunique())
            col3.metric("Faenas", df["faena"].nunique())

            st.bar_chart(df["severity"].value_counts())
            st.dataframe(
                df[["incident_id", "incident_type", "severity", "faena", "turno", "ubicacion"]],
                use_container_width=True,
                height=350,
            )


if __name__ == "__main__":
    main()
