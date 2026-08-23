<div align="center">

# ⛑️ RAG Seguridad Minera Chile

**Sistema de NLP + RAG para clasificación de incidentes y consulta normativa sobre seguridad minera chilena (DS 132)**

🌐 **[English](README.md)** | **[Español](README.es.md)**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-orange)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

</div>

---

## 1. Problema de negocio

Los equipos de prevención de riesgos en faenas mineras chilenas redactan decenas de reportes narrativos de incidentes cada semana (cuasi-accidentes, atrapamientos de equipos, caídas de roca, incidentes de tronadura, colisiones vehiculares). Hoy dos tareas se hacen manualmente:

1. **Triage de severidad** de cada reporte narrativo (LEVE / GRAVE / FATAL) para priorizar la investigación y los plazos legales de denuncia.
2. **Búsqueda de la normativa aplicable** en el *Reglamento de Seguridad Minera* (Decreto Supremo 132, SERNAGEOMIN) — protocolos de acción inmediata, medidas de control exigidas y sanciones aplicables — un texto legal de ~150 páginas que es lento de revisar manualmente bajo presión de tiempo.

Este proyecto automatiza ambas tareas: un clasificador de texto entrenado sobre narrativas de incidentes predice la severidad, y un asistente de Retrieval-Augmented Generation (RAG) responde preguntas en lenguaje natural sobre el reglamento, citando el/los artículo(s) exacto(s) de donde extrae la información.

> ⚠️ **No reemplaza la revisión legal/normativa.** Ver [§8 Disclaimer normativo](#8-disclaimer-normativo).

## 2. Arquitectura

```
                         ┌───────────────────────────┐
                         │   data/raw_incidents.json  │  144 narrativas de
                         │  (sintético, ligado a DS132)│  incidentes sintéticas
                         └──────────────┬─────────────┘  (jerga minera chilena)
                                        │
                       embeddings (Sentence-Transformers multilingüe)
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │  Modelo Logistic Regression severidad │  LEVE / GRAVE / FATAL
                     │  src/nlp/severity_classifier.py      │
                     └──────────────────┬────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    │                                    ▼                                   │
    │                       FastAPI  /classify-incident                     │
    │                                                                        │
    │   data/ds132_sernageomin.txt  ──►  chunking por "Artículo N"  ──►      │
    │   (extracto curado del DS132)      (src/rag/pipeline.py)              │
    │                                          │                             │
    │                     ┌────────────────────┴────────────────────┐       │
    │                     ▼                                         ▼       │
    │           retriever denso                            retriever disperso│
    │      (embeddings HuggingFace + Chroma)                 (BM25, palabras │
    │                     └────────────────────┬────────────────────┘  clave)│
    │                                  EnsembleRetriever (híbrido)           │
    │                                          │                             │
    │                     Ollama → OpenAI → modo extractivo (fallback)       │
    │                              (backend de síntesis intercambiable)      │
    │                                          │                             │
    │                       FastAPI  /rag-query                             │
    └────────────────────────────────────────────────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │   Dashboard Streamlit      │
                          │ clasificar → RAG automático│
                          └─────────────────────────────┘
```

**Decisiones de diseño relevantes:**

- **Chunking consciente de la estructura**: el reglamento se divide por sus propios encabezados `Artículo N` en vez de trocearlo por cantidad de caracteres, de modo que ningún chunk queda a mitad de frase entre dos artículos distintos — clave en un dominio legal donde la precisión de la cita importa.
- **Retrieval híbrido**: un retriever denso (similitud semántica vía embeddings multilingües) se combina con un retriever disperso BM25 (coincidencias exactas de palabras clave) mediante `EnsembleRetriever`. El retrieval denso por sí solo puede perder números de artículo exactos o términos técnicos específicos (*"acuñadura"*, *"pértiga"*) que BM25 captura de forma confiable.
- **Backend LLM intercambiable y siempre funcional**: el pipeline intenta primero Ollama (modelo local), luego OpenAI (si `OPENAI_API_KEY` está configurada), y cae a un modo **extractivo** (sin LLM en absoluto — solo los extractos de artículos citados y rankeados) si ninguno está disponible. Esto hace que el sistema sea 100% testeable y demostrable sin dependencias externas ni API keys.

## 3. Stack tecnológico

| Capa | Elección |
|---|---|
| Lenguaje | Python 3.11+ (también probado en 3.10.11) |
| Framework RAG | LangChain 1.x (`langchain-classic`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`) |
| Vector store | ChromaDB (local, persistido en disco) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (con soporte de español) |
| Retrieval disperso | BM25 (`rank-bm25`) |
| Clasificador | Logistic Regression (scikit-learn) sobre embeddings de oraciones |
| Síntesis LLM | Ollama (local) → API de OpenAI → modo extractivo (fallback) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Tests | pytest |

## 4. Estructura del proyecto

```
rag-seguridad-minera-chile/
├── data/
│   ├── raw_incidents.json          # 144 narrativas de incidentes sintéticas
│   ├── ds132_sernageomin.txt       # Extracto curado del DS 132 (texto real)
│   ├── chroma_db/                  # Vector store (generado, en .gitignore)
│   └── models/                     # Clasificador entrenado (generado, en .gitignore)
├── src/
│   ├── nlp/
│   │   ├── dataset_generator.py    # Generador de dataset sintético de incidentes
│   │   └── severity_classifier.py  # Clasificador embeddings + LogisticRegression
│   ├── rag/
│   │   └── pipeline.py             # Chunking, retriever híbrido, backends LLM
│   ├── api/
│   │   └── main.py                 # FastAPI: /classify-incident, /rag-query
│   └── app/
│       └── streamlit_app.py        # Dashboard interactivo
├── tests/
│   ├── test_dataset_generator.py
│   ├── test_rag_pipeline.py
│   └── test_severity_classifier.py
├── requirements.txt
├── pytest.ini
├── .gitignore
├── README.md
└── README.es.md
```

## 5. Instalación

```powershell
git clone https://github.com/Rxyxs/rag-seguridad-minera-chile.git
cd rag-seguridad-minera-chile

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 6. Uso

**1. Generar el dataset sintético de incidentes**

```powershell
python -m src.nlp.dataset_generator
```
Escribe `data/raw_incidents.json` (144 incidentes en 12 tipos, con seed determinística).

**2. Entrenar el clasificador de severidad**

```powershell
python -m src.nlp.severity_classifier
```
Entrena sobre embeddings de oraciones, evalúa en un split de validación, guarda `data/models/severity_classifier.joblib`.

**3. Ejecutar el pipeline RAG (demo CLI)**

```powershell
python -m src.rag.pipeline
```
Construye el retriever híbrido sobre `data/ds132_sernageomin.txt` y responde una pregunta de demostración, imprimiendo los artículos citados y la respuesta sintetizada.

**4. Levantar la API**

```powershell
uvicorn src.api.main:app --reload
```
Documentación interactiva en `http://127.0.0.1:8000/docs`.

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado del servicio y de los modelos cargados |
| `/classify-incident` | POST | `{"narrative_text": "..."}` → severidad predicha + probabilidades |
| `/rag-query` | POST | `{"question": "..."}` → respuesta sintetizada + artículos del DS132 citados |

**5. Levantar el dashboard**

```powershell
streamlit run src/app/streamlit_app.py
```
Tres pestañas: clasificar un incidente (encadena automáticamente una consulta RAG por los protocolos aplicables), consulta normativa libre, y una vista general del dataset.

**Opcional: usar un LLM real en vez del fallback extractivo**

```powershell
# Local, vía Ollama (requiere `ollama pull llama3.1` primero)
$env:RAG_LLM_BACKEND = "ollama"

# En la nube, vía OpenAI
$env:RAG_LLM_BACKEND = "openai"
$env:OPENAI_API_KEY = "sk-..."
```

## 7. Resultados validados

Todos los números a continuación fueron producidos ejecutando realmente los scripts de este repositorio (no son estimaciones):

| Métrica | Valor |
|---|---|
| Incidentes sintéticos generados | 144 (GRAVE: 71, LEVE: 33, FATAL: 40) |
| Tipos de incidente cubiertos | 12 (con jerga minera chilena: CAEX, chancador, frente de avance, acuñadura, EPP, pértiga, tronadura...) |
| Clasificador de severidad — Accuracy | **0.750** |
| Clasificador de severidad — F1-macro | **0.719** |
| Split train / test | 108 / 36 (75/25, estratificado en la generación) |
| Artículos del DS 132 indexados por el pipeline RAG | 27 (extracto curado — ver disclaimer abajo) |
| Mecanismo de retrieval | Híbrido (denso + BM25), verificado recuperando correctamente el artículo 247 para una pregunta sobre "pértiga" y los artículos 157–162 para una pregunta sobre "fortificación/acuñadura" |
| Suite de tests | 14/14 pasando (`pytest`) |

## 8. Disclaimer normativo

`data/ds132_sernageomin.txt` es un **extracto curado** del texto real y textual del **Decreto Supremo N° 132** (*Reglamento de Seguridad Minera*, SERNAGEOMIN, Chile), que cubre ~27 artículos sobre investigación/denuncia de accidentes, respuesta a emergencias, fortificación y acuñadura, ventilación, tronadura, seguridad vehicular en torno a CAEX, y sanciones. **No** es el reglamento completo de 592 artículos, y este sistema **no** reemplaza la consulta del texto oficial completo y vigente, ni a un profesional de prevención de riesgos o legal calificado. Fuentes:

- SERNAGEOMIN (PDF oficial, replicado vía sigweb.cl): `https://sigweb.cl/wp-content/uploads/biblioteca/ReglamentoSeguridadMinera.pdf`
- Biblioteca del Congreso Nacional (LeyChile): `https://www.bcn.cl/leychile/Navegar?idNorma=221064`

El prompt de sistema del RAG instruye explícitamente al LLM a responder solo en base al contexto recuperado, citar números de artículo, y admitir cuando el contexto es insuficiente — está diseñado para nunca inventar cifras, multas o números de artículo.

## 9. Testing

```powershell
pytest -v
```
14 tests que cubren invariantes de la generación del dataset, chunking del reglamento (sin artículos duplicados, contenido de artículos conocidos), relevancia del retrieval híbrido para dos temas independientes, y validez de la salida del clasificador (conjunto de etiquetas, rangos de métricas, normalización de probabilidades).

## 10. Posibles extensiones

- Reemplazar el extracto curado del DS132 por el texto completo de 592 artículos (la lógica de chunking ya lo soporta — sin cambios de código).
- Agregar re-ranking con cross-encoder sobre el retriever híbrido para mayor precisión en consultas ambiguas.
- Explicaciones SHAP multiclase para el clasificador de severidad (ver [chile-mining-predictive-maintenance](https://github.com/Rxyxs/chile-mining-predictive-maintenance) para un ejemplo de este patrón en un proyecto hermano).
- Persistir clasificaciones de incidentes y respuestas RAG en una base de datos para trazabilidad de auditoría.

## Licencia

MIT — ver [LICENSE](LICENSE).

## Autor

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
