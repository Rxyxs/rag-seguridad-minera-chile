<div align="center">

# ⛑️ RAG Seguridad Minera Chile

**NLP + RAG system for incident classification and regulatory Q&A on Chilean mining safety law (DS 132)**

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

## 1. Business problem

Risk-prevention teams at Chilean mining operations write dozens of narrative incident reports every week (near-misses, equipment entrapments, rockfalls, blasting incidents, vehicle collisions). Two tasks currently done by hand, manually, are:

1. **Triage severity** of each narrative report (LEVE / GRAVE / FATAL) to prioritize investigation and legal reporting deadlines.
2. **Look up applicable regulation** in the *Reglamento de Seguridad Minera* (Supreme Decree 132, SERNAGEOMIN) — immediate-action protocols, required control measures, and the sanctions that apply — a ~150-page legal text that is slow to search manually under time pressure.

This project automates both: a text classifier trained on incident narratives predicts severity, and a Retrieval-Augmented Generation (RAG) assistant answers natural-language questions against the regulation, citing the exact article(s) it draws from.

> ⚠️ **Not a substitute for legal/compliance review.** See [§8 Regulatory disclaimer](#8-regulatory-disclaimer).

## 2. Architecture

```
                         ┌───────────────────────────┐
                         │   data/raw_incidents.json  │  144 synthetic incident
                         │  (synthetic, DS132-linked) │  narratives (Chilean
                         └──────────────┬─────────────┘  mining jargon)
                                        │
                          embed (multilingual Sentence-Transformers)
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │  Logistic Regression severity model  │  LEVE / GRAVE / FATAL
                     │  src/nlp/severity_classifier.py      │
                     └──────────────────┬────────────────────┘
                                        │
    ┌───────────────────────────────────┼───────────────────────────────────┐
    │                                    ▼                                   │
    │                       FastAPI  /classify-incident                     │
    │                                                                        │
    │   data/ds132_sernageomin.txt  ──►  chunk by "Artículo N"  ──►         │
    │   (curated DS132 excerpt)          (src/rag/pipeline.py)              │
    │                                          │                             │
    │                     ┌────────────────────┴────────────────────┐       │
    │                     ▼                                         ▼       │
    │            dense retriever                            sparse retriever│
    │      (HuggingFace embeddings + Chroma)                 (BM25 keyword) │
    │                     └────────────────────┬────────────────────┘       │
    │                                  EnsembleRetriever (hybrid)            │
    │                                          │                             │
    │                     Ollama → OpenAI → extractive fallback              │
    │                              (pluggable synthesis backend)             │
    │                                          │                             │
    │                       FastAPI  /rag-query                             │
    └────────────────────────────────────────────────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          │   Streamlit dashboard      │
                          │  classify → auto RAG query │
                          └─────────────────────────────┘
```

**Design decisions worth calling out:**

- **Structure-aware chunking**: the regulation is split by its own `Artículo N` headers instead of naive character-count splitting, so a chunk is never a half-sentence from two different articles — critical for a legal domain where citation precision matters.
- **Hybrid retrieval**: a dense retriever (semantic similarity via multilingual embeddings) is combined with a sparse BM25 retriever (exact keyword matches) via `EnsembleRetriever`. Dense retrieval alone can miss exact article numbers or narrow technical terms (*"acuñadura"*, *"pértiga"*) that BM25 catches reliably.
- **Pluggable, always-functional LLM backend**: the pipeline tries Ollama (local model) first, then OpenAI (if `OPENAI_API_KEY` is set), and falls back to an **extractive** mode (no LLM at all — just the ranked, cited article excerpts) if neither is available. This means the system is 100% testable and demoable with zero external dependencies or API keys.

## 3. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (tested on 3.10.11 as well) |
| RAG framework | LangChain 1.x (`langchain-classic`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`) |
| Vector store | ChromaDB (local, persisted to disk) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Spanish-capable) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Classifier | Logistic Regression (scikit-learn) on sentence embeddings |
| LLM synthesis | Ollama (local) → OpenAI API → extractive fallback |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Tests | pytest |

## 4. Project structure

```
rag-seguridad-minera-chile/
├── data/
│   ├── raw_incidents.json          # 144 synthetic incident narratives
│   ├── ds132_sernageomin.txt       # Curated excerpt of DS 132 (real text)
│   ├── chroma_db/                  # Vector store (generated, gitignored)
│   └── models/                     # Trained classifier (generated, gitignored)
├── src/
│   ├── nlp/
│   │   ├── dataset_generator.py    # Synthetic incident dataset generator
│   │   └── severity_classifier.py  # Embedding + LogisticRegression classifier
│   ├── rag/
│   │   └── pipeline.py             # Chunking, hybrid retriever, LLM backends
│   ├── api/
│   │   └── main.py                 # FastAPI: /classify-incident, /rag-query
│   └── app/
│       └── streamlit_app.py        # Interactive dashboard
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

## 5. Setup

```powershell
git clone https://github.com/Rxyxs/rag-seguridad-minera-chile.git
cd rag-seguridad-minera-chile

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 6. Usage

**1. Generate the synthetic incident dataset**

```powershell
python -m src.nlp.dataset_generator
```
Writes `data/raw_incidents.json` (144 incidents across 12 incident types, seeded/deterministic).

**2. Train the severity classifier**

```powershell
python -m src.nlp.severity_classifier
```
Trains on sentence embeddings, evaluates on a held-out split, saves `data/models/severity_classifier.joblib`.

**3. Run the RAG pipeline (CLI demo)**

```powershell
python -m src.rag.pipeline
```
Builds the hybrid retriever over `data/ds132_sernageomin.txt` and answers a demo question, printing the cited articles and synthesized answer.

**4. Start the API**

```powershell
uvicorn src.api.main:app --reload
```
Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service + model load status |
| `/classify-incident` | POST | `{"narrative_text": "..."}` → predicted severity + probabilities |
| `/rag-query` | POST | `{"question": "..."}` → synthesized answer + cited DS132 articles |

**5. Start the dashboard**

```powershell
streamlit run src/app/streamlit_app.py
```
Three tabs: classify an incident (auto-chains into a RAG query for applicable protocols), free-form regulatory Q&A, and a dataset overview.

**Optional: use a real LLM instead of the extractive fallback**

```powershell
# Local, via Ollama (requires `ollama pull llama3.1` first)
$env:RAG_LLM_BACKEND = "ollama"

# Cloud, via OpenAI
$env:RAG_LLM_BACKEND = "openai"
$env:OPENAI_API_KEY = "sk-..."
```

## 7. Validated results

All numbers below were produced by actually running the scripts in this repo (not estimated):

| Metric | Value |
|---|---|
| Synthetic incidents generated | 144 (GRAVE: 71, LEVE: 33, FATAL: 40) |
| Incident types covered | 12 (with Chilean mining jargon: CAEX, chancador, frente de avance, acuñadura, EPP, pértiga, tronadura...) |
| Severity classifier — Accuracy | **0.750** |
| Severity classifier — F1-macro | **0.719** |
| Train / test split | 108 / 36 (75/25, stratified by generation) |
| DS 132 articles indexed by the RAG pipeline | 27 (curated excerpt — see disclaimer below) |
| Retrieval mechanism | Hybrid (dense + BM25), verified to correctly retrieve article 247 for a "pértiga" question and articles 157–162 for a "fortificación/acuñadura" question |
| Test suite | 14/14 passing (`pytest`) |

## 8. Regulatory disclaimer

`data/ds132_sernageomin.txt` is a **curated excerpt** of the real, verbatim text of **Decreto Supremo N° 132** (*Reglamento de Seguridad Minera*, SERNAGEOMIN, Chile), covering ~27 articles across accident investigation/reporting, emergency response, ground support (*fortificación y acuñadura*), ventilation, blasting, vehicle safety around CAEX, and sanctions. It is **not** the complete 592-article regulation, and this system is **not** a substitute for consulting the full, currently-in-force official text or a qualified prevention-risk / legal professional. Sources:

- SERNAGEOMIN (official PDF, mirrored via sigweb.cl): `https://sigweb.cl/wp-content/uploads/biblioteca/ReglamentoSeguridadMinera.pdf`
- Biblioteca del Congreso Nacional (LeyChile): `https://www.bcn.cl/leychile/Navegar?idNorma=221064`

The RAG system prompt explicitly instructs the LLM to answer only from retrieved context, cite article numbers, and admit when the context is insufficient — it is designed to never invent figures, fines, or article numbers.

## 9. Testing

```powershell
pytest -v
```
14 tests across dataset generation invariants, regulation chunking (no duplicate articles, known-article content), hybrid retrieval relevance for two independent topics, and classifier output validity (label set, metric ranges, probability normalization).

## 10. Possible extensions

- Swap the curated DS132 excerpt for the full 592-article text (chunking logic already supports it — no code change needed).
- Add cross-encoder re-ranking on top of the hybrid retriever for higher precision on ambiguous queries.
- Multiclass SHAP explanations for the severity classifier (see [chile-mining-predictive-maintenance](https://github.com/Rxyxs/chile-mining-predictive-maintenance) for a worked example of this pattern in a sibling project).
- Persist incident classifications + RAG answers to a database for audit trail.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)
