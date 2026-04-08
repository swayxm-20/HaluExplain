# HaluExplain — Hallucination Detection & Verification Engine

> A production-ready system that evaluates AI-generated claims for factual accuracy using **self-consistency analysis**, **hybrid evidence retrieval**, **cross-encoder NLI**, and **natural language explanation generation**.

---

## System Architecture

```
POST /verify_claim
       │
       ▼
┌─────────────────────┐
│  consistency.py     │  ← 5 mock LLM responses → AgglomerativeClustering
│                     │    → consistency_score + conflicting_statements
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  retriever.py       │  ← SBERT dense + BM25 sparse → RRF fusion
│                     │    → Top-K ranked evidence passages
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  entailment.py      │  ← DeBERTa-v3-large cross-encoder NLI
│                     │    → verdict + confidence + probabilities
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  explanation.py     │  ← BART-large-cnn abstractive summarization
│                     │    → human-readable explanation paragraph
└────────┬────────────┘
         ▼
      JSON Response
```

---

## Models Used

| Role | Model | Library |
|---|---|---|
| Sentence embeddings (consistency + dense retrieval) | `sentence-transformers/all-mpnet-base-v2` | sentence-transformers |
| NLI entailment | `cross-encoder/nli-deberta-v3-large` | sentence-transformers |
| Explanation generation | `facebook/bart-large-cnn` | transformers |
| Sparse retrieval | BM25Okapi | rank-bm25 |
| Clustering | AgglomerativeClustering | scikit-learn |

> **First run note:** HuggingFace will automatically download these models (~2.5 GB total) to `~/.cache/huggingface`. Ensure you have a stable internet connection for the first startup.

---

## Prerequisites

- **Python 3.10 or 3.11** (recommended)
- pip ≥ 23
- ~4 GB RAM (8 GB recommended for all models in-memory simultaneously)
- Disk space: ~3 GB for model downloads

> **GPU (optional):** If you have a CUDA-capable GPU, replace the `torch` line in `requirements.txt` with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/). GPU will significantly reduce latency for the NLI and explanation modules.

---

## Installation

### Step 1 — Clone / navigate to the project

```bash
cd HaluExplain
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 5 — (Optional) Pre-warm models

Running the server the first time will download all HuggingFace models (~2–3 min on a fast connection). You can pre-warm them by running:

```bash
python -c "
from app.consistency import ConsistencyAnalyser
from app.retriever import HybridRetriever
from app.entailment import EntailmentEngine
from app.explanation import ExplanationGenerator
print('All models loaded successfully.')
"
```

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Base**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## API Reference

### `GET /health`

Liveness check. Returns model status.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": true
}
```

---

### `POST /verify_claim`

Runs the full hallucination detection pipeline.

**Request body:**

```json
{
  "claim": "The Great Wall of China is visible from space with the naked eye.",
  "context_docs": ["Optional extra passages to include in retrieval..."],
  "top_k": 5
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `claim` | string | ✅ | The AI-generated claim to verify (min 10 chars) |
| `context_docs` | list[str] | ❌ | Additional domain-specific passages to add to the KB |
| `top_k` | int (1–20) | ❌ | Number of evidence passages to retrieve (default: 5) |

**Response body:**

```json
{
  "claim": "The Great Wall of China is visible from space...",
  "consistency": {
    "score": 0.75,
    "conflicting_statements": ["There is absolutely no scientific evidence that..."],
    "total_sentences": 16,
    "consensus_sentences": 12
  },
  "evidence": [
    {
      "text": "The Great Wall of China is not visible from space...",
      "rrf_score": 0.030303,
      "source": "NASA/Space Facts"
    }
  ],
  "entailment": {
    "verdict": "CONTRADICTION",
    "confidence": 0.9123,
    "probabilities": {
      "contradiction": 0.9123,
      "neutral": 0.0612,
      "entailment": 0.0265
    }
  },
  "explanation": "The claim 'The Great Wall of China is visible from space with the naked eye.' is classified as CONTRADICTION (confidence: 91.2%). Evidence contradicts that the Great Wall of China is not visible from space with the naked eye. Suggested correction: 'Consider revising the claim to align with the evidence.'",
  "processing_time_ms": 4231.5
}
```

---

## Testing with `curl`

```bash
# Health check
curl http://localhost:8000/health

# Verify a claim
curl -X POST http://localhost:8000/verify_claim \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The Great Wall of China is visible from space with the naked eye.",
    "top_k": 3
  }'
```

```powershell
# PowerShell alternative
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/verify_claim" `
  -ContentType "application/json" `
  -Body '{"claim": "Humans only use 10% of their brains.", "top_k": 3}'
```

---

## Project Structure

```
HaluExplain/
├── app/
│   ├── __init__.py
│   ├── consistency.py        # Module 1 — Self-Consistency Analysis
│   ├── retriever.py          # Module 2 — Hybrid Evidence Retrieval
│   ├── entailment.py         # Module 3 — NLI/Entailment Engine
│   ├── explanation.py        # Module 4 — Explanation Generator
│   └── models.py             # Pydantic request/response schemas
├── data/
│   └── knowledge_base.json   # 20 curated factual passages (built-in corpus)
├── main.py                   # FastAPI app + orchestration
├── config.py                 # All hyperparameters and model names
├── requirements.txt
└── README.md
```

---

## Extending the Knowledge Base

Add more passages to `data/knowledge_base.json` in this format:

```json
[
  {
    "text": "Your factual passage text here.",
    "source": "Your source label (optional)"
  }
]
```

You can also inject per-request documents via the `context_docs` field in the API request — no restart required.

---

## Configuration

All tuneable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CLUSTERING_DISTANCE_THRESHOLD` | `0.4` | Cosine distance cut-off for clustering |
| `CONSENSUS_MIN_CLUSTER_SIZE` | `2` | Min sentences to count as consensus |
| `TOP_K_PASSAGES` | `5` | Default retrieved passages |
| `RRF_K` | `60` | RRF smoothing constant |
| `UNCERTAINTY_THRESHOLD` | `0.50` | Min confidence to assign a verdict |
| `EXPLANATION_MAX_LENGTH` | `220` | Max BART output tokens |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `LookupError: Resource punkt not found` | Run the NLTK download step in the README |
| `OutOfMemoryError` during startup | Reduce `EXPLANATION_NUM_BEAMS` in `config.py` or use a smaller explanation model |
| Slow first request | Expected — HuggingFace models initialise lazily on first inference. Use the pre-warm script. |
| `503 Service Unavailable` | Models are still loading. Wait ~30s and retry. |

---

## License

MIT License — free to use, modify, and distribute.
