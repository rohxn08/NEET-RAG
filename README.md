# NEET.info RAG Counselling Assistant

A locally hosted RAG assistant for NEET UG cut-off and seat-matrix Q&A.  
Built for CGPRR AI Developer assignment — Track A.

## Hardware

- **Developed on**: CPU-only (no dedicated GPU)
- **Target deployment**: NVIDIA RTX 6000 Ada (48 GB VRAM)  
  On that card, the same stack runs `llama3.1:70b-instruct-q4_K_M` at ~40 tok/sec.

## Stack

| Layer | Tool |
|-------|------|
| PDF / CSV parsing | PyMuPDF + pandas |
| Embeddings | `nomic-embed-text` via Ollama (768-dim) |
| Vector store | Qdrant (Docker) |
| LLM | `llama3.2:3b` via Ollama |
| RAG agent | pydantic-ai |
| UI | Streamlit |

## Setup

### 1. Prerequisites

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Start Qdrant
docker compose up -d

# Install Python deps (Python 3.11+)
pip install -r requirements.txt
```

### 2. Build the index

```bash
python ingest.py
# Optional: point to a custom data folder
python ingest.py --data-dir /path/to/your/pdfs
```

Place any additional NEET cut-off PDFs or CSVs in `data/` before running.  
A synthetic dataset (`data/neet_cutoffs_2024.csv`) ships with the repo.

### 3. Run the assistant

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run evaluation

```bash
python evaluate.py
```

Prints a pass/fail table for all 20 questions in `eval/eval_set.json`,  
including 8 refusal cases (out-of-index, prompt injection, personal guarantees).

## Data

`data/neet_cutoffs_2024.csv` is **synthetic** data generated for this assignment.  
It follows the structure of NTA MCC published cut-offs (college, course, category, state, year, round, opening rank, closing rank).  
No real student records are used or stored.

## Privacy

- All data, the vector index, and LLM inference stay on the local machine.
- Nothing is sent to an external API during normal operation.
- In an office deployment, each business's data would live in a separate Qdrant collection with access controls.

## AI tools used

Claude Code (Anthropic) was used for code generation and project scaffolding.
