"""
Ingestion pipeline: PDF + CSV → chunks → embeddings → Qdrant
"""
import os
import glob
import pymupdf
import pandas as pd
from src.embedder import embed_batch
from src.store import get_client, ensure_collection, upsert_chunks

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80


def _chunk_text(text: str, source_file: str, page: int) -> list[dict]:
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({"text": chunk, "file": source_file, "page": page})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _load_pdfs(data_dir: str) -> list[dict]:
    chunks = []
    for path in glob.glob(os.path.join(data_dir, "*.pdf")):
        fname = os.path.basename(path)
        doc   = pymupdf.open(path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            chunks.extend(_chunk_text(text, fname, page_num + 1))
        print(f"  [PDF] {fname} → {len(doc)} pages")
    return chunks


def _load_csvs(data_dir: str) -> list[dict]:
    chunks = []
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        fname = os.path.basename(path)
        df    = pd.read_csv(path)
        for _, row in df.iterrows():
            text = " | ".join(
                f"{col}: {val}" for col, val in row.items() if pd.notna(val)
            )
            chunks.append({"text": text, "file": fname, "page": 0})
        print(f"  [CSV] {fname} → {len(df)} rows")
    return chunks


def run_ingest(data_dir: str = "data") -> int:
    print("\n=== NEET RAG — Ingestion ===")
    chunks = _load_pdfs(data_dir) + _load_csvs(data_dir)

    if not chunks:
        raise FileNotFoundError(f"No PDF or CSV files found in '{data_dir}/'")

    print(f"\n  Total chunks : {len(chunks)}")
    print("  Embedding   ...")

    texts   = [c["text"] for c in chunks]
    vectors = embed_batch(texts)

    client = get_client()
    ensure_collection(client)
    upsert_chunks(client, chunks, vectors)

    print(f"  Stored {len(chunks)} chunks in Qdrant")
    print("  Done.\n")
    return len(chunks)
