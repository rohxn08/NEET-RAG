import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from src.embedder import EMBED_DIM

COLLECTION    = "neet_docs"
_QDRANT_HOST  = os.getenv("QDRANT_HOST", "localhost")


def get_client() -> QdrantClient:
    return QdrantClient(host=_QDRANT_HOST, port=6333)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def upsert_chunks(client: QdrantClient, chunks: list[dict], vectors: list[list[float]]) -> None:
    points = [
        PointStruct(id=i, vector=vec, payload=chunk)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    # Qdrant upsert in batches of 100
    for start in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION, points=points[start:start+100])


def search(client: QdrantClient, query_vec: list[float], top_k: int = 5) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text":  hit.payload.get("text", ""),
            "file":  hit.payload.get("file", ""),
            "page":  hit.payload.get("page", 0),
            "score": hit.score,
        }
        for hit in results.points
    ]


def collection_count(client: QdrantClient) -> int:
    return client.get_collection(COLLECTION).points_count
