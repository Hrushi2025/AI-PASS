import fitz  # PyMuPDF
import uuid
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from app.core.config import EVIDENCE_MIN_SCORE, EVIDENCE_MIN_HITS, EVIDENCE_MAX_HITS

COLLECTION = "aipass_docs"

class RAGService:
    def __init__(self, qdrant_url: str = "http://127.0.0.1:6333"):
        self.enabled = False
        self.init_error = None

        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        try:
            self.client = QdrantClient(url=qdrant_url, timeout=2.0)
            self._ensure_collection()
            self.enabled = True
        except Exception as e:
            # No crash: just disable RAG
            self.client = None
            self.enabled = False
            self.init_error = str(e)

    def _ensure_collection(self):
        if not self.client:
            return
        try:
            self.client.get_collection(COLLECTION)
        except Exception:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE),
            )

    def _chunk_text(self, text: str, max_chars: int = 900) -> List[str]:
        text = " ".join(text.split())
        chunks = []
        i = 0
        while i < len(text):
            chunks.append(text[i:i + max_chars])
            i += max_chars
        return chunks

    def ingest_pdf(self, file_bytes: bytes, doc_id: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"doc_id": doc_id, "chunks_indexed": 0, "rag_enabled": False, "error": self.init_error}

        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        points: List[qm.PointStruct] = []
        chunk_count = 0

        for page_idx in range(len(pdf)):
            page_text = pdf[page_idx].get_text("text") or ""
            for c_idx, chunk in enumerate(self._chunk_text(page_text)):
                if not chunk.strip():
                    continue
                emb = self.embedder.encode(chunk).astype(np.float32).tolist()
                chunk_id = f"{doc_id}_p{page_idx}_c{c_idx}"
                point_id = str(uuid.uuid4())

                points.append(
                    qm.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload={
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "page": page_idx + 1,
                            "text": chunk,
                        },
                    )
                )
                chunk_count += 1

        if points:
            self.client.upsert(collection_name=COLLECTION, points=points)

        return {"doc_id": doc_id, "chunks_indexed": chunk_count, "rag_enabled": True}

    def retrieve(self, query: str, doc_ids: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []

        q_emb = self.embedder.encode(query).astype(np.float32).tolist()
        flt = None
        if doc_ids:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))]
            )

        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=q_emb,
            limit=top_k,
            query_filter=flt,
        ).points

        hits = sorted(hits, key=lambda h: float(h.score), reverse=True)

        results: List[Dict[str, Any]] = []
        for h in hits:
            p = h.payload or {}
            results.append({
                "score": float(h.score),
                "doc_id": p.get("doc_id"),
                "chunk_id": p.get("chunk_id"),
                "page": p.get("page"),
                "text": (p.get("text", "") or "")[:500],
            })

        results = [r for r in results if float(r.get("score", 0.0)) >= EVIDENCE_MIN_SCORE]
        if len(results) < EVIDENCE_MIN_HITS:
            return []
        return results[:EVIDENCE_MAX_HITS]
