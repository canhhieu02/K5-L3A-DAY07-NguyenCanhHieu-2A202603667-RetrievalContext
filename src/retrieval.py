"""Day 7 Retrieval Module: Dense (pgvector/chroma), Lexical (TF-IDF/BM25), and Hybrid (RRF)."""
from __future__ import annotations
import json
import math
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text with NFC normalization."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


class PgVectorRetriever:
    """Dense retriever querying PostgreSQL + pgvector with HNSW index."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 55433,
        dbname: str = "day7_lab",
        user: str = "day7",
        password: str = "day7_local_only",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.model = SentenceTransformer(model_name)
        # Test connection
        conn = self._get_connection()
        conn.close()

    def _get_connection(self):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password
        )
        register_vector(conn)
        return conn

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Query pgvector for cosine similarity top_k chunks."""
        q_vec = self.model.encode(query, normalize_embeddings=True)
        conn = self._get_connection()
        cur = conn.cursor()

        where_clauses = []
        params: list[Any] = [q_vec]

        if filters:
            for k, v in filters.items():
                if v is not None:
                    where_clauses.append(f"{k} = %s")
                    params.append(v)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        
        # Order by cosine distance ascending
        sql = f"""
            SELECT id, text, source_id, source, page, section, chunk_type,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        exec_params = [q_vec] + (params[1:] if len(params) > 1 else []) + [q_vec, top_k]
        cur.execute(sql, exec_params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        hits = []
        for r in rows:
            hits.append({
                "id": r[0],
                "text": r[1],
                "source_id": r[2],
                "source": r[3],
                "page": r[4],
                "metadata": {"section": r[5], "chunk_type": r[6]},
                "score": round(float(r[7]), 4),
                "strategy": "dense_pgvector"
            })
        return hits


class LexicalRetriever:
    """Lexical TF-IDF / BM25 retriever."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = {c["id"]: c for c in chunks}
        self.chunk_list = chunks
        counts = [Counter(tokenize(c["text"])) for c in chunks]
        df = Counter(term for row in counts for term in row)
        self.idf = {
            term: math.log((1 + len(chunks)) / (1 + count)) + 1
            for term, count in df.items()
        }
        self.vectors = [self._vector(row) for row in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vec = {term: count * self.idf[term] for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3
    ) -> list[dict[str, Any]]:
        q_vec = self._vector(Counter(tokenize(query)))
        scored = []
        for chunk, doc_vec in zip(self.chunk_list, self.vectors):
            if filters:
                match = True
                meta = chunk.get("metadata", {})
                for k, v in filters.items():
                    if meta.get(k) != v and chunk.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = sum(val * doc_vec.get(term, 0.0) for term, val in q_vec.items())
            if score > 0:
                scored.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source_id": chunk["source_id"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "metadata": chunk.get("metadata", {}),
                    "score": round(score, 4),
                    "strategy": "lexical_tfidf"
                })
        scored.sort(key=lambda x: (-x["score"], x["id"]))
        return scored[:top_k]


class HybridRetriever:
    """Hybrid Retriever combining Dense (pgvector) and Lexical with Weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: PgVectorRetriever,
        lexical_retriever: LexicalRetriever,
        rrf_k: int = 60,
        dense_weight: float = 0.4,
        lexical_weight: float = 0.6
    ):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
        candidate_k: int = 15
    ) -> list[dict[str, Any]]:
        # 1. Fetch top candidates from both retrievers
        dense_hits = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_hits = self.lexical.search(query, filters=filters, top_k=candidate_k)

        # 2. Compute Weighted RRF Scores
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["id"]
            chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.dense_weight * (1.0 / (self.rrf_k + rank))

        for rank, hit in enumerate(lexical_hits, start=1):
            cid = hit["id"]
            if cid not in chunk_map:
                chunk_map[cid] = hit
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.lexical_weight * (1.0 / (self.rrf_k + rank))

        # 3. Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: (-rrf_scores[x], x))
        
        final_results = []
        for cid in sorted_ids[:top_k]:
            item = dict(chunk_map[cid])
            item["score"] = round(rrf_scores[cid], 5)
            item["strategy"] = "hybrid_weighted_rrf"
            final_results.append(item)
        return final_results

