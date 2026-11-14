"""Ingest 230 chunks into PostgreSQL pgvector container."""
import json
import time
from pathlib import Path
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

def main():
    root_dir = Path(__file__).resolve().parent.parent
    chunks_path = root_dir / "data" / "chunks.json"
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        host="127.0.0.1",
        port=55433,
        dbname="day7_lab",
        user="day7",
        password="day7_local_only"
    )
    register_vector(conn)
    cur = conn.cursor()

    # Create schema
    print("Creating table schema in PostgreSQL...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id VARCHAR(255) PRIMARY KEY,
            text TEXT NOT NULL,
            source_id VARCHAR(255) NOT NULL,
            source VARCHAR(255) NOT NULL,
            page INTEGER NOT NULL,
            section VARCHAR(100),
            chunk_type VARCHAR(100),
            embedding vector(384) NOT NULL
        );
    """)
    conn.commit()

    # Load model and embed
    print("Initializing embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    
    t0 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    embed_time = time.time() - t0
    print(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")

    # Insert into database
    print("Inserting chunks into PostgreSQL...")
    insert_sql = """
        INSERT INTO chunks (id, text, source_id, source, page, section, chunk_type, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            text = EXCLUDED.text,
            source_id = EXCLUDED.source_id,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            chunk_type = EXCLUDED.chunk_type,
            embedding = EXCLUDED.embedding;
    """
    for c, emb in zip(chunks, embeddings):
        meta = c.get("metadata", {})
        cur.execute(insert_sql, (
            c["id"],
            c["text"],
            c["source_id"],
            c["source"],
            c["page"],
            meta.get("section", "general"),
            meta.get("chunk_type", "paragraph"),
            emb
        ))
    conn.commit()

    # Create HNSW index
    print("Creating HNSW cosine index...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw 
        ON chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()

    # Verify count
    cur.execute("SELECT count(*) FROM chunks;")
    total = cur.fetchone()[0]
    print(f"Successfully ingested {total} chunks into PostgreSQL with pgvector HNSW index!")

    # Test sample search
    test_query = "ngày thành lập trường bách khoa"
    q_vec = model.encode(test_query, normalize_embeddings=True)
    cur.execute("""
        SELECT id, page, text, 1 - (embedding <=> %s) AS similarity
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT 3;
    """, (q_vec, q_vec))
    results = cur.fetchall()
    print("\nSample Test Search for:", test_query)
    for r in results:
        print(f"  [Sim: {r[3]:.4f}] Page {r[1]} - {r[0]}: {r[2][:80]}...")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()

