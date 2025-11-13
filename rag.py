from __future__ import annotations
import duckdb
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_schema_docs(conn: duckdb.DuckDBPyConnection) -> List[str]:
    docs: List[str] = []
    tables = [r[0] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
    ).fetchall()]
    for t in tables:
        cols = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='main' AND table_name=?
            ORDER BY ordinal_position
            """,
            [t],
        ).fetchall()
        col_str = ", ".join([f"{c}:{d}" for c, d in cols])
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            cnt = None
        docs.append(f"TABLE {t} ({cnt} rows): {col_str}")
        # small sample per table for semantic hints
        try:
            sample = conn.execute(f"SELECT * FROM {t} LIMIT 3").df()
            docs.append(f"SAMPLE {t}:\n" + sample.to_csv(index=False))
        except Exception:
            pass
    return docs


def build_rag_index(texts: List[str]) -> FAISS:
    # Local lightweight embeddings to avoid API usage
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embed)


def query_index(index: FAISS, query: str, k: int = 5) -> List[Tuple[str, float]]:
    docs = index.similarity_search_with_score(query, k=k)
    return [(d.page_content, float(s)) for d, s in docs]
