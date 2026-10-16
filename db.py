import re
import sys
import psycopg
from langchain_postgres import PGVector
import config
import models

vector_store = None


def run(sql, params=None):
    with psycopg.connect(config.DATABASE_URL) as conn:
        cur = conn.execute(sql, params)
        return cur.fetchall() if cur.description else []


def get_vector_store():
    # LangChain PGVector keeps the chunks and their vectors
    global vector_store
    if vector_store is None:
        vector_store = PGVector(
            embeddings=models.get_embeddings(),
            collection_name=config.COLLECTION,
            connection=config.LANGCHAIN_URL,
            embedding_length=config.EMBED_DIM,
            use_jsonb=True,
        )
    return vector_store


def create_tables(reset=False):
    global vector_store
    if reset:
        vector_store = None
        for table in ("langchain_pg_embedding", "langchain_pg_collection", "pages", "assets"):
            run(f"DROP TABLE IF EXISTS {table} CASCADE")

    run("CREATE EXTENSION IF NOT EXISTS vector")

    # pages: page text with placeholders like [[FIG:key]]
    run("""CREATE TABLE IF NOT EXISTS pages (
               pdf_name TEXT NOT NULL, page_no INT NOT NULL, page_text TEXT NOT NULL,
               PRIMARY KEY (pdf_name, page_no))""")

    # assets: one row for every figure and table
    run("""CREATE TABLE IF NOT EXISTS assets (
               asset_key TEXT PRIMARY KEY, pdf_name TEXT NOT NULL, page_no INT NOT NULL,
               kind TEXT NOT NULL, caption TEXT, summary TEXT, table_md TEXT, image_path TEXT)""")

    # this call makes LangChain create its own tables
    get_vector_store()
    run("""CREATE INDEX IF NOT EXISTS chunks_fts_idx ON langchain_pg_embedding
           USING gin (to_tsvector('english', document))""")

    print("tables are ready")


def delete_pdf(pdf_name):
    rows = run("SELECT id FROM langchain_pg_embedding WHERE cmetadata->>'pdf_name' = %s", (pdf_name,))
    if rows:
        get_vector_store().delete(ids=[str(row[0]) for row in rows])
    run("DELETE FROM pages WHERE pdf_name = %s", (pdf_name,))
    run("DELETE FROM assets WHERE pdf_name = %s", (pdf_name,))


def save_page(pdf_name, page_no, page_text):
    page_text = page_text.replace("\x00", "")
    run("""INSERT INTO pages (pdf_name, page_no, page_text) VALUES (%s, %s, %s)
           ON CONFLICT (pdf_name, page_no)
           DO UPDATE SET page_text = EXCLUDED.page_text""",
        (pdf_name, page_no, page_text))


def save_asset(row):
    row = tuple(value.replace("\x00", "") if isinstance(value, str) else value for value in row)
    run("""INSERT INTO assets
           (asset_key, pdf_name, page_no, kind, caption, summary, table_md, image_path)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (asset_key) DO UPDATE SET
           pdf_name = EXCLUDED.pdf_name,
           page_no = EXCLUDED.page_no,
           kind = EXCLUDED.kind,
           caption = EXCLUDED.caption,
           summary = EXCLUDED.summary,
           table_md = EXCLUDED.table_md,
           image_path = EXCLUDED.image_path""", row)


def get_page_text(pdf_name, page_no):
    rows = run("SELECT page_text FROM pages WHERE pdf_name = %s AND page_no = %s", (pdf_name, page_no))
    return rows[0][0] if rows else None


def get_asset(asset_key):
    rows = run("SELECT kind, caption, summary, table_md, image_path FROM assets WHERE asset_key = %s", (asset_key,))
    if not rows:
        return None
    kind, caption, summary, table_md, image_path = rows[0]
    return {"kind": kind, "caption": caption, "summary": summary, "table_md": table_md, "image_path": image_path}


def list_pdfs():
    return run("SELECT pdf_name, COUNT(*) FROM pages GROUP BY pdf_name ORDER BY pdf_name")


def keyword_search(question, k, pdf_name=None):
    # any word of the question can match, more matches rank higher
    words = re.findall(r"[A-Za-z0-9]+", question)
    if not words or k <= 0:
        return []
    query = " | ".join(words)
    return run("""SELECT id, document, cmetadata FROM langchain_pg_embedding
                  WHERE to_tsvector('english', document) @@ to_tsquery('english', %s)
                    AND (%s::text IS NULL OR cmetadata->>'pdf_name' = %s)
                  ORDER BY ts_rank(to_tsvector('english', document), to_tsquery('english', %s)) DESC
                  LIMIT %s""", (query, pdf_name, pdf_name, query, k))


if __name__ == "__main__":
    create_tables(reset="--reset" in sys.argv)