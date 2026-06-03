from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector


SCHEMA = Path(__file__).parent / "schema.sql"


def connect(database_url: str) -> psycopg.Connection:
    conn = psycopg.connect(database_url, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text())
        conn.commit()
        register_vector(conn)
        return conn
    except Exception:
        conn.close()
        raise
