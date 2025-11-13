import os
from pathlib import Path
from typing import Optional
import duckdb
import pandas as pd


DATASET = "olistbr/brazilian-ecommerce"
DATA_DIR = Path("data/olist")


def _ensure_kaggle_config() -> None:
    kj = Path("kaggle.json")
    if kj.exists():
        os.environ.setdefault("KAGGLE_CONFIG_DIR", str(kj.parent.resolve()))
    # Avoid noisy Kaggle stdout prompts
    os.environ.setdefault("KAGGLE_KEEP_DOWNLOADS", "1")


def download_olist_dataset(force: bool = False) -> Path:
    from kaggle.api.kaggle_api_extended import KaggleApi

    _ensure_kaggle_config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    marker = DATA_DIR / ".download_complete"
    if marker.exists() and not force:
        return DATA_DIR

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(DATASET, path=str(DATA_DIR), unzip=True, quiet=False)
    marker.touch()
    return DATA_DIR


def load_into_duckdb(db_path: Optional[str] = None):
    conn = duckdb.connect(database=db_path or ":memory:")
    conn.execute("PRAGMA threads=4")

    csvs = {
        p.stem: p for p in DATA_DIR.glob("*.csv")
    }
    if not csvs:
        raise FileNotFoundError("CSV files not found. Run download_olist_dataset() first.")

    for name, path in csvs.items():
        # Prefer DuckDB's streaming CSV reader to avoid pandas OOM
        try:
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE {name} AS
                SELECT * FROM read_csv_auto('{str(path).replace("'","''")}',
                                             SAMPLE_SIZE=-1,
                                             IGNORE_ERRORS=TRUE);
                """
            )
            continue
        except Exception:
            pass

        # Fallback: chunked pandas read to avoid memory spikes
        first = True
        for chunk in pd.read_csv(
            path,
            chunksize=50_000,  # Reduced from 100_000 for faster processing
            low_memory=False,
            encoding_errors="ignore",
            on_bad_lines="skip",
        ):
            # Normalize column names to snake_case
            chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
            # Best-effort datetime parsing on chunk
            for col in list(chunk.columns):
                lc = col.lower()
                if any(k in lc for k in ["date", "timestamp", "datetime"]):
                    try:
                        chunk[col] = pd.to_datetime(chunk[col], errors="coerce")
                    except Exception:
                        pass
            conn.register("tmp_chunk", chunk)
            if first:
                conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_chunk")
                first = False
            else:
                conn.execute(f"INSERT INTO {name} SELECT * FROM tmp_chunk")
            conn.unregister("tmp_chunk")

    return conn


def describe_schema(conn) -> str:
    tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    schema_desc = []
    for (t,) in tables:
        cols = conn.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='main' AND table_name=?
            ORDER BY ordinal_position
            """,
            [t],
        ).fetchall()
        schema_desc.append(f"TABLE {t}: " + ", ".join([f"{c}:{d}" for c, d in cols]))
    return "\n".join(schema_desc)
