from typing import Tuple, Optional
import pandas as pd
import duckdb
from hybrid_llm import generate_sql, generate_analysis


def _preview_str(df: pd.DataFrame, n: int = 5) -> str:
    return df.head(n).to_csv(index=False)


def answer_question(question: str, conn: duckdb.DuckDBPyConnection, schema: str, hint: str = "", provider: str = "auto") -> Tuple[pd.DataFrame, str, str]:
    """
    Answer a natural language question using the hybrid LLM system.
    
    Args:
        question: Natural language question
        conn: DuckDB connection
        schema: Database schema description
        hint: Conversation context hint
        provider: LLM provider ("gemini", "deepseek", or "auto")
    """
    try:
        # Generate SQL using selected provider
        sql = generate_sql(question, schema, hint, provider)
        
        # Execute the query
        df = conn.execute(sql).df()
        
        # Generate analysis using the same provider
        df_sample = df.head(10).to_string() if not df.empty else "No results"
        analysis = generate_analysis(question, df_sample, provider)
        
        return df, sql, analysis
        
    except Exception as e:
        # Return empty dataframe and error info
        empty_df = pd.DataFrame({"Error": [str(e)]})
        error_sql = f"-- Error generating SQL for: {question}\n-- {str(e)}"
        error_analysis = f"Unable to process the question: {str(e)}"
        
        return empty_df, error_sql, error_analysis
