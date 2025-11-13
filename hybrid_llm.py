import os
import re
import time
from typing import Any, Dict
from app_secrets import load_tokens_from_file


def _ensure_api_keys() -> None:
    load_tokens_from_file()
    if not any([os.getenv("GEMINI_API_KEY"), os.getenv("DEEPSEEK_API_KEY"), 
                os.getenv("OPENAI_API_KEY"), os.getenv("OPENROUTER_API_KEY")]):
        raise RuntimeError("At least one API key must be set in Token.txt")


def _generate_text_gemini(prompt: str) -> str:
    """Generate text using Gemini models."""
    import google.generativeai as genai
    
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # Use configured model or fallback
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    
    overall_timeout = int(os.getenv("GEMINI_TIMEOUT", "45"))
    start = time.time()
    attempts = 0
    
    while attempts < 2:
        attempts += 1
        if time.time() - start > overall_timeout:
            raise TimeoutError("Gemini request timed out.")
        
        try:
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            # Handle rate limits with backoff
            if "429" in msg or "rate limit" in msg.lower():
                wait = 3 * attempts
                remaining = overall_timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError("Gemini request timed out due to rate limits.")
                time.sleep(min(wait, max(1, int(remaining))))
                continue
            raise e
    
    raise RuntimeError("Gemini generation failed after retries.")


def _generate_text_openai(prompt: str) -> str:
    """Generate text using OpenAI models."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    overall_timeout = int(os.getenv("OPENAI_TIMEOUT", "45"))
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1,
            timeout=overall_timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI generation failed: {str(e)}")


def _generate_text_openrouter(prompt: str) -> str:
    """Generate text using OpenRouter models."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )
    
    model_name = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
    overall_timeout = int(os.getenv("OPENROUTER_TIMEOUT", "45"))
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1,
            timeout=overall_timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenRouter generation failed: {str(e)}")


def _generate_text_deepseek(prompt: str) -> str:
    """Generate text using DeepSeek models."""
    from openai import OpenAI
    
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com"
    )
    
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    overall_timeout = int(os.getenv("DEEPSEEK_TIMEOUT", "45"))
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1,
            timeout=overall_timeout
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"DeepSeek generation failed: {str(e)}")



def generate_text(prompt: str, provider: str = "auto") -> str:
    """
    Generate text using the specified provider.
    
    Args:
        prompt: The input prompt
        provider: "gemini", "deepseek", "openai", "openrouter", or "auto"
    """
    _ensure_api_keys()
    
    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not found")
        return _generate_text_gemini(prompt)
    
    elif provider == "deepseek":
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY not found")
        return _generate_text_deepseek(prompt)
    
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found")
        return _generate_text_openai(prompt)
    
    elif provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY not found")
        return _generate_text_openrouter(prompt)
    
    elif provider == "auto":
        # Try providers in order of preference
        providers_to_try = []
        if os.getenv("GEMINI_API_KEY"):
            providers_to_try.append("gemini")
        if os.getenv("OPENAI_API_KEY"):
            providers_to_try.append("openai")
        if os.getenv("OPENROUTER_API_KEY"):
            providers_to_try.append("openrouter")
        if os.getenv("DEEPSEEK_API_KEY"):
            providers_to_try.append("deepseek")
        
        last_error = None
        for prov in providers_to_try:
            try:
                return generate_text(prompt, prov)
            except Exception as e:
                last_error = e
                continue
        
        if last_error:
            raise last_error
        raise RuntimeError("No valid API keys found")
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_available_providers() -> list[str]:
    """Get list of available LLM providers based on API keys."""
    _ensure_api_keys()
    providers = []
    
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append("openrouter")
    
    if os.getenv("DEEPSEEK_API_KEY"):
        providers.append("deepseek")
    
    return providers


# SQL Generation functions
SYSTEM_PROMPT_SQL = (
    "You are a data analyst that writes precise DuckDB SQL for an e-commerce dataset."
    " Use only available tables and columns."
    " Return only the SQL in a fenced block."
)


def build_sql_prompt(question: str, schema: str, conversation_hint: str = "") -> str:
    return (
        f"{SYSTEM_PROMPT_SQL}\n\n"
        f"Schema:\n{schema}\n\n"
        f"Context: {conversation_hint}\n\n"
        f"Question: {question}\n"
        "Write a single valid SQL query for DuckDB that answers the question."
    )


def generate_sql(question: str, schema: str, conversation_hint: str = "", provider: str = "auto") -> str:
    prompt = build_sql_prompt(question, schema, conversation_hint)
    txt = generate_text(prompt, provider)
    
    # Extract code block if present
    if "```" in txt:
        parts = txt.split("```")
        for part in parts:
            if "SELECT" in part.upper() or "WITH" in part.upper():
                # Clean up the SQL
                sql = part.strip()
                if sql.startswith("sql\n"):
                    sql = sql[4:]
                return sql.strip()
    
    return txt.strip()


def generate_analysis(question: str, df_head: str, provider: str = "auto") -> str:
    prompt = (
        f"Analyze this query result and provide business insights:\n\n"
        f"Question: {question}\n\n"
        f"Data sample:\n{df_head}\n\n"
        f"Provide a concise analysis with key findings and business recommendations."
    )
    return generate_text(prompt, provider)
