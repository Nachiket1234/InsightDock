import os
import re
import time
from typing import Any, Dict
from app_secrets import load_tokens_from_file


def _ensure_gemini_key() -> None:
    load_tokens_from_file()
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set. Place it in Token.txt or env.")


def _candidate_models() -> list[str]:
    env_model = os.getenv("GEMINI_MODEL")
    # If user pins a model, only try that preference to avoid picking deprecated aliases.
    if env_model:
        return [env_model]
    candidates = [
        # Use the latest flash model as requested
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-pro-002",
    ]
    return [m for m in candidates if m]


def _list_models():
    _ensure_gemini_key()
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    try:
        return list(genai.list_models())
    except Exception:
        return []


def _supports_generate_content(model_obj) -> bool:
    methods = getattr(model_obj, "supported_generation_methods", None)
    if isinstance(methods, (list, tuple, set)):
        return any(str(m).lower() in ("generatecontent", "generate_content") for m in methods)
    return True  # be permissive if not provided


def _resolve_model_name(preferences: list[str]) -> str | None:
    models = _list_models()
    names = [getattr(m, "name", "") for m in models]
    lower_map = {n.lower(): n for n in names if n}

    # Exact match first (case-insensitive)
    for pref in preferences:
        key = pref.lower()
        if key in lower_map and _supports_generate_content(next(m for m in models if getattr(m, "name", "").lower() == key)):
            return lower_map[key]

    # Suffix match (allow shorthand without "models/")
    for pref in preferences:
        short = pref.split("/")[-1].lower()
        for m in models:
            n = getattr(m, "name", "")
            if n and n.lower().endswith(short) and _supports_generate_content(m):
                return n

    # Contains match (e.g., "2.5" and "pro")
    for pref in preferences:
        parts = [p for p in pref.lower().replace("models/", "").split("-") if p]
        for m in models:
            n = getattr(m, "name", "").lower()
            if n and all(p in n for p in parts) and _supports_generate_content(m):
                return getattr(m, "name", None)

    # Last resort: first model that supports generateContent
    for m in models:
        if _supports_generate_content(m):
            return getattr(m, "name", None)
    return None


def _generate_text(prompt: str) -> str:
    _ensure_gemini_key()
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    last_err: Exception | None = None
    
    # Get the model to use
    model_name = _resolve_model_name(_candidate_models())
    if not model_name:
        raise RuntimeError("No suitable Gemini model found.")
    
    model = genai.GenerativeModel(model_name)
    
    # Try resolved name from live model list first
    # Respect explicit env pin strictly; ignore cached model if it doesn't match pin
    env_pin = os.getenv("GEMINI_MODEL")
    cached = os.getenv("GEMINI_RESOLVED_MODEL")
    if env_pin and cached and cached.lower().split("/")[-1] != env_pin.lower().split("/")[-1]:
        os.environ.pop("GEMINI_RESOLVED_MODEL", None)
    overall_timeout = int(os.getenv("GEMINI_TIMEOUT", "45"))
    start = time.time()
    attempts = 0
    while attempts < 2:
        attempts += 1
        if time.time() - start > overall_timeout:
            raise TimeoutError("LLM request timed out. Please try again or switch model via GEMINI_MODEL.")
        try:
            resp = model.generate_content(prompt)
            return (resp.text or "").strip()
        except Exception as e:
            msg = str(e)
            last_err = e
            # 429 handling with backoff
            if "429" in msg or "rate limit" in msg.lower() or "retry" in msg.lower():
                # Try to parse suggested retry seconds
                m = re.search(r"retry.*?(\d+)(?:\.(\d+))?s", msg.lower())
                wait = 3 * attempts
                if m:
                    seconds = int(m.group(1))
                    wait = max(3, min(seconds, 15))
                # Do not exceed overall timeout
                remaining = overall_timeout - (time.time() - start)
                if remaining <= 0:
                    raise TimeoutError("LLM request timed out while waiting due to rate limits.")
                time.sleep(min(wait, max(1, int(remaining))))
                continue
            # Try next model for 404 or unsupported errors
            if "404" in msg or "not found" in msg or "not supported" in msg:
                break
            # Otherwise break fast
            break
    if last_err:
        raise last_err
    raise RuntimeError("No Gemini models available to try.")


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


def generate_sql(question: str, schema: str, conversation_hint: str = "") -> str:
    prompt = build_sql_prompt(question, schema, conversation_hint)
    txt = _generate_text(prompt)
    # Extract code block if present
    if "```" in txt:
        parts = txt.split("```")
        if len(parts) >= 2:
            body = parts[1]
            if body.strip().startswith("sql"):
                body = "\n".join(body.splitlines()[1:])
            return body.strip()
    return txt.strip()


def analyze_and_chart_advice(question: str, sql: str, preview: str) -> Dict[str, Any]:
    prompt = (
        "You will get a user question, the SQL used, and a text preview of results (first few rows)."
        " Provide: a 1-2 sentence insight summary; and a recommended chart type among [bar, line, scatter, pie, none]"
        f"\nQuestion: {question}\nSQL: {sql}\nPreview:\n{preview}\n"
    )
    text = _generate_text(prompt)
    return {"analysis": text}
