import os
from pathlib import Path


def load_tokens_from_file(file_path: str | os.PathLike = "Token.txt") -> None:
    p = Path(file_path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v and not os.getenv(k):
                os.environ[k] = v
    except Exception:
        pass
