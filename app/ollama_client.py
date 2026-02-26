import os
import httpx
from typing import Any, Dict, List, Optional

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
DEFAULT_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "-1")

class OllamaError(Exception):
    pass

def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    keep_alive: Optional[str] = None,
    timeout_s: int = 600,
) -> str:
    payload: Dict[str, Any] = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "stream": False,
    }

    ka = keep_alive if keep_alive is not None else DEFAULT_KEEP_ALIVE
    if ka is not None:
        payload["keep_alive"] = ka

    if temperature is not None:
        payload.setdefault("options", {})["temperature"] = float(temperature)

    url = f"{OLLAMA_BASE_URL}/api/chat"
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise OllamaError(str(e))

    msg = data.get("message", {})
    content = msg.get("content")
    if not content:
        raise OllamaError(f"Unexpected Ollama response: {data}")
    return content
