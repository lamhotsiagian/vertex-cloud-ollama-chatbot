import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.safe_calc import safe_calculate, CalcError
from app.ollama_client import chat as ollama_chat, OllamaError

app = FastAPI(title="Vertex Ollama (Mistral) Chatbot", version="1.0.0")

READY_FILE = os.getenv("READY_FILE", "/tmp/ollama_ready")
MODEL_NAME_DEFAULT = os.getenv("OLLAMA_MODEL", "mistral")

# Conservative math detector: numbers, operators, parentheses, whitespace, decimal point, caret.
_MATH_RE = re.compile(r"^[0-9\.\+\-\*\/\%\(\)\s\^]+$")

def is_math_expression(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if not _MATH_RE.match(t):
        return False
    return any(ch in t for ch in "+-*/%^()")

def normalize_math(expr: str) -> str:
    return expr.replace("^", "**")

class VertexPredictRequest(BaseModel):
    instances: List[Any]
    parameters: Optional[Dict[str, Any]] = None

@app.get("/health")
def health():
    # Vertex health checks expect 200 OK when ready.
    if os.path.exists(READY_FILE):
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Not ready")

@app.post("/predict")
def predict(req: VertexPredictRequest):
    params = req.parameters or {}
    model = params.get("model", MODEL_NAME_DEFAULT)
    temperature = params.get("temperature")
    keep_alive = params.get("keep_alive")

    outputs: List[Dict[str, Any]] = []

    for inst in req.instances:
        prompt, messages = _extract_prompt_or_messages(inst)

        # Calculator tool
        if prompt is not None and is_math_expression(prompt):
            expr = normalize_math(prompt)
            try:
                value = safe_calculate(expr)
                outputs.append({"type": "calculator", "input": prompt, "result": value})
            except CalcError as e:
                outputs.append({"type": "calculator", "input": prompt, "error": str(e)})
            continue

        # LLM
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        try:
            answer = ollama_chat(
                messages=messages,
                model=model,
                temperature=temperature,
                keep_alive=keep_alive,
            )
            outputs.append({"type": "llm", "model": model, "response": answer})
        except OllamaError as e:
            outputs.append({"type": "llm", "model": model, "error": str(e)})

    return {"predictions": outputs}

def _extract_prompt_or_messages(inst: Any) -> Tuple[Optional[str], Optional[List[Dict[str, str]]]]:
    # Supported instance types:
    # - "hello"
    # - {"prompt": "hello"}
    # - {"messages": [{"role":"user","content":"hello"}]}
    if isinstance(inst, str):
        return inst, None
    if isinstance(inst, dict):
        if "messages" in inst:
            msgs = inst["messages"]
            if not isinstance(msgs, list):
                raise HTTPException(status_code=400, detail="messages must be a list")
            return None, msgs
        if "prompt" in inst:
            return str(inst["prompt"]), None
    raise HTTPException(status_code=400, detail=f"Unsupported instance format: {type(inst)}")
