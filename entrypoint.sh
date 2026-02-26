    #!/usr/bin/env bash
    set -euo pipefail

    export AIP_HTTP_PORT="${AIP_HTTP_PORT:-8080}"
    export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
    export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

    export OLLAMA_BASE_MODEL="${OLLAMA_BASE_MODEL:-mistral}"
    export OLLAMA_MODEL="${OLLAMA_MODEL:-mistral}"

    export OLLAMA_ADAPTER_DIR="${OLLAMA_ADAPTER_DIR:-}"
    export OLLAMA_CUSTOM_MODEL="${OLLAMA_CUSTOM_MODEL:-mistral-vertex-bot}"

    export READY_FILE="${READY_FILE:-/tmp/ollama_ready}"

    echo "[entrypoint] Starting Ollama..."
    /bin/ollama serve >/tmp/ollama_serve.log 2>&1 &
    OLLAMA_PID=$!

    echo "[entrypoint] Waiting for Ollama API..."
    for i in {1..180}; do
      if curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
        break
      fi
      sleep 1
    done

    if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
      echo "[entrypoint] Ollama API failed to start."
      tail -n 200 /tmp/ollama_serve.log || true
      exit 1
    fi

    echo "[entrypoint] Pulling base model: ${OLLAMA_BASE_MODEL}"
    /bin/ollama pull "${OLLAMA_BASE_MODEL}"

    # If adapter exists, create custom model (Safetensors adapter directory)
    if [[ -n "${OLLAMA_ADAPTER_DIR}" ]] && [[ -f "${OLLAMA_ADAPTER_DIR}/adapter_model.safetensors" || -f "${OLLAMA_ADAPTER_DIR}/adapter.safetensors" ]]; then
      echo "[entrypoint] Found adapter dir: ${OLLAMA_ADAPTER_DIR}"
      echo "[entrypoint] Creating custom model: ${OLLAMA_CUSTOM_MODEL}"

      cat > /tmp/Modelfile <<EOF
FROM ${OLLAMA_BASE_MODEL}
ADAPTER ${OLLAMA_ADAPTER_DIR}
SYSTEM """
You are a concise assistant.
If the user asks for math, the server will handle it with a calculator tool.
"""
PARAMETER temperature 0.2
EOF

      /bin/ollama create "${OLLAMA_CUSTOM_MODEL}" -f /tmp/Modelfile
      export OLLAMA_MODEL="${OLLAMA_CUSTOM_MODEL}"
    fi

    # Preload and keep alive (optional)
    echo "[entrypoint] Preloading model: ${OLLAMA_MODEL}"
    curl -sf "${OLLAMA_BASE_URL}/api/chat"       -d "{"model":"${OLLAMA_MODEL}","messages":[],"keep_alive":-1,"stream":false}" >/dev/null || true

    touch "${READY_FILE}"
    echo "[entrypoint] Ready. Starting API on port ${AIP_HTTP_PORT}"

    exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${AIP_HTTP_PORT}" --workers 1
