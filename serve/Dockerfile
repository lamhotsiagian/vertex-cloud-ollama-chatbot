# Ollama official Docker image
FROM ollama/ollama:latest

# Install Python + curl for the wrapper API and readiness checks
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY serve/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY serve/app /app/app
COPY serve/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Optional: include adapter directory in the image (you can also mount at runtime)
# COPY serve/adapters /app/adapters

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_MODEL=mistral
ENV OLLAMA_MODEL=mistral
ENV OLLAMA_BASE_URL=http://127.0.0.1:11434
ENV READY_FILE=/tmp/ollama_ready

ENTRYPOINT ["/app/entrypoint.sh"]
