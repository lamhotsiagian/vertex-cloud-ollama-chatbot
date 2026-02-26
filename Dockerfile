FROM ollama/ollama:latest

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Optional: bake adapter into image
# COPY serve/adapters /app/adapters

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV OLLAMA_BASE_MODEL=mistral
ENV OLLAMA_MODEL=mistral
ENV OLLAMA_BASE_URL=http://127.0.0.1:11434
ENV READY_FILE=/tmp/ollama_ready

ENTRYPOINT ["/app/entrypoint.sh"]
