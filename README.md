# Vertex AI Endpoint + Ollama (Mistral) Chatbot API (with Calculator Tool)

This repo deploys a **chatbot API** to **Google Vertex AI Online Prediction** using a **custom container**.
Inside the container:
- **Ollama** runs locally (localhost:11434) serving the **mistral** model.
- A **FastAPI** wrapper exposes Vertex-compatible endpoints:
  - `GET /health`
  - `POST /predict`
- A built-in **safe calculator tool** handles math expressions deterministically.

## What you get
- Vertex Prediction endpoint that accepts requests like:
  ```json
  { "instances": [ {"prompt": "Hello"}, {"prompt": "(12+8)*3"} ] }
  ```
- For math-like prompts -> calculator.
- Otherwise -> Ollama `/api/chat`.

---

## 0) Prereqs
- Google Cloud project with billing enabled
- gcloud CLI installed and authenticated:
  ```bash
  gcloud auth login
  gcloud auth application-default login
  ```

Enable required services:
```bash
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
```

---

## 1) Build & push container image (Artifact Registry)

Set variables:
```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export REPO="vertex-llm"
export IMAGE_NAME="ollama-mistral-chatbot"
export TAG="v1"
gcloud config set project "$PROJECT_ID"
```

Create repo:
```bash
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"
```

Build + push:
```bash
export IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
gcloud builds submit . --tag "$IMAGE_URI"
```

---

## 2) Upload to Vertex Model Registry
```bash
gcloud ai models upload   --region="$REGION"   --display-name="$IMAGE_NAME"   --container-image-uri="$IMAGE_URI"   --container-ports=8080   --container-health-route="/health"   --container-predict-route="/predict"
```

Get MODEL_ID:
```bash
gcloud ai models list --region="$REGION" --filter="displayName=$IMAGE_NAME"
```

---

## 3) Create endpoint + deploy model

Create endpoint:
```bash
gcloud ai endpoints create --region="$REGION" --display-name="${IMAGE_NAME}-endpoint"
```

Get ENDPOINT_ID:
```bash
gcloud ai endpoints list --region="$REGION" --filter="displayName=${IMAGE_NAME}-endpoint"
```

Deploy (recommended: **g2** machines with L4 GPU):
```bash
gcloud ai endpoints deploy-model "$ENDPOINT_ID"   --region="$REGION"   --model="$MODEL_ID"   --display-name="${IMAGE_NAME}-deployed"   --min-replica-count=1   --max-replica-count=1   --machine-type="g2-standard-8"   --traffic-split=0=100
```

> First deploy can take longer because the container pulls `mistral` inside Ollama before `/health` becomes 200.

---

## 4) Test predictions

Create `request.json`:
```json
{
  "instances": [
    { "prompt": "What is RAG in 1 sentence?" },
    { "prompt": "(12 + 8) * 3" },
    {
      "messages": [
        { "role": "system", "content": "Be very concise." },
        { "role": "user", "content": "Explain Vertex AI endpoints." }
      ]
    }
  ],
  "parameters": {
    "model": "mistral",
    "temperature": 0.2,
    "keep_alive": "-1"
  }
}
```

Call Vertex:
```bash
gcloud ai endpoints predict "$ENDPOINT_ID" --region="$REGION" --json-request="request.json"
```

---

## 5) Optional: Fine-tune Mistral with LoRA and load adapter in Ollama

See: `train/README_TRAINING.md` and `train/dataset_format.md`.

High level:
1) Create `train.jsonl` in the **messages JSONL** format.
2) Run `train/train_lora_mistral.py` to output a **Safetensors LoRA adapter directory**.
3) Copy adapter directory into `serve/adapters/my_lora/` (or mount it at runtime).
4) Set `OLLAMA_ADAPTER_DIR=/app/adapters/my_lora` and `OLLAMA_CUSTOM_MODEL=mistral-vertex-bot`.
5) Rebuild + redeploy.

---

## Environment variables (serving)
- `OLLAMA_BASE_MODEL` (default: `mistral`)
- `OLLAMA_MODEL` (default: `mistral`) — model used for inference (overridden to custom if adapter exists)
- `OLLAMA_ADAPTER_DIR` (optional) — directory containing `adapter_model.safetensors`, etc.
- `OLLAMA_CUSTOM_MODEL` (default: `mistral-vertex-bot`) — name created via `ollama create`
- `AIP_HTTP_PORT` (default: `8080`) — Vertex sets this; FastAPI must listen on it

---

## Repo layout
- `app/`        Vertex-compatible FastAPI server + calculator tool
- `serve/`      Dockerfile + entrypoint that starts Ollama and the API
- `train/`      Dataset format + LoRA training scripts for Mistral
# vertex-cloud-ollama-chatbot
