#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${REPO:=vertex-llm}"
: "${IMAGE_NAME:=ollama-mistral-chatbot}"
: "${TAG:=v1}"

gcloud config set project "$PROJECT_ID"
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Create repo if missing (ignore errors)
gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" >/dev/null 2>&1 || true

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
echo "Building image: $IMAGE_URI"
gcloud builds submit . --tag "$IMAGE_URI"

echo "Uploading model to Vertex..."
gcloud ai models upload       --region="$REGION"       --display-name="$IMAGE_NAME"       --container-image-uri="$IMAGE_URI"       --container-ports=8080       --container-health-route="/health"       --container-predict-route="/predict"

echo "Done. Now create an endpoint and deploy the model (see README.md)."
