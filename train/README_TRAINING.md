# Fine-tuning Mistral (LoRA) for Ollama + Vertex Serving

This folder provides:
- A **dataset format** (messages JSONL)
- Training scripts:
  - `train_lora_mistral.py` (recommended for Ollama import: non-quantized LoRA adapter)
  - `train_qlora_mistral.py` (optional for smaller GPUs; not ideal for direct Ollama adapter import)

## 1) Install training deps
Create a new Python env and install:
```bash
pip install -r requirements-train.txt
```

## 2) Put your dataset at `train_data/train.jsonl`
We include a tiny sample file you can start from.

## 3) Run LoRA training (recommended)
```bash
export HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
python train_lora_mistral.py \
  --train_jsonl train_data/train.jsonl \
  --output_dir outputs/mistral_lora \
  --epochs 1 \
  --lr 2e-4 \
  --max_seq_len 2048
```

Output: `outputs/mistral_lora/` containing `adapter_model.safetensors` + config.

## 4) Load adapter in Ollama (inside your serving container)
Ollama supports importing Safetensors adapters by pointing `ADAPTER` at the adapter directory.

For this repo (serving):
- Copy your adapter directory into `serve/adapters/my_lora/` (or mount it)
- Set:
  - `OLLAMA_ADAPTER_DIR=/app/adapters/my_lora`
  - `OLLAMA_BASE_MODEL=mistral`
  - `OLLAMA_CUSTOM_MODEL=mistral-vertex-bot`

Then rebuild + redeploy.

## Why avoid QLoRA adapters for Ollama import?
Ollama's docs recommend using **non-quantized (non-QLoRA) adapters** when importing Safetensors adapters for best compatibility.
If you must use QLoRA, consider merging to a full model or converting to GGUF adapter.
