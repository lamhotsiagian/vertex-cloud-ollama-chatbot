# Dataset format (recommended) — Messages JSONL

Use **one JSON object per line** (JSONL). Each record must contain a `messages` array.

## Required field
- `messages`: array of message objects, in order.

Each message object:
- `role`: `"system" | "user" | "assistant"`
- `content`: string

## Optional fields
- `metadata`: any JSON object (source, tags, difficulty, etc.)

## Example (one line)
```json
{"messages":[
  {"role":"system","content":"You are concise."},
  {"role":"user","content":"Explain what Vertex AI endpoints do."},
  {"role":"assistant","content":"They host models for online prediction behind managed endpoints."}
], "metadata":{"source":"internal","topic":"vertex"}}
```

## Multi-turn example (one line)
```json
{"messages":[
  {"role":"system","content":"You are concise."},
  {"role":"user","content":"What is RAG?"},
  {"role":"assistant","content":"RAG retrieves relevant context and conditions the answer on it."},
  {"role":"user","content":"Give one benefit."},
  {"role":"assistant","content":"It reduces hallucinations by grounding responses in retrieved text."}
]}
```

## Notes for Mistral-Instruct
Mistral instruct models are trained with special instruction tokens (e.g., `[INST] ... [/INST]`).
During training we **DO NOT** manually insert these tokens. We use the tokenizer's
`apply_chat_template(messages, ...)` to format the conversation correctly.
