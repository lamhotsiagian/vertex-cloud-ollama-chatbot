import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    return p.parse_args()

def main():
    args = parse_args()
    model_name = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Non-quantized LoRA adapter (recommended for Ollama safetensors adapter import)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    ds = load_dataset("json", data_files=args.train_jsonl, split="train")

    def to_text(ex):
        # Use model's chat template (important for Mistral Instruct)
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="adamw_torch",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=train_args,
    )
    trainer.train()

    # Save adapter (Safetensors)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved LoRA adapter to:", args.output_dir)

if __name__ == "__main__":
    main()
