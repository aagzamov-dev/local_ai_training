# =============================================================================
# FINETUNE SCRIPT FOR INTENT PARSER
# =============================================================================
# Example run command (do not execute automatically):
# python finetune.py
# =============================================================================

import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model_name": "unsloth/Qwen2.5-3B-Instruct",
    "dataset_path": "data/training_data.jsonl",
    "output_dir": "qwen2.5-intent-lora",
    "max_seq_length": 1024,
    "load_in_4bit": True,
    "dtype": None,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 5,
    "max_steps": 60,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "logging_steps": 1,
    "seed": 3407,
    "export_gguf": True,
}


# =============================================================================
# DATASET VALIDATION
# =============================================================================


def validate_assistant_json(content: str) -> tuple[bool, str]:
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        return False, f"invalid JSON: {e}"

    required_top = {"query", "filters", "sort", "page", "page_size", "confidence"}
    if set(obj.keys()) != required_top:
        return False, f"top keys mismatch: {sorted(obj.keys())}"

    filters = obj.get("filters")
    if not isinstance(filters, dict):
        return False, "filters not a dict"

    required_filters = {"brand", "categories", "price", "in_stock", "sku", "ean"}
    if set(filters.keys()) != required_filters:
        return False, f"filters keys mismatch: {sorted(filters.keys())}"

    return True, ""


def validate_dataset(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_rows = []
    errors = 0

    for i, row in enumerate(data):
        messages = row.get("messages", [])

        if len(messages) != 3:
            print(f"[row {i}] expected 3 messages, got {len(messages)}")
            errors += 1
            continue

        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            print(f"[row {i}] roles must be system/user/assistant, got {roles}")
            errors += 1
            continue

        assistant_content = messages[2].get("content", "")
        valid, err = validate_assistant_json(assistant_content)
        if not valid:
            print(f"[row {i}] assistant JSON invalid: {err}")
            errors += 1
            continue

        valid_rows.append(row)

    print(f"Validated {len(valid_rows)}/{len(data)} rows. Errors: {errors}")
    return valid_rows


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# =============================================================================
# FORMATTING
# =============================================================================


def format_conversation(messages: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def create_dataset(rows: list[dict[str, Any]], tokenizer) -> Dataset:
    texts = []
    for row in rows:
        messages = row["messages"]
        text = format_conversation(messages, tokenizer)
        texts.append(text)

    return Dataset.from_dict({"text": texts})


# =============================================================================
# MAIN
# =============================================================================


def run_finetuning():
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=CONFIG["dtype"],
        load_in_4bit=CONFIG["load_in_4bit"],
    )

    print("=" * 60)
    print("ADDING LORA ADAPTERS")
    print("=" * 60)

    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=CONFIG["target_modules"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
        use_rslora=False,
        loftq_config=None,
    )

    print("=" * 60)
    print("LOADING AND VALIDATING DATASET")
    print("=" * 60)

    raw_data = load_jsonl(CONFIG["dataset_path"])
    valid_data = validate_dataset(raw_data)

    if len(valid_data) == 0:
        raise ValueError("No valid training rows found!")

    dataset = create_dataset(valid_data, tokenizer)
    print(f"Created dataset with {len(dataset)} examples")

    print("=" * 60)
    print("SETTING UP COMPLETION-ONLY TRAINING")
    print("=" * 60)

    response_template = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(
        response_template,
        add_special_tokens=False,
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )

    print(f"Response template: {repr(response_template)}")
    print(f"Response template IDs: {response_template_ids}")

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=False,
        data_collator=collator,
        args=TrainingArguments(
            per_device_train_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            warmup_steps=CONFIG["warmup_steps"],
            max_steps=CONFIG["max_steps"],
            learning_rate=CONFIG["learning_rate"],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=CONFIG["logging_steps"],
            optim="adamw_8bit",
            weight_decay=CONFIG["weight_decay"],
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
            seed=CONFIG["seed"],
            output_dir=CONFIG["output_dir"],
        ),
    )

    trainer.train()

    print("=" * 60)
    print("SAVING ADAPTERS")
    print("=" * 60)

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    if CONFIG["export_gguf"]:
        print("=" * 60)
        print("EXPORTING GGUF")
        print("=" * 60)

        try:
            model.save_pretrained_gguf(
                CONFIG["output_dir"],
                tokenizer,
                quantization_method="f16",
            )
            model.save_pretrained_gguf(
                CONFIG["output_dir"],
                tokenizer,
                quantization_method="q4_k_m",
            )
            print(f"GGUF models saved to {CONFIG['output_dir']}")
        except Exception as e:
            print(f"GGUF export failed (requires llama.cpp): {e}")

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    run_finetuning()
