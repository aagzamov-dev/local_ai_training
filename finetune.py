import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments

try:
    from unsloth import FastLanguageModel
except Exception as e:
    raise RuntimeError(f"Missing dependency: unsloth\nError: {e}")

try:
    from trl import SFTTrainer
except Exception as e:
    raise RuntimeError(f"Missing dependency: trl\nError: {e}")


CONFIG: Dict[str, Any] = {
    "seed": 42,
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "dataset_path": "data/training_data.jsonl",
    "output_dir": "runs/finetuned_intent_parser",
    "max_seq_length": 4096,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 10,
    "max_steps": 200,
    "logging_steps": 5,
    "lr_scheduler_type": "cosine",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "export_gguf": False,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_dataset(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        msgs = row.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 3:
            continue
        if msgs[0].get("role") != "system":
            continue
        if msgs[1].get("role") != "user":
            continue
        if msgs[2].get("role") != "assistant":
            continue
        if not isinstance(msgs[1].get("content"), str):
            continue
        if not isinstance(msgs[2].get("content"), str):
            continue
        valid.append(row)
    return valid


def create_dataset(rows: List[Dict[str, Any]], tokenizer: Any) -> Dataset:
    texts: List[str] = []
    for row in rows:
        messages = row["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    response_template_ids: List[int]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        labels = input_ids.clone()

        tpl = self.response_template_ids
        tpl_len = len(tpl)

        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()

            start = -1
            max_j = len(seq) - tpl_len
            for j in range(max_j + 1):
                if seq[j : j + tpl_len] == tpl:
                    start = j + tpl_len
                    break

            if start == -1:
                labels[i].fill_(-100)
            else:
                labels[i, :start] = -100

        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        batch["labels"] = labels
        return batch


def run_finetuning() -> None:
    set_seed(CONFIG["seed"])

    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

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
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
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
            report_to=[],
        ),
    )

    trainer.train()

    print("=" * 60)
    print("SAVING ADAPTERS")
    print("=" * 60)

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    if CONFIG["export_gguf"]:
        print("=" * 60)
        print("EXPORTING GGUF")
        print("=" * 60)
        try:
            model.save_pretrained_gguf(CONFIG["output_dir"], tokenizer, quantization_method="f16")
            model.save_pretrained_gguf(CONFIG["output_dir"], tokenizer, quantization_method="q4_k_m")
            print(f"GGUF models saved to {CONFIG['output_dir']}")
        except Exception as e:
            print(f"GGUF export failed (requires llama.cpp): {e}")

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    run_finetuning()
