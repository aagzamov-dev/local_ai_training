from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json

# --- CONFIGURATION ---
MAX_SEQ_LENGTH = 1024
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
DATASET_PATH = "data/training_data.jsonl"
OUTPUT_DIR = "qwen2.5-intent-lora"

def formatting_prompts_func(examples):
    """
    Maps the dataset to the standard ChatML format expected by Unsloth/Qwen train loop.
    The input 'training_data.jsonl' is already in {"messages": [...]} format.
    Unsloth's SFTTrainer with `dataset_text_field` often expects text, 
    but modern TRL supports 'messages' column directly if configured.
    
    However, mostly for Unsloth, using `to_sharegpt` or standard text generation is typical.
    Here we implement a robust conversion to a text field for training.
    """
    conversations = examples["messages"]
    texts = []
    
    tokenizer = FastLanguageModel.get_chat_template(
        None, chat_template = "qwen-2.5", mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"}
    )
    
    # Simple manual ChatML formatting if tokenizer template not reliable per version
    # <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
    
    for conv in conversations:
        # conv is list of {role, content}
        text = ""
        for turn in conv:
            role = turn["role"]
            content = turn["content"]
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        # Training on the 'assistant' response. Unsloth handles masking automatically 
        # if using `DataCollatorForCompletionOnlyLM` but simpler is just causal LM on full text 
        # or using unsloth's chat templates.
        # For simplicity and robustness in this script, we output the full text string.
        texts.append(text)
        
    return { "text" : texts }

def run_finetuning():
    print("Loading Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
    )

    print("Adding LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Optimized to 0
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("Loading Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("Starting Training...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can set True for speedup
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # ~1 epoch for 200 items (200 / (2*4) = 25 steps per epoch) -> 60 is ~2.5 epochs
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
        ),
    )

    trainer.train()

    print("Saving Adapters...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Saving GGUF (f16 and q4_k_m)...")
    try:
        model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "f16")
        model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")
        print(f"Done! GGUF models saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"GGUF Export failed (might need llama.cpp installed): {e}")

if __name__ == "__main__":
    run_finetuning()
