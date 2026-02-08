from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LORA_PATH = "runs/finetuned_intent_parser"
OUTPUT = "runs/merged_model"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype="auto",
    offload_folder="offload",
    offload_state_dict=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading LoRA...")
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    offload_folder="offload"
)

print("Merging...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)

print("DONE")
