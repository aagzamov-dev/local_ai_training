import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")

SYSTEM = """You convert e-commerce search text into a strict JSON object.
Return ONLY valid JSON. No markdown. No extra keys.

Schema:
{
  "query": "",
  "filters": {
    "brand": [],
    "categories": [],
    "price": {"min": null, "max": null, "currency": "EUR"},
    "in_stock": null,
    "supplier": [],
    "sku": null,
    "ean": null
  },
  "sort": {"by": "relevance", "order": "desc"},
  "page": 1,
  "page_size": 24,
  "confidence": 0.0
}
"""

def build_messages(user_text: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
    ]

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    user_text = "sony WH-1000XM5 black under 350 eur in stock"
    messages = build_messages(user_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()
