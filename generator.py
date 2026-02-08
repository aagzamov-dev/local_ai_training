import json
import random
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# --- CONFIGURATION ---

BRANDS = [
    "Apple", "Samsung", "Sony", "Bosch", "LG", "Dell", "HP", "Lenovo", "Asus",
    "Philips", "Siemens", "Miele", "Dyson", "Logitech", "Razer", "Nikon", "Canon",
    "Microsoft", "Nintendo", "Acer"
]

CATEGORIES = [
    "Smartphones", "Laptops", "Tablets", "Televisions", "Headphones", "Smart Watches",
    "Washing Machines", "Refrigerators", "Vacuum Cleaners", "Coffee Machines",
    "Gaming Consoles", "Monitors", "Cameras", "Keyboards", "Mice"
]

RESIDUAL_KEYWORDS = [
    "gaming", "wireless", "bluetooth", "noise cancelling", "black", "white", "silver", "gold",
    "portable", "4k", "oled", "qled", "smart", "android", "ios", "windows", "mac",
    "pro", "mini", "max", "ultra", "slim", "lightweight", "mechanical", "ergonomic"
]

PRICE_PHRASES = {
    "min": ["over", "above", "more than", "higher than", "minimum", "at least"],
    "max": ["under", "below", "cheaper than", "less than", "maximum", "at most"],
    "range": ["between", "from"]
}

STOCK_PHRASES = ["in stock", "available now", "ready to ship", "available"]

SORT_PHRASES = {
    "cheapest": {"by": "price", "order": "asc"},
    "lowest price": {"by": "price", "order": "asc"},
    "budget": {"by": "price", "order": "asc"},
    "premium": {"by": "price", "order": "desc"},
    "high end": {"by": "price", "order": "desc"},
    "most expensive": {"by": "price", "order": "desc"},
    "top tier": {"by": "price", "order": "desc"}
}

# --- DATA STRUCTURE (IMPROVED FOR FINETUNING) ---

@dataclass
class SearchIntent:
    brands: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    residuals: List[str] = field(default_factory=list)
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    in_stock: bool = False
    sku: Optional[str] = None
    ean: Optional[str] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    query_string: str = ""
    explanation: str = ""
    confidence: float = 0.0

# --- GENERATOR LOGIC ---

def generate_sku() -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))

def generate_ean() -> str:
    return "".join(random.choices("0123456789", k=13))

def construct_intent() -> SearchIntent:
    intent = SearchIntent()

    intent_type = random.choices(
        ["brand_cat", "brand_cat_res", "cat_res", "brand_only", "sku", "ean"],
        weights=[30, 30, 20, 10, 5, 5]
    )[0]

    if intent_type == "sku":
        intent.sku = generate_sku()
        intent.confidence = 0.98
        return _finalize_intent(intent, f"sku {intent.sku}")

    if intent_type == "ean":
        intent.ean = generate_ean()
        intent.confidence = 0.98
        return _finalize_intent(intent, f"ean {intent.ean}")

    if "brand" in intent_type:
        intent.brands = [random.choice(BRANDS)]

    if "cat" in intent_type:
        intent.categories = [random.choice(CATEGORIES)]

    if "res" in intent_type:
        intent.residuals = random.sample(RESIDUAL_KEYWORDS, k=random.randint(1, 2))

    has_price = random.random() < 0.3
    has_stock = random.random() < 0.2
    has_sort = random.random() < 0.2

    sort_phrase_used = ""
    if has_sort:
        phrase = random.choice(list(SORT_PHRASES.keys()))
        intent.sort_by = SORT_PHRASES[phrase]["by"]
        intent.sort_order = SORT_PHRASES[phrase]["order"]
        sort_phrase_used = phrase

    price_phrase_used = ""
    if has_price:
        p_type = random.choice(["min", "max", "range"])
        amount = random.randint(5, 50) * 10

        if p_type == "max":
            intent.price_max = amount
            price_phrase_used = f"{random.choice(PRICE_PHRASES['max'])} {amount} eur"

        elif p_type == "min":
            intent.price_min = amount
            price_phrase_used = f"{random.choice(PRICE_PHRASES['min'])} {amount} eur"

        else:
            amount2 = amount + random.randint(10, 100) * 10
            intent.price_min = amount
            intent.price_max = amount2
            price_phrase_used = f"between {amount} and {amount2} euros"

    stock_phrase_used = ""
    if has_stock:
        intent.in_stock = True
        stock_phrase_used = random.choice(STOCK_PHRASES)

    query_parts = []

    if sort_phrase_used:
        query_parts.append(sort_phrase_used)

    res_str = " ".join(intent.residuals)

    if res_str and random.choice([True, False]):
        query_parts.append(res_str)

    if intent.brands:
        query_parts.append(intent.brands[0])

    if intent.categories:
        query_parts.append(intent.categories[0])
    elif intent.brands and not intent.residuals and random.random() < 0.3:
        query_parts.append("products")

    if res_str and res_str not in " ".join(query_parts):
        query_parts.append(res_str)

    if price_phrase_used:
        query_parts.append(price_phrase_used)

    if stock_phrase_used:
        query_parts.append(stock_phrase_used)

    final_query_str = " ".join(query_parts)

    score = 0.70
    if intent.brands: score += 0.10
    if intent.categories: score += 0.10
    if intent.residuals: score += 0.05
    if intent.price_min or intent.price_max: score += 0.05
    if intent.in_stock: score += 0.05

    if score > 0.99:
        score = 0.98

    intent.confidence = round(score, 2)

    return _finalize_intent(intent, final_query_str)

def _finalize_intent(intent: SearchIntent, query_str: str) -> SearchIntent:
    intent.query_string = query_str
    return intent

def intent_to_json(intent: SearchIntent) -> Dict[str, Any]:
    return {
        "query": " ".join(intent.residuals),
        "filters": {
            "brand": intent.brands,
            "categories": intent.categories,
            "price": {
                "min": intent.price_min,
                "max": intent.price_max,
                "currency": "EUR"
            },
            "in_stock": True if intent.in_stock else None,
            "sku": intent.sku,
            "ean": intent.ean
        },
        "sort": {
            "by": intent.sort_by,
            "order": intent.sort_order
        },
        "page": 1,
        "page_size": 24,
        "confidence": intent.confidence
    }

def generate_datasets():
    baseline_data = []

    while len(baseline_data) < 100:
        intent = construct_intent()
        output_json = intent_to_json(intent)

        baseline_data.append({
            "input_query": intent.query_string,
            "expected_output_json": output_json
        })

    with open("data/baseline_eval.json", "w") as f:
        json.dump(baseline_data, f, indent=2)

    training_data = []

    SYS_PROMPT = """You are a deterministic ecommerce search intent parser for a European electronics ecommerce platform. Convert user search queries into STRICT JSON for Meilisearch. Output ONLY valid JSON."""

    while len(training_data) < 200:
        intent = construct_intent()
        output_json = intent_to_json(intent)

        assistant_content = json.dumps(output_json, separators=(',', ':'))

        training_data.append({
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": intent.query_string},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    with open("data/training_data.jsonl", "w") as f:
        for entry in training_data:
            f.write(json.dumps(entry) + "\n")

    print("Generated datasets.")

if __name__ == "__main__":
    generate_datasets()
