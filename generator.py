import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BRANDS = [
    "Apple",
    "Samsung",
    "Sony",
    "Bosch",
    "LG",
    "Dell",
    "HP",
    "Lenovo",
    "Asus",
    "Philips",
    "Siemens",
    "Miele",
    "Dyson",
    "Logitech",
    "Razer",
    "Nikon",
    "Canon",
    "Microsoft",
    "Nintendo",
    "Acer",
]

BRAND_ALIASES: Dict[str, List[str]] = {
    "HP": ["HP", "Hp", "hp"],
    "LG": ["LG", "Lg", "lg"],
    "Asus": ["Asus", "ASUS", "asus"],
    "Sony": ["Sony", "SONY", "sony"],
    "Dell": ["Dell", "DELL", "dell"],
    "Lenovo": ["Lenovo", "LENOVO", "lenovo"],
    "Samsung": ["Samsung", "SAMSUNG", "samsung"],
    "Apple": ["Apple", "APPLE", "apple"],
}

CATEGORIES = [
    "Smartphones",
    "Laptops",
    "Tablets",
    "Televisions",
    "Headphones",
    "Smart Watches",
    "Washing Machines",
    "Refrigerators",
    "Vacuum Cleaners",
    "Coffee Machines",
    "Gaming Consoles",
    "Monitors",
    "Cameras",
    "Keyboards",
    "Mice",
]

CATEGORY_ALIASES: Dict[str, List[str]] = {
    "Smart Watches": ["Smart Watches", "smart watches", "smartwatch", "smart watch"],
    "Gaming Consoles": ["Gaming Consoles", "gaming consoles", "console", "consoles"],
    "Vacuum Cleaners": ["Vacuum Cleaners", "vacuum cleaners", "vacuum", "cleaner"],
    "Coffee Machines": ["Coffee Machines", "coffee machines", "coffee machine", "espresso machine"],
}

RESIDUAL_PHRASES: List[List[str]] = [
    ["gaming"],
    ["wireless"],
    ["bluetooth"],
    ["noise", "cancelling"],
    ["black"],
    ["white"],
    ["silver"],
    ["gold"],
    ["portable"],
    ["4k"],
    ["oled"],
    ["qled"],
    ["smart"],
    ["android"],
    ["ios"],
    ["windows"],
    ["mac"],
    ["pro"],
    ["mini"],
    ["max"],
    ["ultra"],
    ["slim"],
    ["lightweight"],
    ["mechanical"],
    ["ergonomic"],
]

STOCK_PHRASES = ["in stock", "available now", "ready to ship", "available"]

SORT_PHRASES: Dict[str, Dict[str, str]] = {
    "cheapest": {"by": "price", "order": "asc"},
    "lowest price": {"by": "price", "order": "asc"},
    "budget": {"by": "price", "order": "asc"},
    "premium": {"by": "price", "order": "desc"},
    "high end": {"by": "price", "order": "desc"},
    "most expensive": {"by": "price", "order": "desc"},
    "top tier": {"by": "price", "order": "desc"},
}

PRICE_MAX_PHRASES = ["under", "below", "less than", "at most", "maximum"]
PRICE_MIN_PHRASES = ["over", "above", "more than", "at least", "minimum"]


@dataclass
class SearchIntent:
    brand: Optional[str] = None
    category: Optional[str] = None
    residual_tokens: List[str] = field(default_factory=list)
    price_min: Optional[int] = None
    price_max: Optional[int] = None
    in_stock: Optional[bool] = None
    sku: Optional[str] = None
    ean: Optional[str] = None
    sort_by: str = "relevance"
    sort_order: str = "desc"
    query_string: str = ""
    confidence: float = 0.0


def generate_sku() -> str:
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))


def generate_ean() -> str:
    return "".join(random.choices("0123456789", k=13))


def pick_brand_surface(brand: str) -> str:
    return random.choice(BRAND_ALIASES.get(brand, [brand]))


def pick_category_surface(category: str) -> str:
    return random.choice(CATEGORY_ALIASES.get(category, [category]))


def pick_residuals(min_n: int, max_n: int) -> List[str]:
    n = random.randint(min_n, max_n)
    phrases = random.sample(RESIDUAL_PHRASES, k=n)
    tokens: List[str] = []
    for p in phrases:
        tokens.extend(p)
    return tokens


def build_price_phrase(intent: SearchIntent) -> str:
    t = random.choices(["none", "max", "min", "range"], weights=[55, 20, 15, 10])[0]
    if t == "none":
        return ""
    a = random.randint(5, 200) * 10
    if t == "max":
        intent.price_max = a
        return f"{random.choice(PRICE_MAX_PHRASES)} {a} eur"
    if t == "min":
        intent.price_min = a
        return f"{random.choice(PRICE_MIN_PHRASES)} {a} eur"
    b = a + random.randint(10, 150) * 10
    intent.price_min = a
    intent.price_max = b
    return f"between {a} and {b} euros"


def maybe_stock(intent: SearchIntent) -> str:
    if random.random() < 0.25:
        intent.in_stock = True
        return random.choice(STOCK_PHRASES)
    intent.in_stock = None
    return ""


def maybe_sort(intent: SearchIntent) -> str:
    if random.random() < 0.22:
        phrase = random.choice(list(SORT_PHRASES.keys()))
        intent.sort_by = SORT_PHRASES[phrase]["by"]
        intent.sort_order = SORT_PHRASES[phrase]["order"]
        return phrase
    return ""


def compute_confidence(intent: SearchIntent) -> float:
    if intent.sku or intent.ean:
        return 1.0
    score = 0.75
    if intent.brand and intent.category:
        score = 0.95
    elif intent.brand or intent.category:
        score = 0.85
    if intent.price_min is not None or intent.price_max is not None:
        score = min(0.98, score + 0.03)
    if intent.in_stock:
        score = min(0.98, score + 0.02)
    if intent.sort_by != "relevance":
        score = min(0.98, score + 0.01)
    return round(score, 2)


def construct_intent() -> SearchIntent:
    intent = SearchIntent()

    intent_type = random.choices(
        ["brand_cat", "brand_cat_res", "cat_res", "brand_only", "cat_only", "sku", "ean"],
        weights=[24, 26, 18, 10, 12, 5, 5],
    )[0]

    if intent_type == "sku":
        intent.sku = generate_sku()
        intent.query_string = f"sku {intent.sku}"
        intent.confidence = 1.0
        return intent

    if intent_type == "ean":
        intent.ean = generate_ean()
        intent.query_string = f"ean {intent.ean}"
        intent.confidence = 1.0
        return intent

    if intent_type in ["brand_cat", "brand_cat_res", "brand_only"]:
        intent.brand = random.choice(BRANDS)

    if intent_type in ["brand_cat", "brand_cat_res", "cat_res", "cat_only"]:
        intent.category = random.choice(CATEGORIES)

    if intent_type in ["brand_cat_res", "cat_res"]:
        intent.residual_tokens = pick_residuals(1, 3)
    else:
        if random.random() < 0.25:
            intent.residual_tokens = pick_residuals(1, 2)

    sort_phrase = maybe_sort(intent)
    price_phrase = build_price_phrase(intent)
    stock_phrase = maybe_stock(intent)

    parts: List[str] = []

    if sort_phrase:
        parts.append(sort_phrase)

    if intent.residual_tokens and random.random() < 0.5:
        parts.append(" ".join(intent.residual_tokens))

    if intent.brand:
        parts.append(pick_brand_surface(intent.brand))

    if intent.category:
        parts.append(pick_category_surface(intent.category))
    else:
        if intent.brand and random.random() < 0.35:
            parts.append("products")

    if intent.residual_tokens and " ".join(intent.residual_tokens) not in " ".join(parts):
        parts.append(" ".join(intent.residual_tokens))

    if price_phrase:
        parts.append(price_phrase)

    if stock_phrase:
        parts.append(stock_phrase)

    intent.query_string = " ".join([p for p in parts if p]).strip()
    intent.confidence = compute_confidence(intent)
    return intent


def intent_to_json(intent: SearchIntent) -> Dict[str, Any]:
    query = " ".join(intent.residual_tokens).strip()
    return {
        "query": query,
        "filters": {
            "brand": [intent.brand] if intent.brand else [],
            "categories": [intent.category] if intent.category else [],
            "price": {"min": intent.price_min, "max": intent.price_max, "currency": "EUR"},
            "in_stock": intent.in_stock,
            "sku": intent.sku,
            "ean": intent.ean,
        },
        "sort": {"by": intent.sort_by, "order": intent.sort_order},
        "page": 1,
        "page_size": 24,
        "confidence": intent.confidence,
    }


def note_for_eval(intent: SearchIntent) -> str:
    if intent.sku:
        return "Type: SKU. Exact identifier extracted."
    if intent.ean:
        return "Type: EAN. Exact identifier extracted."
    bits: List[str] = ["Type: Query."]
    if intent.brand:
        bits.append("Brand extracted.")
    if intent.category:
        bits.append("Category extracted.")
    if intent.price_min is not None or intent.price_max is not None:
        bits.append("Price constraint extracted.")
    if intent.in_stock:
        bits.append("Stock constraint extracted.")
    if intent.sort_by != "relevance":
        bits.append("Sort extracted.")
    if intent.residual_tokens:
        bits.append(f"Residuals kept: '{' '.join(intent.residual_tokens)}'.")
    else:
        bits.append("No residuals; query is empty.")
    return " ".join(bits)


SYS_PROMPT = """You are a deterministic ecommerce search intent parser for a European electronics ecommerce platform.
Convert user search queries into STRICT JSON for Meilisearch.
Output MUST be a single valid JSON object and nothing else. No markdown. No explanations.
Schema:
{
"query":"",
"filters":{
"brand":[],
"categories":[],
"price":{"min":null,"max":null,"currency":"EUR"},
"in_stock":null,
"sku":null,
"ean":null
},
"sort":{"by":"relevance","order":"desc"},
"page":1,
"page_size":24,
"confidence":0.0
}
Rules:
- Keep ONLY leftover descriptive keywords in "query" (residuals).
- Extract brand/category/price/in_stock/sku/ean only when explicit.
- If unknown, keep [] or null.
- Return valid JSON only."""


def generate_datasets(
    out_dir: str = "data",
    baseline_n: int = 100,
    train_n: int = 200,
    seed: Optional[int] = 42,
) -> None:
    if seed is not None:
        random.seed(seed)

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    baseline: List[Dict[str, Any]] = []
    for _ in range(baseline_n):
        intent = construct_intent()
        baseline.append(
            {
                "input_query": intent.query_string,
                "expected_output_json": intent_to_json(intent),
                "note": note_for_eval(intent),
            }
        )

    (p / "baseline_eval.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    with (p / "training_data.jsonl").open("w", encoding="utf-8") as f:
        for _ in range(train_n):
            intent = construct_intent()
            out = intent_to_json(intent)
            assistant_content = json.dumps(out, separators=(",", ":"), ensure_ascii=False)
            row = {
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": intent.query_string},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {baseline_n} baseline examples and {train_n} training examples in '{out_dir}/'.")


if __name__ == "__main__":
    generate_datasets()
