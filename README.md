# Ecommerce Search Intent Dataset

This dataset is designed to train a **deterministic search intent parser** for a standard e-commerce platform. The goal is to convert natural language queries into strict, structured JSON for search engines like Meilisearch or Elasticsearch.

## 🎯 Purpose

- **Input**: User natural language query (e.g., "cheap Sony headphones wireless").
- **Output**: Structured JSON Intent (e.g., `brand="Sony"`, `residuals="wireless"`, `sort="price ASC"`).
- **Core Philosophy**: **Strict Residuals**. The `query` field in the output must ONLY contain keywords that could not be mapped to specific filters. Providing "Sony" in both `filters.brand` and `query` is considered a hallucination/error.

## 📂 Files

- **`data/training_data.jsonl`**: 200 high-quality examples in ChatML format.
- **`data/baseline_eval.json`**: 100 "Before vs After" examples with explanatory notes for ground-truth evaluation.
- **`generator.py`**: The deterministic Python script used to generate these datasets.

## 🧬 Schema & Rules

### JSON Output Schema

```json
{
  "query": "string (residuals only)",
  "filters": {
    "brand": ["string"],
    "categories": ["string"],
    "price": { "min": number|null, "max": number|null, "currency": "EUR" },
    "in_stock": boolean|null,
    "sku": "string|null",
    "ean": "string|null"
  },
  "sort": { "by": "relevance|price", "order": "desc|asc" },
  "page": 1,
  "page_size": 24,
  "confidence": number (0.0 - 1.0)
}
```

### 🧠 Parsing Rules (The "CRITICAL" List)

1.  **Query Residuals**:
    - **NEVER** include extracted entities (Brand, Category) in the `query` string.
    - **NEVER** include filter phrases ("in stock", "under 500 dollars") in the `query` string.
    - **NEVER** include sorting phrases ("cheapest", "premium") in the `query` string.
    - **NEVER** use generic nouns like "products" or "items" as the query.
    - _Example_: "high end Sony products" -> `query: ""` (Sort=Desc, Brand=Sony).
    - _Example_: "wireless black Sony headphones" -> `query: "wireless black"`.

2.  **Sorting Logic**:
    - **Relevance (Default)**: `sort: { "by": "relevance", "order": "desc" }`
    - **Cheapest**: Keywords "cheap", "budget", "lowest price", "least expensive" -> `sort: { "by": "price", "order": "asc" }`
    - **Premium**: Keywords "premium", "high end", "top tier", "most expensive" -> `sort: { "by": "price", "order": "desc" }`

3.  **Comparators**:
    - "under", "below", "max", "cheaper than" -> `price.max`
    - "over", "above", "min", "more than" -> `price.min`

4.  **Identifiers**:
    - If a SKU or EAN is detected, the `query` field must be empty `""`.

5.  **Confidence**:
    - **1.0**: SKU/EAN exact match.
    - **~0.95**: Full struct (Brand + Category + Filter).
    - **~0.85**: Partial struct (Brand only, or Category + Residual).
    - **~0.75**: Generic/Ambiguous.

## 🚀 Usage

### 1. Fine-tuning

Use `finetune.py` (QLoRA) with the `data/training_data.jsonl` file.

```bash
python3 finetune.py
```

### 2. Evaluation

Use `evaluate.py` to compare your model's output against `data/baseline_eval.json`.

```bash
python3 evaluate.py
```

## ✅ Validation

To validate model outputs:

1.  Ensure output is valid JSON.
2.  Check that `query` does not contain any word found in `filters.brand` or `filters.categories`.
3.  Check that `query` does not contain "sku" or "ean".
# local_ai_training
# local_ai_training
# local_ai_training
