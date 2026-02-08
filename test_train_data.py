import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

REQUIRED_TOP = {"query", "filters", "sort", "page", "page_size", "confidence"}
REQUIRED_FILTERS = {"brand", "categories", "price", "in_stock", "sku", "ean"}
REQUIRED_PRICE = {"min", "max", "currency"}
REQUIRED_SORT = {"by", "order"}

def validate_target(obj: Dict[str, Any]) -> List[str]:
    errs: List[str] = []

    if set(obj.keys()) != REQUIRED_TOP:
        errs.append(f"top keys mismatch: {sorted(obj.keys())}")

    if not isinstance(obj.get("query"), str):
        errs.append("query must be string")

    filters = obj.get("filters")
    if not isinstance(filters, dict):
        errs.append("filters must be object")
        return errs

    if set(filters.keys()) != REQUIRED_FILTERS:
        errs.append(f"filters keys mismatch: {sorted(filters.keys())}")

    if not isinstance(filters.get("brand"), list):
        errs.append("filters.brand must be list")
    if not isinstance(filters.get("categories"), list):
        errs.append("filters.categories must be list")

    price = filters.get("price")
    if not isinstance(price, dict):
        errs.append("filters.price must be object")
    else:
        if set(price.keys()) != REQUIRED_PRICE:
            errs.append(f"price keys mismatch: {sorted(price.keys())}")
        if price.get("currency") != "EUR":
            errs.append("price.currency must be EUR")

    if filters.get("sku") is not None and not isinstance(filters.get("sku"), str):
        errs.append("filters.sku must be string or null")
    if filters.get("ean") is not None and not isinstance(filters.get("ean"), str):
        errs.append("filters.ean must be string or null")

    sort = obj.get("sort")
    if not isinstance(sort, dict):
        errs.append("sort must be object")
    else:
        if set(sort.keys()) != REQUIRED_SORT:
            errs.append(f"sort keys mismatch: {sorted(sort.keys())}")
        if sort.get("by") not in {"relevance", "price"}:
            errs.append("sort.by must be relevance or price")
        if sort.get("order") not in {"asc", "desc"}:
            errs.append("sort.order must be asc or desc")

    if obj.get("page") != 1:
        errs.append("page must be 1")
    if obj.get("page_size") != 24:
        errs.append("page_size must be 24")

    if not isinstance(obj.get("confidence"), (int, float)):
        errs.append("confidence must be number")

    return errs

def main(path: str) -> None:
    p = Path(path)
    bad = 0
    total = 0
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        msgs = row.get("messages", [])
        if len(msgs) != 3:
            print(f"[line {line_no}] messages must be len=3")
            bad += 1
            continue
        if msgs[0].get("role") != "system" or msgs[1].get("role") != "user" or msgs[2].get("role") != "assistant":
            print(f"[line {line_no}] roles must be system/user/assistant")
            bad += 1
            continue

        target_raw = msgs[2].get("content", "")
        try:
            target = json.loads(target_raw)
        except json.JSONDecodeError as e:
            print(f"[line {line_no}] assistant content not valid json: {e}")
            bad += 1
            continue

        errs = validate_target(target)
        if errs:
            print(f"[line {line_no}] schema errors:")
            for e in errs:
                print("  -", e)
            bad += 1

    print(f"Checked {total} rows. Bad rows: {bad}.")

if __name__ == "__main__":
    main("data/training_data.jsonl")
