# python -m py_compile intent_schema.py

from typing import Any

from generator import BRANDS, BRAND_ALIASES, CATEGORIES, CATEGORY_ALIASES

VALID_SORT_BY = {"relevance", "price"}
VALID_SORT_ORDER = {"asc", "desc"}

REQUIRED_TOP_KEYS = {"query", "filters", "sort", "page", "page_size", "confidence"}
REQUIRED_FILTER_KEYS = {"brand", "categories", "price", "in_stock", "sku", "ean"}
REQUIRED_PRICE_KEYS = {"min", "max", "currency"}
REQUIRED_SORT_KEYS = {"by", "order"}


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            alias_map[alias.lower()] = canonical
    for brand in BRANDS:
        alias_map[brand.lower()] = brand
    return alias_map


def _build_category_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            alias_map[alias.lower()] = canonical
    for cat in CATEGORIES:
        alias_map[cat.lower()] = cat
    return alias_map


BRAND_ALIAS_MAP = _build_alias_map()
CATEGORY_ALIAS_MAP = _build_category_alias_map()


def _check_type(value: Any, expected_type: type | tuple, field_name: str, errors: list[str]) -> bool:
    if not isinstance(value, expected_type):
        errors.append(f"{field_name}: expected {expected_type}, got {type(value).__name__}")
        return False
    return True


def _check_list_of_strings(value: Any, field_name: str, errors: list[str]) -> bool:
    if not isinstance(value, list):
        errors.append(f"{field_name}: expected list, got {type(value).__name__}")
        return False
    for i, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{field_name}[{i}]: expected str, got {type(item).__name__}")
            return False
    return True


def _check_optional_type(value: Any, expected_type: type | tuple, field_name: str, errors: list[str]) -> bool:
    if value is None:
        return True
    return _check_type(value, expected_type, field_name, errors)


def strict_validate(obj: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []

    if not isinstance(obj, dict):
        errors.append(f"root: expected dict, got {type(obj).__name__}")
        return False, errors

    missing_top = REQUIRED_TOP_KEYS - set(obj.keys())
    if missing_top:
        errors.append(f"missing top-level keys: {sorted(missing_top)}")

    if "query" in obj:
        _check_type(obj["query"], str, "query", errors)

    if "filters" in obj:
        filters = obj["filters"]
        if not isinstance(filters, dict):
            errors.append(f"filters: expected dict, got {type(filters).__name__}")
        else:
            missing_filter = REQUIRED_FILTER_KEYS - set(filters.keys())
            if missing_filter:
                errors.append(f"filters missing keys: {sorted(missing_filter)}")

            if "brand" in filters:
                _check_list_of_strings(filters["brand"], "filters.brand", errors)

            if "categories" in filters:
                _check_list_of_strings(filters["categories"], "filters.categories", errors)

            if "price" in filters:
                price = filters["price"]
                if not isinstance(price, dict):
                    errors.append(f"filters.price: expected dict, got {type(price).__name__}")
                else:
                    missing_price = REQUIRED_PRICE_KEYS - set(price.keys())
                    if missing_price:
                        errors.append(f"filters.price missing keys: {sorted(missing_price)}")
                    if "min" in price:
                        _check_optional_type(price["min"], (int, float), "filters.price.min", errors)
                    if "max" in price:
                        _check_optional_type(price["max"], (int, float), "filters.price.max", errors)
                    if "currency" in price and price["currency"] != "EUR":
                        errors.append(f"filters.price.currency: expected 'EUR', got '{price['currency']}'")

            if "in_stock" in filters:
                _check_optional_type(filters["in_stock"], bool, "filters.in_stock", errors)

            if "sku" in filters:
                _check_optional_type(filters["sku"], str, "filters.sku", errors)

            if "ean" in filters:
                _check_optional_type(filters["ean"], str, "filters.ean", errors)

    if "price" in obj and "filters" in obj and isinstance(obj["filters"], dict):
        if "price" not in obj["filters"]:
            errors.append("price at top-level instead of filters.price")

    if "sort" in obj:
        sort = obj["sort"]
        if not isinstance(sort, dict):
            errors.append(f"sort: expected dict, got {type(sort).__name__}")
        else:
            missing_sort = REQUIRED_SORT_KEYS - set(sort.keys())
            if missing_sort:
                errors.append(f"sort missing keys: {sorted(missing_sort)}")
            if "by" in sort and sort["by"] not in VALID_SORT_BY:
                errors.append(f"sort.by: expected one of {VALID_SORT_BY}, got '{sort['by']}'")
            if "order" in sort and sort["order"] not in VALID_SORT_ORDER:
                errors.append(f"sort.order: expected one of {VALID_SORT_ORDER}, got '{sort['order']}'")

    if "page" in obj:
        _check_type(obj["page"], int, "page", errors)

    if "page_size" in obj:
        _check_type(obj["page_size"], int, "page_size", errors)

    if "confidence" in obj:
        _check_type(obj["confidence"], (int, float), "confidence", errors)

    return len(errors) == 0, errors


def normalize(obj: Any) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []

    if not isinstance(obj, dict):
        return None, ["cannot normalize non-dict"]

    result = {}

    result["query"] = obj.get("query", "")
    if not isinstance(result["query"], str):
        result["query"] = str(result["query"])
        warnings.append("query coerced to string")

    filters: dict[str, Any] = {}
    src_filters = obj.get("filters", {})
    if not isinstance(src_filters, dict):
        src_filters = {}
        warnings.append("filters was not a dict, using empty")

    brands = src_filters.get("brand", [])
    if isinstance(brands, list):
        normalized_brands = []
        for b in brands:
            if isinstance(b, str):
                canonical = BRAND_ALIAS_MAP.get(b.lower(), b)
                if canonical != b:
                    warnings.append(f"brand '{b}' canonicalized to '{canonical}'")
                normalized_brands.append(canonical)
        filters["brand"] = normalized_brands
    else:
        filters["brand"] = []
        warnings.append("brand was not a list, using empty")

    categories = src_filters.get("categories", [])
    if isinstance(categories, list):
        normalized_cats = []
        for c in categories:
            if isinstance(c, str):
                canonical = CATEGORY_ALIAS_MAP.get(c.lower(), c)
                if canonical != c:
                    warnings.append(f"category '{c}' canonicalized to '{canonical}'")
                normalized_cats.append(canonical)
        filters["categories"] = normalized_cats
    else:
        filters["categories"] = []
        warnings.append("categories was not a list, using empty")

    src_price = src_filters.get("price", {})
    top_price = obj.get("price")
    if top_price is not None and isinstance(top_price, dict):
        src_price = top_price
        warnings.append("price moved from top-level into filters.price")

    if not isinstance(src_price, dict):
        src_price = {}

    price: dict[str, Any] = {
        "min": src_price.get("min"),
        "max": src_price.get("max"),
        "currency": src_price.get("currency", "EUR"),
    }
    if price["currency"] != "EUR":
        warnings.append(f"currency '{price['currency']}' changed to 'EUR'")
        price["currency"] = "EUR"
    if "currency" not in src_price:
        warnings.append("currency was missing, set to 'EUR'")
    filters["price"] = price

    filters["in_stock"] = src_filters.get("in_stock")

    sku = src_filters.get("sku")
    if isinstance(sku, list):
        if len(sku) == 1 and isinstance(sku[0], str):
            sku = sku[0]
            warnings.append("sku converted from single-element list to string")
        elif len(sku) == 0:
            sku = None
        else:
            sku = sku[0] if sku else None
            warnings.append("sku was list with multiple elements, took first")
    filters["sku"] = sku if isinstance(sku, str) else None

    ean = src_filters.get("ean")
    if isinstance(ean, list):
        if len(ean) == 1 and isinstance(ean[0], str):
            ean = ean[0]
            warnings.append("ean converted from single-element list to string")
        elif len(ean) == 0:
            ean = None
        else:
            ean = ean[0] if ean else None
            warnings.append("ean was list with multiple elements, took first")
    filters["ean"] = ean if isinstance(ean, str) else None

    result["filters"] = filters

    src_sort = obj.get("sort", {})
    if not isinstance(src_sort, dict):
        src_sort = {}
    sort_by = src_sort.get("by", "relevance")
    sort_order = src_sort.get("order", "desc")
    if sort_by not in VALID_SORT_BY:
        warnings.append(f"sort.by '{sort_by}' invalid, defaulting to 'relevance'")
        sort_by = "relevance"
    if sort_order not in VALID_SORT_ORDER:
        warnings.append(f"sort.order '{sort_order}' invalid, defaulting to 'desc'")
        sort_order = "desc"
    result["sort"] = {"by": sort_by, "order": sort_order}

    result["page"] = obj.get("page", 1)
    if not isinstance(result["page"], int):
        result["page"] = 1

    result["page_size"] = obj.get("page_size", 24)
    if not isinstance(result["page_size"], int):
        result["page_size"] = 24

    confidence = obj.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
        warnings.append("confidence was not numeric, set to 0.0")
    if "confidence" not in obj:
        warnings.append("confidence was missing, set to 0.0")
    result["confidence"] = float(confidence)

    return result, warnings
