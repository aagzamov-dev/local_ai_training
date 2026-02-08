# =============================================================================
# EVALUATE SCRIPT FOR INTENT PARSER
# =============================================================================
# Example run commands (do not execute automatically):
# python evaluate.py --tag before --baseline data/baseline_eval.json --out runs/
# python evaluate.py --tag after --baseline data/baseline_eval.json --out runs/
# =============================================================================

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from intent_schema import normalize, strict_validate
from llm_client import LLMClient
from utils_json import extract_json


def load_baseline(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("baseline file must be a JSON array")
    return data


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compare_lists(a: list | None, b: list | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return sorted(a) == sorted(b)


def compare_values(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    return a == b


def compute_field_matches(
    predicted: dict[str, Any] | None,
    expected: dict[str, Any],
) -> dict[str, bool]:
    if predicted is None:
        return {
            "brand_exact_match": False,
            "categories_exact_match": False,
            "in_stock_match": False,
            "sku_match": False,
            "ean_match": False,
            "price_min_match": False,
            "price_max_match": False,
            "sort_by_match": False,
            "sort_order_match": False,
            "query_match": False,
        }

    pred_filters = predicted.get("filters", {}) or {}
    exp_filters = expected.get("filters", {}) or {}
    pred_price = pred_filters.get("price", {}) or {}
    exp_price = exp_filters.get("price", {}) or {}
    pred_sort = predicted.get("sort", {}) or {}
    exp_sort = expected.get("sort", {}) or {}

    return {
        "brand_exact_match": compare_lists(
            pred_filters.get("brand"),
            exp_filters.get("brand"),
        ),
        "categories_exact_match": compare_lists(
            pred_filters.get("categories"),
            exp_filters.get("categories"),
        ),
        "in_stock_match": compare_values(
            pred_filters.get("in_stock"),
            exp_filters.get("in_stock"),
        ),
        "sku_match": compare_values(
            pred_filters.get("sku"),
            exp_filters.get("sku"),
        ),
        "ean_match": compare_values(
            pred_filters.get("ean"),
            exp_filters.get("ean"),
        ),
        "price_min_match": compare_values(
            pred_price.get("min"),
            exp_price.get("min"),
        ),
        "price_max_match": compare_values(
            pred_price.get("max"),
            exp_price.get("max"),
        ),
        "sort_by_match": compare_values(
            pred_sort.get("by"),
            exp_sort.get("by"),
        ),
        "sort_order_match": compare_values(
            pred_sort.get("order"),
            exp_sort.get("order"),
        ),
        "query_match": compare_values(
            predicted.get("query"),
            expected.get("query"),
        ),
    }


def compute_overall_match(
    normalized: dict[str, Any] | None,
    expected: dict[str, Any],
) -> bool:
    if normalized is None:
        return False

    expected_normalized, _ = normalize(expected)
    if expected_normalized is None:
        return False

    return json.dumps(normalized, sort_keys=True) == json.dumps(
        expected_normalized,
        sort_keys=True,
    )


def categorize_schema_error(error: str) -> str:
    if "missing top-level keys" in error:
        return "missing_top_keys"
    if "filters missing keys" in error:
        return "missing_filter_keys"
    if "price missing keys" in error:
        return "missing_price_keys"
    if "sort missing keys" in error:
        return "missing_sort_keys"
    if "expected list" in error or "expected dict" in error or "expected str" in error:
        return "type_error"
    if "price at top-level" in error:
        return "price_at_top_level"
    if "sort.by" in error or "sort.order" in error:
        return "invalid_sort_value"
    return "other"


def run_evaluation(
    baseline: list[dict[str, Any]],
    client: LLMClient,
    system_prompt: str,
    extra_system: str | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(baseline)

    for i, item in enumerate(baseline):
        query = item["input_query"]
        expected = item["expected_output_json"]

        print(f"[{i + 1}/{total}] {query[:60]}...", file=sys.stderr)

        raw_response, latency, error = client.query(system_prompt, query, extra_system)

        if error:
            field_matches = compute_field_matches(None, expected)
            results.append({
                "input_query": query,
                "expected_output_json": expected,
                "raw_response_text": raw_response,
                "parsed_json": None,
                "json_valid": False,
                "schema_valid": False,
                "schema_errors": [error],
                "normalized_json": None,
                "latency_seconds": latency,
                **field_matches,
                "overall_exact_match": False,
            })
            continue

        parsed, _, parse_error = extract_json(raw_response)
        json_valid = parsed is not None

        if not json_valid:
            field_matches = compute_field_matches(None, expected)
            results.append({
                "input_query": query,
                "expected_output_json": expected,
                "raw_response_text": raw_response,
                "parsed_json": None,
                "json_valid": False,
                "schema_valid": False,
                "schema_errors": [parse_error or "unknown parse error"],
                "normalized_json": None,
                "latency_seconds": latency,
                **field_matches,
                "overall_exact_match": False,
            })
            continue

        schema_valid, schema_errors = strict_validate(parsed)
        normalized, _ = normalize(parsed)
        field_matches = compute_field_matches(normalized, expected)
        overall_match = compute_overall_match(normalized, expected)

        results.append({
            "input_query": query,
            "expected_output_json": expected,
            "raw_response_text": raw_response,
            "parsed_json": parsed,
            "json_valid": True,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "normalized_json": normalized,
            "latency_seconds": latency,
            **field_matches,
            "overall_exact_match": overall_match,
        })

    return results


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    json_valid = sum(1 for r in results if r["json_valid"])
    schema_valid = sum(1 for r in results if r["schema_valid"])
    exact_match = sum(1 for r in results if r["overall_exact_match"])

    field_keys = [
        "brand_exact_match",
        "categories_exact_match",
        "in_stock_match",
        "sku_match",
        "ean_match",
        "price_min_match",
        "price_max_match",
        "sort_by_match",
        "sort_order_match",
        "query_match",
    ]

    summary: dict[str, Any] = {
        "json_valid_percentage": json_valid / total * 100,
        "schema_valid_percentage": schema_valid / total * 100,
        "exact_match_percentage": exact_match / total * 100,
    }

    for key in field_keys:
        count = sum(1 for r in results if r.get(key, False))
        summary[f"{key}_percentage"] = count / total * 100

    avg_latency = sum(r["latency_seconds"] for r in results) / total
    summary["avg_latency_seconds"] = avg_latency

    error_counter: Counter[str] = Counter()
    for r in results:
        for err in r.get("schema_errors", []):
            category = categorize_schema_error(err)
            error_counter[category] += 1

    summary["schema_error_counts"] = dict(error_counter)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM predictions against baseline",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/v1/chat/completions",
        help="LLM API endpoint",
    )
    parser.add_argument(
        "--model",
        default="default",
        help="Model name to send in request",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Tag for output file (e.g., before/after)",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline eval JSON file",
    )
    parser.add_argument(
        "--out",
        default="runs/",
        help="Output directory",
    )
    parser.add_argument(
        "--system-prompt",
        default="system_prompt.txt",
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--extra-system",
        default=None,
        help="Additional text to append to system prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens for LLM response",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline(args.baseline)
    system_prompt = load_system_prompt(args.system_prompt)

    client = LLMClient(
        url=args.url,
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    results = run_evaluation(baseline, client, system_prompt, args.extra_system)
    summary = compute_summary(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"{args.tag}_{timestamp}_eval.json"

    output = {
        "tag": args.tag,
        "timestamp": timestamp,
        "api_url": args.url,
        "model": args.model,
        "total_samples": len(results),
        "summary": summary,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("EVALUATION COMPLETE", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"JSON valid:     {summary.get('json_valid_percentage', 0):.1f}%", file=sys.stderr)
    print(f"Schema valid:   {summary.get('schema_valid_percentage', 0):.1f}%", file=sys.stderr)
    print(f"Exact match:    {summary.get('exact_match_percentage', 0):.1f}%", file=sys.stderr)
    print(f"Query match:    {summary.get('query_match_percentage', 0):.1f}%", file=sys.stderr)
    print(f"Avg latency:    {summary.get('avg_latency_seconds', 0):.3f}s", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    error_counts = summary.get("schema_error_counts", {})
    if error_counts:
        print("Schema error breakdown:", file=sys.stderr)
        for err_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err_type}: {count}", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

    print(f"Results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
