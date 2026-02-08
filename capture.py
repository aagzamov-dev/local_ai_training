# python capture.py --url http://localhost:8000/v1/chat/completions \
#     --model qwen2.5-3b --tag before --in data/questions.json --out runs/

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from intent_schema import normalize, strict_validate
from llm_client import LLMClient
from utils_json import extract_json


def load_questions(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("questions file must be a JSON array of strings")
    return [str(q) for q in data]


def load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_capture(
    questions: list[str],
    client: LLMClient,
    system_prompt: str,
    extra_system: str | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(questions)

    for i, query in enumerate(questions):
        print(f"[{i + 1}/{total}] {query[:60]}...", file=sys.stderr)

        raw_response, latency, error = client.query(system_prompt, query, extra_system)

        if error:
            results.append({
                "input_query": query,
                "raw_response_text": raw_response,
                "parsed_json": None,
                "json_valid": False,
                "schema_valid": False,
                "schema_errors": [error],
                "normalized_json": None,
                "latency_seconds": latency,
            })
            continue

        parsed, _, parse_error = extract_json(raw_response)
        json_valid = parsed is not None

        if not json_valid:
            results.append({
                "input_query": query,
                "raw_response_text": raw_response,
                "parsed_json": None,
                "json_valid": False,
                "schema_valid": False,
                "schema_errors": [parse_error or "unknown parse error"],
                "normalized_json": None,
                "latency_seconds": latency,
            })
            continue

        schema_valid, schema_errors = strict_validate(parsed)
        normalized, _ = normalize(parsed)

        results.append({
            "input_query": query,
            "raw_response_text": raw_response,
            "parsed_json": parsed,
            "json_valid": True,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "normalized_json": normalized,
            "latency_seconds": latency,
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture LLM predictions for questions")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions", help="LLM API endpoint")
    parser.add_argument("--model", default="default", help="Model name to send in request")
    parser.add_argument("--tag", required=True, help="Tag for output file (e.g., before/after)")
    parser.add_argument("--in", dest="input_file", required=True, help="Path to questions JSON file")
    parser.add_argument("--out", default="runs/", help="Output directory")
    parser.add_argument("--system-prompt", default="system_prompt.txt", help="Path to system prompt file")
    parser.add_argument("--extra-system", default=None, help="Additional text to append to system prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for LLM response")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.input_file)
    system_prompt = load_system_prompt(args.system_prompt)

    client = LLMClient(
        url=args.url,
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    results = run_capture(questions, client, system_prompt, args.extra_system)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = out_dir / f"{args.tag}_{timestamp}_capture.json"

    json_valid_count = sum(1 for r in results if r["json_valid"])
    schema_valid_count = sum(1 for r in results if r["schema_valid"])
    total = len(results)

    output = {
        "tag": args.tag,
        "timestamp": timestamp,
        "api_url": args.url,
        "model": args.model,
        "total_samples": total,
        "json_valid_percentage": (json_valid_count / total * 100) if total > 0 else 0.0,
        "schema_valid_percentage": (schema_valid_count / total * 100) if total > 0 else 0.0,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nCapture complete!", file=sys.stderr)
    print(f"JSON valid: {json_valid_count}/{total} ({output['json_valid_percentage']:.1f}%)", file=sys.stderr)
    print(f"Schema valid: {schema_valid_count}/{total} ({output['schema_valid_percentage']:.1f}%)", file=sys.stderr)
    print(f"Results saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
