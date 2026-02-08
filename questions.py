import json
import requests
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_API_URL = "http://localhost:8080/v1/chat/completions"
SYSTEM_PROMPT_PATH = "system_prompt.txt"
QUESTIONS_PATH = "data/questions.json"
OUT_DIR = "runs"

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def load_questions(path: str) -> List[str]:
    data = json.loads(load_text(path))
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    raise ValueError(f"{path} must be a JSON array of strings")

def query_llm(
    url: str,
    system_prompt: str,
    user_query: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 30,
) -> Optional[Dict[str, Any]]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def extract_json_content(response: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    if not response:
        return None, "No response"

    if "error" in response:
        return None, response["error"]

    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        return None, json.dumps(response, ensure_ascii=False)

    raw = content
    clean = content.strip()

    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        if clean.endswith("```"):
            clean = clean[:-3].strip()

    try:
        return json.loads(clean), raw
    except json.JSONDecodeError:
        return None, raw

def run_capture(
    api_url: str,
    tag: str,
    model_name: str,
    questions_path: str,
    system_prompt_path: str,
) -> str:
    system_prompt = load_text(system_prompt_path)
    questions = load_questions(questions_path)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(OUT_DIR) / f"{tag}_{ts}.json")

    results = []
    valid_json = 0

    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q}")

        t0 = time.time()
        api_resp = query_llm(api_url, system_prompt, q, model_name=model_name)
        latency = time.time() - t0

        parsed, raw = extract_json_content(api_resp)
        is_valid = parsed is not None
        if is_valid:
            valid_json += 1

        results.append(
            {
                "input_query": q,
                "predicted": parsed,
                "raw_response": raw,
                "is_valid_json": is_valid,
                "latency_seconds": latency,
            }
        )

    summary = {
        "tag": tag,
        "timestamp": ts,
        "api_url": api_url,
        "model": model_name,
        "total_samples": len(questions),
        "valid_json_percentage": (valid_json / len(questions)) * 100 if questions else 0,
        "results": results,
    }

    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved: {out_path}")
    print(f"Valid JSON: {valid_json}/{len(questions)} ({summary['valid_json_percentage']:.1f}%)")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture model outputs before/after fine-tune")
    parser.add_argument("--url", default=DEFAULT_API_URL)
    parser.add_argument("--tag", required=True, help="before | after | any label")
    parser.add_argument("--model", default="default")
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--system", default=SYSTEM_PROMPT_PATH)
    args = parser.parse_args()

    run_capture(
        api_url=args.url,
        tag=args.tag,
        model_name=args.model,
        questions_path=args.questions,
        system_prompt_path=args.system,
    )
