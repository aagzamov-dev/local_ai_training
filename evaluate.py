import json
import requests
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any

# --- CONFIGURATION ---
DEFAULT_API_URL = "http://localhost:8080/v1/chat/completions"
SYSTEM_PROMPT_PATH = "system_prompt.txt"
BASELINE_DATA_PATH = "data/baseline_eval.json"

def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def query_llm(url: str, system_prompt: str, user_query: str, model_name: str = "default") -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0, # Deterministic
        "max_tokens": 512,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying LLM: {e}")
        return None

def extract_json_content(response: Dict[str, Any]) -> tuple[Dict, str]:
    """
    Extracts JSON from LLM response. 
    Returns (parsed_json, raw_content_string).
    """
    if not response or "choices" not in response:
        return None, "No valid response"
        
    content = response["choices"][0]["message"]["content"]
    
    # Attempt to parse
    try:
        # Strip markdown code blocks if present
        clean_content = content.strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        
        return json.loads(clean_content), content
    except json.JSONDecodeError:
        return None, content

def run_evaluation(api_url: str):
    print(f"Starting evaluation against {api_url}...")
    
    # Load resources
    try:
        system_prompt = load_file(SYSTEM_PROMPT_PATH)
        baseline_data = json.loads(load_file(BASELINE_DATA_PATH))
    except FileNotFoundError as e:
        print(f"Critical Error: {e}")
        return

    results = []
    valid_json_count = 0
    total = len(baseline_data)
    
    for i, item in enumerate(baseline_data):
        input_q = item["input_query"]
        expected = item["expected_output_json"]
        
        print(f"[{i+1}/{total}] Query: {input_q}")
        
        start_time = time.time()
        api_response = query_llm(api_url, system_prompt, input_q)
        latency = time.time() - start_time
        
        prediction, raw_text = extract_json_content(api_response)
        
        is_valid = prediction is not None
        if is_valid:
            valid_json_count += 1
            
        result_entry = {
            "input_query": input_q,
            "expected": expected,
            "predicted": prediction,
            "raw_response": raw_text,
            "is_valid_json": is_valid,
            "latency_seconds": latency
        }
        results.append(result_entry)

    # Summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"eval_results_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_samples": total,
        "valid_json_percentage": (valid_json_count / total) * 100,
        "api_url": api_url,
        "results": results
    }
    
    with open(output_filename, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nEvaluation Complete!")
    print(f"Valid JSON: {valid_json_count}/{total} ({(valid_json_count/total)*100:.1f}%)")
    print(f"Detailed results saved to: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on Qwen2.5 Intent Parser")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="LLM API Endpoint URL")
    args = parser.parse_args()
    
    run_evaluation(args.url)
