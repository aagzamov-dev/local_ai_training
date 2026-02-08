# python -m py_compile llm_client.py

import time
from typing import Any

import requests


class LLMClient:
    def __init__(
        self,
        url: str = "http://localhost:8000/v1/chat/completions",
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: int = 60,
    ):
        self.url = url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def query(
        self,
        system_prompt: str,
        user_query: str,
        extra_system: str | None = None,
    ) -> tuple[str, float, str | None]:
        full_system = system_prompt
        if extra_system:
            full_system = f"{system_prompt}\n\n{extra_system}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_query},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        start = time.perf_counter()
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            elapsed = max(0.0, time.perf_counter() - start)
            return "", elapsed, f"connection refused: {self.url}"
        except requests.exceptions.Timeout:
            elapsed = max(0.0, time.perf_counter() - start)
            return "", elapsed, f"timeout after {self.timeout}s"
        except requests.exceptions.HTTPError as e:
            elapsed = max(0.0, time.perf_counter() - start)
            return "", elapsed, f"HTTP error: {e}"
        except requests.exceptions.RequestException as e:
            elapsed = max(0.0, time.perf_counter() - start)
            return "", elapsed, f"request error: {e}"

        elapsed = max(0.0, time.perf_counter() - start)

        try:
            data = response.json()
        except ValueError:
            return "", elapsed, "invalid JSON response from API"

        try:
            content = data["choices"][0]["message"]["content"]
            return content, elapsed, None
        except (KeyError, IndexError, TypeError):
            return "", elapsed, "unexpected API response structure"
