import os
from typing import Optional

import requests


def normalize_base_url(url: Optional[str]):
    base = url or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in base:
        base = "http://" + base
    return base.rstrip("/")


class OllamaClient:
    def __init__(self, model="llama3:8b", base_url=None, timeout=120):
        self.model = model
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies = {"http": None, "https": None}

    def available_models(self):
        response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])]

    def ensure_model(self):
        models = self.available_models()
        if not any(self.model == name or self.model.split(":")[0] == name.split(":")[0] for name in models):
            raise RuntimeError(f"Ollama model is not installed: {self.model}")

    def generate(self, prompt, json_output=False, temperature=0.2, num_predict=None, keep_alive="5m"):
        options = {"temperature": float(temperature), "top_p": 0.9}
        if num_predict is not None:
            options["num_predict"] = int(num_predict)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        }
        if json_output:
            payload["format"] = "json"
        response = self.session.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return (response.json().get("response") or "").strip()
