import ast
import json
import os
import re
from string import Template

import numpy as np

from .llm_client import OllamaClient


CALL_COUNT = 0
PROMPT_TEMPLATE = Template("""你是主题建模助手。仅基于 TopWords 与 DocSnippets，为该主题生成严格 JSON：
{
  "label": "",
  "description": "",
  "alt_keywords": ["", "", ""]
}
只输出 JSON，键名固定为 label、description、alt_keywords，alt_keywords 必须为字符串数组。

[TopWords]
$top_words

[DocSnippets]
$doc_snippets
""")
JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
TRAILING_COMMA = re.compile(r",\s*([}\]])")
SINGLE_QUOTE_VALUE = re.compile(r"(?P<key>\"[^\"]+\"\s*:\s*)'(?P<val>[^']*)'")


def build_prompt(top_words, snippets, max_snippet_chars=400):
    snippets = [(text or "")[:max_snippet_chars].strip().replace("\n", " ") for text in snippets if text and text.strip()]
    words = ", ".join(top_words[:20]) if top_words else ""
    block = "\n\n".join(f"- {text}" for text in snippets[:5]) if snippets else "- （无代表片段）"
    return PROMPT_TEMPLATE.substitute(top_words=words, doc_snippets=block)


def validate_payload(payload):
    payload = dict(payload or {})
    label = str(payload.get("label", ""))[:80]
    description = str(payload.get("description", ""))[:300]
    keywords = payload.get("alt_keywords", [])
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",") if item.strip()]
    elif isinstance(keywords, list):
        keywords = [str(item).strip() for item in keywords if str(item).strip()]
    elif keywords is None:
        keywords = []
    else:
        keywords = [str(keywords)]
    return {"label": label, "description": description, "alt_keywords": keywords[:6]}


def parse_payload(text):
    match = JSON_BLOCK.search(text or "")
    if match:
        candidate = match.group(1)
    else:
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        candidate = text[start:end + 1] if start >= 0 and end > start else ""
    if not candidate:
        return None
    candidate = TRAILING_COMMA.sub(r"\1", candidate)
    candidate = SINGLE_QUOTE_VALUE.sub(r'\g<key>"\g<val>"', candidate)
    candidate = candidate.replace("，", ",").replace("、", ",")
    try:
        return validate_payload(json.loads(candidate))
    except Exception:
        try:
            value = ast.literal_eval(candidate)
            return validate_payload(value) if isinstance(value, dict) else None
        except Exception:
            return None


def label_topics(top_words_per_topic, documents, theta, documents_per_topic=3, host="http://127.0.0.1:11434", model="llama3:8b", max_snippet_chars=400, output_json=None, retries=1):
    global CALL_COUNT
    client = OllamaClient(model=model, base_url=host, timeout=90)
    client.ensure_model()
    topic_count = len(top_words_per_topic)
    if theta.shape[1] != topic_count:
        raise ValueError("theta topic dimension does not match topic count")
    results = []
    for topic_id in range(topic_count):
        indices = np.argsort(-theta[:, topic_id])[:max(1, documents_per_topic)]
        snippets = [documents[i] for i in indices if 0 <= i < len(documents)]
        prompt = build_prompt(top_words_per_topic[topic_id], snippets, max_snippet_chars)
        payload = None
        raw = ""
        for attempt in range(max(1, retries + 1)):
            try:
                CALL_COUNT += 1
                raw = client.generate(prompt, json_output=True, temperature=0.0)
                payload = validate_payload(json.loads(raw))
                break
            except Exception:
                if attempt == retries:
                    payload = parse_payload(raw)
        if payload is None:
            words = top_words_per_topic[topic_id]
            payload = {"label": " / ".join(words[:3] or [f"Topic-{topic_id}"]), "description": "", "alt_keywords": words[3:8]}
        results.append({"topic_id": topic_id, "top_words": top_words_per_topic[topic_id], **validate_payload(payload)})
    output = {"model": model, "host": host, "results": results}
    if output_json:
        directory = os.path.dirname(output_json)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2)
    return output


def get_label_call_count():
    return int(CALL_COUNT)
