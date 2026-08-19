import json
import re

import torch

from .llm_client import OllamaClient


def normalize_rows(x, eps=1e-12):
    return x / x.sum(dim=1, keepdim=True).clamp_min(eps)


def parse_json(text):
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("No JSON object found")
    return json.loads(match.group(0))


class AnchorPriorBuilder:
    def __init__(self, vocab, idf=None):
        self.vocab = vocab
        self.word_to_id = {word: index for index, word in enumerate(vocab)}
        self.idf = idf or {}

    def build(self, top_words_per_topic, client: OllamaClient, anchors_per_topic=15, anti_anchors_per_topic=8, group_size=8, smoothing=1e-12, bonus=1.0, anti_penalty=0.0, fallback_on_error=True):
        topic_count = len(top_words_per_topic)
        vocab_size = len(self.vocab)
        prior = torch.full((topic_count, vocab_size), smoothing, dtype=torch.float32)
        for start in range(0, topic_count, group_size):
            end = min(start + group_size, topic_count)
            subset = top_words_per_topic[start:end]
            prompt = self._prompt(subset, anchors_per_topic, anti_anchors_per_topic)
            try:
                data = parse_json(client.generate(prompt, json_output=True, temperature=0.2))
                topics = data.get("topics", [])
                if len(topics) != len(subset):
                    raise ValueError("Topic count mismatch")
                for offset, item in enumerate(topics):
                    anchors = [word.strip() for word in item.get("anchors", []) if str(word).strip()]
                    anti_anchors = [word.strip() for word in item.get("anti_anchors", []) if str(word).strip()]
                    prior[start + offset] = self._weights(anchors, anti_anchors, vocab_size, smoothing, bonus, anti_penalty)
            except Exception:
                if not fallback_on_error:
                    raise
                for offset, words in enumerate(subset):
                    prior[start + offset] = self._weights(words[:anchors_per_topic], [], vocab_size, smoothing, bonus, anti_penalty)
        return normalize_rows(prior, smoothing)

    def _weights(self, anchors, anti_anchors, vocab_size, smoothing, bonus, anti_penalty):
        row = torch.full((vocab_size,), smoothing, dtype=torch.float32)
        for word in anchors:
            index = self.word_to_id.get(word)
            if index is not None:
                row[index] = max(float(row[index]), float(self.idf.get(word, 1.0)) * bonus)
        for word in anti_anchors:
            index = self.word_to_id.get(word)
            if index is not None:
                row[index] = min(float(row[index]), anti_penalty + smoothing)
        return row

    @staticmethod
    def _prompt(groups, anchors, anti_anchors):
        topic_lines = []
        for index, words in enumerate(groups):
            topic_lines.append(f"- topic_{index}: {', '.join(words[:15])}")
        block = "\n".join(topic_lines)
        return f"""你是主题建模助手。根据给出的主题关键词，为每个主题生成锚点词和反锚点词。
每个主题生成 {anchors} 个 anchors 和最多 {anti_anchors} 个 anti_anchors。
严格返回 JSON，结构为 {{"topics":[{{"anchors":[],"anti_anchors":[]}}]}}，顺序与输入一致，不输出额外文本。

{block}
""".strip()
