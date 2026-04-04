from __future__ import annotations

from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TokenCounter:
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def count_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def usage(self, prompt: str, completion: str) -> TokenUsage:
        pt = self.count_text(prompt)
        ct = self.count_text(completion)
        return TokenUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
        )
