from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from src.pcg.certificates import build_certificate


class Prover:
    def __init__(self, model_name: str, device_map: str = "auto"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.config = AutoConfig.from_pretrained(model_name)

        if getattr(self.config, "is_encoder_decoder", False):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = "seq2seq"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = "causal"

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str, choices: list[str]) -> tuple[str, str]:
        choice_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(choices)])

        prompt = (
            "Answer the following medical multiple choice question.\n"
            f"Question: {question}\n"
            f"Choices:\n{choice_block}\n"
            "Return only the best answer text."
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        if self.model_type == "seq2seq":
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        else:
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[0][input_len:]
            completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return prompt, completion

    def make_certificate(self, claim: str, evidence_texts: list[str], meta: dict):
        return build_certificate(claim=claim, evidence_texts=evidence_texts, meta=meta)
