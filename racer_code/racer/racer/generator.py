"""Answer generators.  ExtractiveGenerator is deterministic (no LLM); LLM backends use urllib only."""
from __future__ import annotations

import json
import urllib.request


def build_prompt(tier: str, products: list[dict], clauses: list[dict], feedback: str = "") -> str:
    lines = [f"You are a financial-advisory assistant. Client risk tier: {tier}.",
             "Recommend ONLY from the products listed. Cite products as [P#] and regulatory clauses as [R#].",
             "Use only numbers that appear in the context. Do not invent products or clauses.", "", "PRODUCTS:"]
    for i, p in enumerate(products, 1):
        lines.append(f"[P{i}] {p['name']} | category={p.get('category')} | risk_level={p.get('risk_level')} "
                     f"| expense_ratio={p.get('expense_ratio')} | rating={p.get('rating')} | sharpe={p.get('sharpe')}")
    lines += ["", "REGULATORY CLAUSES:"]
    for i, c in enumerate(clauses, 1):
        lines.append(f"[R{i}] ({c.get('source','')}) {c['text']}")
    if feedback:
        lines += ["", "YOUR PREVIOUS ANSWER WAS REJECTED: " + feedback + " Fix this."]
    return "\n".join(lines)


class ExtractiveGenerator:
    """Template answer built only from retrieved context -> grounded by construction."""
    name = "extractive"

    def generate(self, prompt: str, ctx: dict) -> str:
        parts = []
        for i, p in enumerate(ctx["products"], 1):
            parts.append(f"{p['name']} [P{i}] (risk level {p.get('risk_level'):g}, expense ratio "
                         f"{p.get('expense_ratio')}) fits a {ctx['tier']} risk profile.")
        for i, c in enumerate(ctx["clauses"][:2], 1):
            parts.append(f"Relevant requirement [R{i}]: {c['text'][:200]}")
        return " ".join(parts) or "No suitable product found."


class ScriptedBackend:
    """Returns pre-set responses in order (tests / regeneration-loop demos)."""
    name = "scripted"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def generate(self, prompt: str, ctx: dict) -> str:
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


class OpenAICompatBackend:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, Ollama, ...)."""

    def __init__(self, base_url, api_key, model, temperature=0.0, timeout=120):
        self.url, self.key, self.model = base_url.rstrip("/") + "/chat/completions", api_key, model
        self.temperature, self.timeout, self.name = temperature, timeout, f"openai:{model}"

    def generate(self, prompt: str, ctx: dict) -> str:
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json",
                                                      "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]


class AnthropicBackend:
    # temperature=None omits the field; current Claude models reject it.
    def __init__(self, api_key, model, max_tokens=600, temperature=None, timeout=120,
                 base_url="https://api.anthropic.com"):
        self.url, self.key, self.model = base_url.rstrip("/") + "/v1/messages", api_key, model
        self.max_tokens, self.temperature, self.timeout, self.name = max_tokens, temperature, timeout, f"anthropic:{model}"

    def generate(self, prompt: str, ctx: dict) -> str:
        payload = {"model": self.model, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json", "x-api-key": self.key,
                                                      "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return "".join(b.get("text", "") for b in json.load(r)["content"])
