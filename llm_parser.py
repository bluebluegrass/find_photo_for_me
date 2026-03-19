"""Local LLM-based query parsing for smart search intent extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from utils import default_llm_endpoint, default_llm_model, default_llm_timeout, normalize_location_text

LOGGER = logging.getLogger(__name__)


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = item.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


@dataclass(slots=True)
class QueryIntent:
    """Normalized parse result for one user query."""

    raw_query: str
    visual_query: str
    objects: list[str]
    attributes: list[str]
    location_text: str | None
    normalized_location_text: str | None
    time_text: str | None
    expanded_queries: list[str]
    parse_mode: str = "fallback"


class OllamaClient:
    """Minimal local Ollama HTTP client (offline/local use)."""

    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_sec: float = 2.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec

    def generate(self, prompt: str) -> str:
        """Return text response from local Ollama model."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        req = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_sec) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return str(parsed.get("response", "")).strip()


class SmartQueryParser:
    """Parse raw natural language query into structured search intent."""

    def __init__(
        self,
        model: str | None = None,
        timeout_sec: float | None = None,
        endpoint: str | None = None,
    ) -> None:
        model_name = model or default_llm_model()
        timeout = timeout_sec if timeout_sec is not None else default_llm_timeout()
        url = endpoint or default_llm_endpoint()
        self.client = OllamaClient(model=model_name, endpoint=url, timeout_sec=timeout)

    def parse(self, query: str) -> QueryIntent:
        """Return structured query intent with robust fallback behavior."""
        raw = query.strip()
        if not raw:
            return QueryIntent(
                raw_query=query,
                visual_query="",
                objects=[],
                attributes=[],
                location_text=None,
                normalized_location_text=None,
                time_text=None,
                expanded_queries=[],
                parse_mode="fallback",
            )

        prompt = self._build_prompt(raw)
        try:
            response_text = self.client.generate(prompt)
            intent = self._normalize_llm_response(raw, response_text)
            if intent is not None:
                return intent
        except (TimeoutError, URLError, OSError, ValueError) as exc:
            LOGGER.info("LLM parser unavailable, using fallback parse: %s", exc)
        except Exception:
            LOGGER.exception("Unexpected smart-query parse failure; falling back.")

        return self._fallback_parse(raw)

    def _build_prompt(self, query: str) -> str:
        return (
            "You are a query parser for a local photo/video search app. "
            "Return ONLY valid compact JSON with this schema: "
            "{\"visual_query\": string, \"objects\": string[], \"attributes\": string[], "
            "\"location_text\": string|null, \"time_text\": string|null, "
            "\"expanded_queries\": string[]}. "
            "No markdown, no commentary. "
            f"Query: {query}"
        )

    def _normalize_llm_response(self, raw_query: str, response_text: str) -> QueryIntent | None:
        text = response_text.strip()
        if not text:
            return None

        # Some models may wrap JSON in text; extract first object block if needed.
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or start >= end:
                return None
            text = text[start : end + 1]

        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        visual_query = str(data.get("visual_query") or raw_query).strip()
        objects = _coerce_string_list(data.get("objects"))
        attributes = _coerce_string_list(data.get("attributes"))
        location_text = _coerce_optional_str(data.get("location_text"))
        time_text = _coerce_optional_str(data.get("time_text"))
        expanded = _coerce_string_list(data.get("expanded_queries"))

        if not expanded:
            expanded = _default_expansions(visual_query)

        return QueryIntent(
            raw_query=raw_query,
            visual_query=visual_query,
            objects=_dedupe_preserve(objects),
            attributes=_dedupe_preserve(attributes),
            location_text=location_text,
            normalized_location_text=normalize_location_text(location_text),
            time_text=time_text,
            expanded_queries=_dedupe_preserve(expanded),
            parse_mode="llm",
        )

    def _fallback_parse(self, raw_query: str) -> QueryIntent:
        location_text = None
        visual_query = raw_query

        # Lightweight heuristic for "<visual> in <location>" patterns.
        m = re.match(r"^(.+?)\s+in\s+([a-zA-Z][a-zA-Z\s'\-]+)$", raw_query, flags=re.IGNORECASE)
        if m:
            visual_query = m.group(1).strip()
            location_text = m.group(2).strip()

        tokens = [t for t in re.split(r"\s+", visual_query) if t]
        objects = [tokens[-1]] if tokens else []
        attributes = tokens[:-1] if len(tokens) > 1 else []

        return QueryIntent(
            raw_query=raw_query,
            visual_query=visual_query,
            objects=_dedupe_preserve(objects),
            attributes=_dedupe_preserve(attributes),
            location_text=location_text,
            normalized_location_text=normalize_location_text(location_text),
            time_text=None,
            expanded_queries=_default_expansions(visual_query),
            parse_mode="fallback",
        )


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
        return out
    return []


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _default_expansions(visual_query: str) -> list[str]:
    v = visual_query.strip()
    if not v:
        return []
    return [
        v,
        f"a photo of {v}",
        f"a picture of {v}",
        f"an image of {v}",
    ]
