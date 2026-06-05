"""LLM client abstraction.

Supports DeepSeek / OpenAI / Anthropic (optional) / Mock.
Mock mode lets you run the full graph without any API key — useful for tests
and for demoing the project end-to-end.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from dotenv import load_dotenv

load_dotenv()


def _maybe_wrap_for_langsmith(openai_client):
    """Return the openai client wrapped for LangSmith span capture, or as-is.

    Wraps only when both:
      - the `langsmith` package is importable, AND
      - `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set.
    Failures here are silent — observability must never break the pipeline.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return openai_client
    if os.getenv("LANGSMITH_TRACING", "").lower() not in ("1", "true", "yes"):
        return openai_client
    try:
        from langsmith.wrappers import wrap_openai  # type: ignore

        return wrap_openai(openai_client)
    except Exception:
        return openai_client


@dataclass
class LLMResponse:
    text: str
    raw: dict | None = None


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse: ...


# ---------- real providers ----------


class OpenAICompatibleClient:
    """Works for OpenAI, DeepSeek, and any OpenAI-compatible endpoint."""

    def __init__(self, *, api_key: str, base_url: Optional[str], model: str):
        # Lazy import: avoid hard dep if user picks mock.
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        # If LangSmith is configured, wrap the client so chat completions appear
        # as proper LLM spans (with prompt + response + token counts) in the
        # LangSmith dashboard. No-op when langsmith isn't installed or the
        # tracing env vars aren't set.
        self._client = _maybe_wrap_for_langsmith(client)
        self._model = model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        kwargs = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Retry transient connection errors. DashScope (and other endpoints)
        # occasionally drop a connection mid-batch; without retry a single
        # blip kills a 30-minute benchmark run.
        import time

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return LLMResponse(
                    text=text,
                    raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                )
            except Exception as exc:
                # Only retry on connection-flavoured errors; surface 4xx/auth
                # immediately so misconfigurations don't quietly burn quota.
                name = type(exc).__name__
                if name not in (
                    "APIConnectionError",
                    "APITimeoutError",
                    "ReadTimeout",
                    "ConnectionError",
                    "TimeoutError",
                    "InternalServerError",
                ):
                    raise
                last_exc = exc
                if attempt == 2:
                    break
                time.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc


class AnthropicClient:
    def __init__(self, *, api_key: str, model: str):
        from anthropic import Anthropic  # type: ignore

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        # Anthropic has no native json_mode; instruct via prompt.
        if json_mode:
            user = user + "\n\nRespond with a single JSON object. No prose, no code fences."
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(text=text)


# ---------- mock provider ----------


class MockClient:
    """Deterministic scripted responses keyed by content fingerprint.

    The mock recognises a small set of intents and returns plausible structured
    output. Sufficient to walk the full graph without hitting an API.
    """

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> LLMResponse:
        text_l = (system + "\n" + user).lower()

        if "parse" in text_l and "diff" in text_l:
            payload = {
                "changed_ops": [
                    {
                        "name": "matmul_fp32",
                        "file_path": "tinyinfer/src/matmul.cpp",
                        "summary": "modified inner accumulation loop",
                    }
                ]
            }
            return LLMResponse(text=json.dumps(payload))

        if "route" in text_l or "select" in text_l and "skill" in text_l:
            return LLMResponse(text=json.dumps({"skills": ["numerical"]}))

        if "critic" in text_l or "review" in text_l:
            return LLMResponse(
                text=json.dumps(
                    {
                        "passed": True,
                        "missing_dimensions": [],
                        "feedback": "Coverage acceptable for MVP (mock).",
                    }
                )
            )

        if "generate" in text_l and "test" in text_l:
            # Return a placeholder GTest body; the numerical skill builds the
            # real test file deterministically from numpy oracle anyway.
            return LLMResponse(
                text=json.dumps(
                    {
                        "rationale": "mock: cover small square + rectangular shapes",
                        "shapes": [
                            {"m": 2, "k": 2, "n": 2},
                            {"m": 3, "k": 4, "n": 2},
                            {"m": 1, "k": 5, "n": 3},
                        ],
                    }
                )
            )

        return LLMResponse(text="{}")


# ---------- factory ----------


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "mock").lower().strip()

    if provider == "mock":
        return MockClient()

    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            return MockClient()
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return MockClient()
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=None,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return MockClient()
        return AnthropicClient(
            api_key=api_key,
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        )

    if provider == "glm":
        api_key = os.getenv("GLM_API_KEY", "")
        if not api_key:
            return MockClient()
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            model=os.getenv("GLM_MODEL", "glm-4-plus"),
        )

    if provider in ("qwen", "dashscope"):
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            return MockClient()
        return OpenAICompatibleClient(
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        )

    return MockClient()
