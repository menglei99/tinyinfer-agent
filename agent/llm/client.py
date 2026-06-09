"""LLM client 抽象。

支持 DeepSeek / OpenAI / Anthropic（可选）/ Mock。
Mock 模式让你在没 API key 时跑通整张 graph —— 测试和 demo 端到端时很有用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Protocol

from dotenv import load_dotenv

load_dotenv()


def _maybe_wrap_for_langsmith(openai_client):
    """同时满足两个条件时，把 openai client 包成 LangSmith span 捕获模式；
    否则原样返回。

      - `langsmith` 包能 import；AND
      - `LANGSMITH_TRACING=true` 且 `LANGSMITH_API_KEY` 已设。
    这里任何失败都静默 —— observability 绝不能让 pipeline 挂掉。
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


# ---------- 真 provider ----------


class OpenAICompatibleClient:
    """适用于 OpenAI、DeepSeek 和任意 OpenAI 兼容 endpoint。"""

    def __init__(self, *, api_key: str, base_url: Optional[str], model: str):
        # 懒 import：避免用户选 mock 时强依赖 openai
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        # 如果 LangSmith 配上了，把 client 包一下，chat completion 就会以正经
        # LLM span（含 prompt / response / token count）出现在 LangSmith dashboard。
        # langsmith 没装或 tracing env 没设时是 no-op。
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

        # 对 transient 连接错误做 retry。DashScope（和其他 endpoint）偶尔会
        # 在中途掐连接；没有 retry 的话一次抖动就毁掉 30 min 的 benchmark。
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
                # 只对连接类错误 retry；4xx / 鉴权错误立刻 surface，防止配置错误
                # 静默烧 quota。
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
        # Anthropic 没有原生 json_mode；在 prompt 里嘱咐一下
        if json_mode:
            user = user + "\n\n只用一个 JSON object 回答。不要散文，不要 code fence。"
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
    """按内容指纹返回确定性脚本响应。

    mock 认得几种 intent，返回看起来合理的结构化输出。够让完整 graph 跑起来，
    不需要打 API。
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
            # 返回一个占位 GTest body；numerical skill 反正会基于 numpy oracle
            # 确定性地构造真正的测试文件。
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
