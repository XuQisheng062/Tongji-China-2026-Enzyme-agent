from __future__ import annotations

import json
import os
from typing import Any


class DeepSeekError(RuntimeError):
    pass


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client.

    API keys are accepted at runtime (CLI argument or DEEPSEEK_API_KEY) and are
    never written into the project config/output files by this class.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise DeepSeekError(
                "\u672a\u63d0\u4f9b DeepSeek API key\u3002\u8bf7\u4f7f\u7528\u547d\u4ee4\u884c --api-key\uff0c"
                "\u6216\u8bbe\u7f6e\u73af\u5883\u53d8\u91cf DEEPSEEK_API_KEY\u3002"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepSeekError(
                "\u4e3b\u73af\u5883\u7f3a\u5c11 openai SDK\uff0c\u8bf7\u6267\u884c: python -m pip install openai"
            ) from exc

        self.client = OpenAI(api_key=key, base_url=base_url)

    def text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        thinking: bool = True,
        reasoning_effort: str = "high",
    ) -> str:
        if reasoning_effort not in {"high", "max"}:
            raise DeepSeekError("reasoning_effort \u53ea\u80fd\u662f high/max")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        def _call(use_thinking: bool):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "stream": False,
                "extra_body": {
                    "thinking": {"type": "enabled" if use_thinking else "disabled"}
                },
            }
            if use_thinking:
                kwargs["reasoning_effort"] = reasoning_effort
            return self.client.chat.completions.create(**kwargs)

        response = _call(thinking)
        if not getattr(response, "choices", None):
            raise DeepSeekError("DeepSeek \u8fd4\u56de\u7a7a choices")

        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        finish_reason = getattr(choice, "finish_reason", None)
        if content and content.strip():
            return content.strip()

        print(
            "[WARN] DeepSeek \u7b2c\u4e00\u6b21\u8fd4\u56de\u7a7a content: "
            f"model={model}, finish_reason={finish_reason}, "
            f"reasoning_chars={len(reasoning_content or '')}"
        )

        # Thinking can consume the response budget before a final answer is
        # emitted. Retry once with thinking disabled, preserving the same prompt.
        if thinking:
            print("[WARN] \u81ea\u52a8\u5173\u95ed thinking\uff0c\u91cd\u8bd5\u6700\u7ec8\u56de\u7b54...")
            retry = _call(False)
            if not getattr(retry, "choices", None):
                raise DeepSeekError("DeepSeek \u91cd\u8bd5\u540e\u4ecd\u8fd4\u56de\u7a7a choices")
            retry_choice = retry.choices[0]
            retry_content = getattr(retry_choice.message, "content", None)
            retry_finish = getattr(retry_choice, "finish_reason", None)
            if retry_content and retry_content.strip():
                return retry_content.strip()
            raise DeepSeekError(
                "DeepSeek \u91cd\u8bd5\u540e\u4ecd\u8fd4\u56de\u7a7a\u5185\u5bb9\uff1b"
                f"finish_reason={retry_finish}"
            )

        raise DeepSeekError(
            "DeepSeek \u8fd4\u56de\u7a7a\u5185\u5bb9\uff1b"
            f"finish_reason={finish_reason}"
        )

    def json_object(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        attempts: int = 2,
    ) -> dict[str, Any]:
        """Request strict JSON output with a small fixed retry count."""
        last_error: Exception | None = None
        for _ in range(max(1, int(attempts))):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=int(max_tokens),
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                if not getattr(response, "choices", None):
                    raise DeepSeekError("DeepSeek JSON Output \u8fd4\u56de\u7a7a choices")
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise DeepSeekError("DeepSeek JSON Output \u8fd4\u56de\u7a7a\u5185\u5bb9")
                parsed = json.loads(content)
                if not isinstance(parsed, dict):
                    raise DeepSeekError("DeepSeek JSON Output \u9876\u5c42\u5fc5\u987b\u662f object")
                return parsed
            except Exception as exc:  # preserve upstream API/JSON details
                last_error = exc
        raise DeepSeekError(f"DeepSeek JSON Output \u5931\u8d25: {last_error}") from last_error
