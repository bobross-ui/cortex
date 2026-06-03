from __future__ import annotations

from abc import ABC, abstractmethod


class ChatClient(ABC):
    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        """Return the assistant's full text for these chat messages."""

    def stream(self, messages: list[dict]):
        """Yield assistant text chunks. Implementations may override for true streaming."""
        yield self.complete(messages)


class DeepSeekChatClient(ChatClient):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        *,
        disable_thinking: bool = True,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is not set")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_body = {"thinking": {"type": "disabled"}} if disable_thinking else None

    def complete(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=False,
            extra_body=self._extra_body,
        )
        return resp.choices[0].message.content or ""

    def stream(self, messages: list[dict]):
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
            extra_body=self._extra_body,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
