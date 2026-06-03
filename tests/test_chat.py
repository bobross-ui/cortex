import sys
from types import ModuleType, SimpleNamespace

import pytest

from cortex.rag.chat import DeepSeekChatClient


def test_deepseek_chat_client_requires_api_key():
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY is not set"):
        DeepSeekChatClient(
            "",
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            0.2,
            800,
        )


def test_deepseek_chat_client_calls_openai_compatible_api_with_thinking_disabled(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    FakeOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    client = DeepSeekChatClient(
        "key",
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        0.2,
        800,
        disable_thinking=True,
        timeout_s=60.0,
        max_retries=2,
    )
    messages = [{"role": "user", "content": "hello"}]

    assert client.complete(messages) == "grounded answer [1]"

    fake = FakeOpenAI.instances[0]
    assert fake.init_kwargs == {
        "api_key": "key",
        "base_url": "https://api.deepseek.com",
        "timeout": 60.0,
        "max_retries": 2,
    }
    assert fake.create_kwargs == {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_deepseek_chat_client_can_leave_thinking_default(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    FakeOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    client = DeepSeekChatClient(
        "key",
        "https://api.deepseek.com",
        "deepseek-v4-pro",
        0.2,
        800,
        disable_thinking=False,
    )
    client.complete([{"role": "user", "content": "hello"}])

    assert FakeOpenAI.instances[0].create_kwargs["extra_body"] is None


def test_deepseek_chat_client_streams_openai_compatible_chunks(monkeypatch):
    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    FakeOpenAI.instances = []
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    client = DeepSeekChatClient(
        "key",
        "https://api.deepseek.com",
        "deepseek-v4-flash",
        0.2,
        800,
    )
    messages = [{"role": "user", "content": "hello"}]

    assert list(client.stream(messages)) == ["grounded ", "answer [1]"]

    assert FakeOpenAI.instances[0].create_kwargs == {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800,
        "stream": True,
        "extra_body": {"thinking": {"type": "disabled"}},
    }


class FakeOpenAI:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.create_kwargs = None
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create),
        )
        self.instances.append(self)

    def _create(self, **kwargs):
        self.create_kwargs = kwargs
        if kwargs.get("stream"):
            return [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="grounded "),
                        ),
                    ],
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None),
                        ),
                    ],
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="answer [1]"),
                        ),
                    ],
                ),
            ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="grounded answer [1]"),
                ),
            ],
        )
