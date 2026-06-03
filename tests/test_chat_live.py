from datetime import datetime, timezone

import pytest

from cortex.config import settings
from cortex.rag.chat import DeepSeekChatClient
from cortex.rag.prompt import build_messages, cited_indices
from cortex.rag.retriever import Source


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not settings.deepseek_api_key,
        reason="DEEPSEEK_API_KEY is required for live DeepSeek tests",
    ),
]


def test_live_deepseek_answer_is_non_empty_and_cites_source():
    client = DeepSeekChatClient(
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        settings.deepseek_model,
        settings.chat_temperature,
        settings.chat_max_tokens,
        disable_thinking=settings.deepseek_disable_thinking,
        timeout_s=settings.chat_timeout_s,
        max_retries=settings.chat_max_retries,
    )
    answer = client.complete(build_messages("What do they think about remote work?", _sources()))

    assert answer.strip()
    assert cited_indices(answer, max_index=1) == {1}


def test_live_deepseek_stream_is_non_empty_and_cites_source():
    client = DeepSeekChatClient(
        settings.deepseek_api_key,
        settings.deepseek_base_url,
        settings.deepseek_model,
        settings.chat_temperature,
        settings.chat_max_tokens,
        disable_thinking=settings.deepseek_disable_thinking,
        timeout_s=settings.chat_timeout_s,
        max_retries=settings.chat_max_retries,
    )

    answer = "".join(client.stream(build_messages("What do they think about remote work?", _sources())))

    assert answer.strip()
    assert cited_indices(answer, max_index=1) == {1}


def _sources():
    return [
        Source(
            index=1,
            document_id=1,
            external_id="1001",
            platform="twitter",
            content_type="thread",
            url="https://twitter.com/cortex_demo/status/1001",
            author_handle="cortex_demo",
            created_at=datetime(2024, 5, 15, tzinfo=timezone.utc),
            snippet="Remote work and async communication",
            context_text=(
                "The person says remote work can be excellent for deep focus when teams "
                "use async communication deliberately, but it can lose hallway serendipity "
                "and mentorship if teams do not design for those needs."
            ),
            score=0.9,
        )
    ]
