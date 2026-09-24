import pytest

from research_assistant.config import Config
from research_assistant.llm.client import LLMClient


def test_generate_returns_content():
    try:
        config = Config()
    except Exception as e:
        pytest.skip(f"no .env / API key configured: {e}")
    client = LLMClient(config)
    response = client.generate([{"role": "user", "content": "whats the capital of China"}])
    assert response.choices[0].message.content
