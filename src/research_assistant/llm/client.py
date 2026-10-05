import logging
import time

import openai

from research_assistant.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: Config, max_retries: int = 3):
        self._client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)
        self._model = config.model_id
        self._max_retries = max_retries

    def generate(self, messages: list[dict], tools: list[dict] | None = None):
        """
        Send one turn of the current agent run to the LLM and return its raw response.
        Retries on transient failures (timeout, rate limit, connection error)
        with exponential backoff, up to `max_retries` attempts, before raising.

        param: messages: OpenAI-format list of message dicts covering every turn
            of the current `agent.run()` call so far.
        param: tools: optional list of OpenAI-format tool schemas currently
            available for the model to call this turn.
        return: the raw ChatCompletion response from the API
        """
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(model=self._model, messages=messages, tools=tools)
                logger.info(
                    f"prompt token: {response.usage.prompt_tokens}, response token: {response.usage.completion_tokens}."
                )
                return response
            except (openai.APITimeoutError, openai.RateLimitError, openai.APIConnectionError) as e:
                if attempt == self._max_retries - 1:
                    logger.error(f"LLM call failed after {self._max_retries} attempts: {e}")
                    raise
                backoff_seconds = 2**attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying in {backoff_seconds}s."
                )
                time.sleep(backoff_seconds)
