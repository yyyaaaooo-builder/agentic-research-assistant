import logging
import time

import openai

from research_assistant.config import Config

logger = logging.getLogger(__name__)


class EmptyModelReplyError(Exception):
    """The API returned a response with no reply, e.g. when the model's provider failed behind OpenRouter."""


class LLMClient:
    def __init__(self, config: Config, max_retries: int = 3, model: str | None = None):
        self._client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)
        self._model = model or config.model_id
        self._max_retries = max_retries

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Send one turn of the current agent run to the LLM and return its raw response.
        Retries on transient failures (timeout, rate limit, connection error)
        with exponential backoff, up to `max_retries` attempts, before raising.

        param: messages: OpenAI-format list of message dicts covering every turn
            of the current `agent.run()` call so far.
        param: tools: optional list of OpenAI-format tool schemas currently
            available for the model to call this turn.
        param: response_format: optional structured-output format, e.g. a JSON schema.
        param: max_tokens: optional cap on the length of this one reply.
        return: the raw ChatCompletion response from the API
        """
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    response_format=response_format,
                    max_tokens=max_tokens,
                )
                if not response.choices:
                    raise EmptyModelReplyError(f"model returned no reply: {getattr(response, 'error', None)}")
                usage = response.usage
                if usage:
                    logger.info(f"prompt token: {usage.prompt_tokens}, response token: {usage.completion_tokens}.")
                else:
                    logger.info("Model reply had no usage data; token counts unavailable for this call.")
                return response
            except (
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.APIConnectionError,
                EmptyModelReplyError,
            ) as e:
                if attempt == self._max_retries - 1:
                    logger.error(f"LLM call failed after {self._max_retries} attempts: {e}")
                    raise
                backoff_seconds = 2**attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries} failed: {e}. Retrying in {backoff_seconds}s."
                )
                time.sleep(backoff_seconds)
