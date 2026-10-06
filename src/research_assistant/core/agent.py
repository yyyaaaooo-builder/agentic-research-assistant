import logging
from dataclasses import dataclass

from research_assistant.core.history import History
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import FINAL_ANSWER_PROMPT
from research_assistant.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the agent made during a run, and the result it got back."""

    name: str
    arguments: str
    result: str


class ReActAgent:
    """An agent that answers a question by alternating between reasoning and tool calls."""

    def __init__(self, llm: LLMClient, registry: ToolRegistry, max_steps: int, system_prompt: str) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be at least 1 so the agent can call tools, but received {max_steps!r}.")
        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._system_prompt = system_prompt
        self.tool_call_records: list[ToolCallRecord] = []

    def run(self, question: str) -> str:
        """Answer the question, calling tools for up to max_steps steps.

        Always returns an answer: if the step budget runs out, the agent answers
        from what it has gathered so far.
        """
        logger.info("Starting run: %s", question)
        self.tool_call_records = []
        tools = self._registry.to_openai_tools()

        history = History()
        history.add_system(self._system_prompt)
        history.add_user(question)

        for step in range(1, self._max_steps + 1):
            logger.info("Step %d/%d", step, self._max_steps)
            response = self._llm.generate(history.to_openai_messages(), tools=tools)
            message = response.choices[0].message

            if not message.tool_calls:
                logger.info("Answered at step %d", step)
                return message.content

            history.add_assistant(
                content=message.content,
                tool_calls=[call.model_dump() for call in message.tool_calls],
            )
            for call in message.tool_calls:
                name = call.function.name
                arguments = call.function.arguments
                logger.info("Tool call: %s(%s)", name, arguments)
                result = self._registry.invoke(name, arguments)
                history.add_tool_result(result, call.id)
                self.tool_call_records.append(ToolCallRecord(name, arguments, result))

        logger.warning("Used all %d steps; requesting a final answer without tools", self._max_steps)
        history.add_user(FINAL_ANSWER_PROMPT)
        response = self._llm.generate(history.to_openai_messages())
        return response.choices[0].message.content
