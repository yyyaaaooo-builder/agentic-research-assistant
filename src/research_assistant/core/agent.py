import logging
import time
from dataclasses import dataclass
from typing import Literal

from research_assistant.core.history import History
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import FINAL_ANSWER_PROMPT, TOOL_BUDGET_EXHAUSTED
from research_assistant.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

StopReason = Literal["answered", "step_limit", "tool_call_limit", "time_limit"]


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the agent made during a run, and the result it got back."""

    name: str
    arguments: str
    result: str
    seconds: float


@dataclass(frozen=True)
class StepRecord:
    """One LLM call as a step: its timing, token usage, and the tool calls LLM made."""

    step: int
    llm_seconds: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: tuple[ToolCallRecord, ...]


@dataclass(frozen=True)
class RunResult:
    """Record metadata about one run: the question, the answer, why it stopped, and each step."""

    question: str
    answer: str
    stop_reason: StopReason
    steps: tuple[StepRecord, ...]
    seconds: float

    @property
    def peak_prompt_tokens(self) -> int:
        return max(step.prompt_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(step.prompt_tokens + step.completion_tokens for step in self.steps)

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)


def _step_record(step: int, llm_seconds: float, response, tool_calls: tuple[ToolCallRecord, ...]) -> StepRecord:
    """Build the record for one LLM call, treating missing usage data as zero tokens."""
    usage = response.usage
    return StepRecord(
        step=step,
        llm_seconds=llm_seconds,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        tool_calls=tool_calls,
    )


class ReActAgent:
    """An agent that answers a question by alternating between reasoning and tool calls."""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        max_steps: int,
        system_prompt: str,
        max_tool_calls: int,
        max_run_seconds: float,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps (LLM calls per run) must be at least 1, but received {max_steps!r}.")
        if max_tool_calls < 1:
            raise ValueError(
                f"max_tool_calls (tool calls per run) must be at least 1, but received {max_tool_calls!r}."
            )
        if max_run_seconds < 1:
            raise ValueError(f"max_run_seconds (time per run) must be at least 1, but received {max_run_seconds!r}.")

        self._llm = llm
        self._registry = registry
        self._max_steps = max_steps
        self._system_prompt = system_prompt
        self._max_tool_calls = max_tool_calls
        self._max_run_seconds = max_run_seconds

    def run(self, question: str) -> RunResult:
        """Answer the question and return the answer with a record of every step.

        Always returns an answer: if any budget runs out (steps, tool calls, or time), the
        agent makes one final call without tools and answers from what it has gathered so far.
        """
        logger.info(f"Starting run: {question}")
        run_started = time.perf_counter()
        tool_schemas = self._registry.to_openai_tools()
        history = self._create_history(question)

        steps: list[StepRecord] = []
        tool_calls_used = 0
        stop_reason: StopReason = "step_limit"  # kept if the loop uses every step
        answer = ""

        for step_number in range(1, self._max_steps + 1):
            logger.info(f"Step {step_number}/{self._max_steps}")

            # Time budget check: don't start another LLM call once the run is out of time.
            if time.perf_counter() - run_started >= self._max_run_seconds:
                stop_reason = "time_limit"
                break

            # Reason: the model decides to answer or to call tools.
            response, llm_seconds = self._call_llm(history, tool_schemas)
            reply = response.choices[0].message

            if not reply.tool_calls:
                steps.append(_step_record(step_number, llm_seconds, response, ()))
                stop_reason = "answered"
                answer = reply.content
                break

            # Act: run the requested tools and feed their results back.
            calls_left = self._max_tool_calls - tool_calls_used
            tool_records = self._execute_tool_requests(history, reply, calls_left)
            tool_calls_used += len(tool_records)
            steps.append(_step_record(step_number, llm_seconds, response, tuple(tool_records)))

            # Tool call budget check: stop once the budget is used up.
            if tool_calls_used >= self._max_tool_calls:
                stop_reason = "tool_call_limit"
                break

        if stop_reason != "answered":
            answer = self._request_final_answer(history, steps, stop_reason)

        run_seconds = time.perf_counter() - run_started
        logger.info(f"Run finished: {stop_reason} after {len(steps)} steps in {run_seconds:.1f}s")
        return RunResult(
            question=question,
            answer=answer,
            stop_reason=stop_reason,
            steps=tuple(steps),
            seconds=run_seconds,
        )

    def _create_history(self, question: str) -> History:
        """Create a fresh history holding the system prompt and the question."""
        history = History()
        history.add_system(self._system_prompt)
        history.add_user(question)
        return history

    def _call_llm(self, history: History, tool_schemas: list[dict] | None = None):
        """Send the history to the model; return its response and how long the call took."""
        started = time.perf_counter()
        response = self._llm.generate(history.to_openai_messages(), tools=tool_schemas)
        return response, time.perf_counter() - started

    def _invoke_tool(self, tool_call) -> ToolCallRecord:
        """Run one tool call through the registry; record its arguments, result, and duration."""
        name, arguments = tool_call.function.name, tool_call.function.arguments
        logger.info(f"Tool call: {name}({arguments})")
        started = time.perf_counter()
        result = self._registry.invoke(name, arguments)
        return ToolCallRecord(name, arguments, result, time.perf_counter() - started)

    def _execute_tool_requests(self, history: History, reply, calls_left: int) -> list[ToolCallRecord]:
        """Run every tool call in the model's reply, up to calls_left, and add all results to history.

        Calls beyond calls_left are not run but still get a result, because the API needs
        one result per tool call. Returns records for the calls that actually ran.
        """
        history.add_assistant(
            content=reply.content,
            tool_calls=[tool_call.model_dump() for tool_call in reply.tool_calls],
        )
        records: list[ToolCallRecord] = []
        for tool_call in reply.tool_calls:
            if len(records) < calls_left:
                record = self._invoke_tool(tool_call)
                records.append(record)
                result = record.result
            else:
                result = TOOL_BUDGET_EXHAUSTED
            history.add_tool_result(result, tool_call.id)
        return records

    def _request_final_answer(self, history: History, steps: list[StepRecord], stop_reason: StopReason) -> str:
        """Ask for an answer without tools after a budget runs out; record that call as a step."""
        logger.warning(f"Stopped by {stop_reason}; requesting a final answer without tools")
        history.add_user(FINAL_ANSWER_PROMPT)
        response, llm_seconds = self._call_llm(history)
        steps.append(_step_record(len(steps) + 1, llm_seconds, response, ()))
        return response.choices[0].message.content
