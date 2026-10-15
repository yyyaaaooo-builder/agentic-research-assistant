import json
import logging
import time
from dataclasses import dataclass, replace
from typing import Literal

from research_assistant.core.history import History
from research_assistant.core.ledger import (
    Finding,
    Ledger,
    cited_finding_ids,
    extract,
    format_citation,
    unbacked_citations,
    write_answer_from_ledger,
)
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import FINAL_ANSWER_PROMPT, TOOL_BUDGET_EXHAUSTED
from research_assistant.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

StopReason = Literal["answered", "step_limit", "tool_call_limit", "time_limit"]


@dataclass(frozen=True)
class ToolCallRecord:
    """One tool call the agent made during a run: what the tool returned, and what the history kept."""

    name: str
    arguments: str
    tool_result: str
    seconds: float
    history_result: str | None = None


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
    findings: tuple[Finding, ...] = ()
    open_questions: tuple[str, ...] = ()
    unbacked_citations: tuple[str, ...] = ()

    @property
    def peak_prompt_tokens(self) -> int:
        return max(step.prompt_tokens for step in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(step.prompt_tokens + step.completion_tokens for step in self.steps)

    @property
    def tool_call_count(self) -> int:
        return sum(len(step.tool_calls) for step in self.steps)

    @property
    def estimated_peak_without_ledger(self) -> int:
        """The estimated peak prompt size had raw paper text stayed in history instead of being replaced."""
        peak = 0
        kept_out = 0
        for step in self.steps:
            peak = max(peak, step.prompt_tokens + kept_out // 4)
            kept_out += sum(
                len(call.tool_result) - len(call.history_result)
                for call in step.tool_calls
                if call.history_result is not None  # only calls whose result was replaced
            )
        return peak


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
        note_source: dict[str, tuple[str, str]] | None = None,
        extraction_llm: LLMClient | None = None,
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
        self._note_source = note_source or {}
        self._extraction_llm = extraction_llm or llm

    def run(self, question: str) -> RunResult:
        """Answer the question and return the answer with a record of every step.

        Always returns an answer: if any budget runs out (steps, tool calls, or time), the
        agent makes one final call without tools and answers from what it has gathered so far.
        """
        logger.info(f"Starting run: {question}")
        run_started = time.perf_counter()
        tool_schemas = self._registry.to_openai_tools()
        history = self._create_history(question)
        ledger = Ledger()

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
            response, llm_seconds = self._call_llm(history, tool_schemas, ledger)
            reply = response.choices[0].message

            if not reply.tool_calls:
                steps.append(_step_record(step_number, llm_seconds, response, ()))
                stop_reason = "answered"
                answer = reply.content
                break

            # Act: run the requested tools and feed their results back.
            calls_left = self._max_tool_calls - tool_calls_used
            tool_records = self._execute_tool_requests(history, question, ledger, reply, calls_left)
            tool_calls_used += len(tool_records)
            steps.append(_step_record(step_number, llm_seconds, response, tuple(tool_records)))

            # Tool call budget check: stop once the budget is used up.
            if tool_calls_used >= self._max_tool_calls:
                stop_reason = "tool_call_limit"
                break

        if ledger.findings:
            answer = self._answer_from_ledger(question, ledger, steps)
        elif stop_reason != "answered":
            answer = self._request_final_answer(history, steps, stop_reason)

        flagged = unbacked_citations(answer, ledger)
        if flagged:
            logger.warning(f"Answer cites sources not in the ledger: {flagged}")
        answer += ledger.render_sources(cited_finding_ids(answer))

        run_seconds = time.perf_counter() - run_started
        logger.info(f"Run finished: {stop_reason} after {len(steps)} steps in {run_seconds:.1f}s")
        return RunResult(
            question=question,
            answer=answer,
            stop_reason=stop_reason,
            steps=tuple(steps),
            seconds=run_seconds,
            findings=tuple(ledger.findings),
            open_questions=tuple(ledger.open_questions),
            unbacked_citations=tuple(flagged),
        )

    def _create_history(self, question: str) -> History:
        """Create a fresh history holding the system prompt and the question."""
        history = History()
        history.add_system(self._system_prompt)
        history.add_user(question)
        return history

    def _call_llm(self, history: History, tool_schemas: list[dict] | None = None, ledger: Ledger | None = None):
        """Send the history and current notes to the model; return its response and how long the call took.

        The notes are added to this call only and are never saved to the history.
        """
        messages = history.to_openai_messages()
        if ledger and ledger.findings:
            messages.append({"role": "user", "content": ledger.render()})
        started = time.perf_counter()
        response = self._llm.generate(messages, tools=tool_schemas)
        return response, time.perf_counter() - started

    def _execute_tool_requests(
        self,
        history: History,
        question: str,
        ledger: Ledger,
        reply,
        calls_left: int,
    ) -> list[ToolCallRecord]:
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
                result = self._capture_notes(question, ledger, tool_call, record.tool_result)
                if result != record.tool_result:
                    record = replace(record, history_result=result)
                records.append(record)
            else:
                result = TOOL_BUDGET_EXHAUSTED
            history.add_tool_result(result, tool_call.id)
        return records

    def _invoke_tool(self, tool_call) -> ToolCallRecord:
        """Run one tool call through the registry; record its arguments, result, and duration."""
        name, arguments = tool_call.function.name, tool_call.function.arguments
        logger.info(f"Tool call: {name}({arguments})")
        started = time.perf_counter()
        result = self._registry.invoke(name, arguments)
        return ToolCallRecord(name=name, arguments=arguments, tool_result=result, seconds=time.perf_counter() - started)

    def _capture_notes(self, question: str, ledger: Ledger, tool_call, result: str) -> str:
        """Turn a paper read into ledger notes and return the receipt that replaces its text in history.

        Returns the result unchanged when the call is not a paper read or the extraction fails.
        """
        name = tool_call.function.name
        if name not in self._note_source:
            return result

        if result.startswith("Error:"):
            return result

        paper_key, section_key = self._note_source[name]
        arguments = json.loads(tool_call.function.arguments)
        paper_id = arguments.get(paper_key, "")
        section = str(arguments.get(section_key, ""))

        extraction = extract(self._extraction_llm, question, ledger, result)
        if extraction is None:
            return result

        changes = ledger.add(extraction, paper_id, section)
        summary = " · ".join(changes) or "no new findings"
        return f"{format_citation(paper_id, section)} → {summary}. Saved to notes; full text not kept."

    def _answer_from_ledger(self, question: str, ledger: Ledger, steps: list[StepRecord]) -> str:
        """Write the final answer from the ledger and record that model call as a step."""
        started = time.perf_counter()
        response = write_answer_from_ledger(self._llm, question, ledger)
        llm_seconds = time.perf_counter() - started
        steps.append(_step_record(len(steps) + 1, llm_seconds, response, ()))
        return response.choices[0].message.content or ""

    def _request_final_answer(self, history: History, steps: list[StepRecord], stop_reason: StopReason) -> str:
        """Ask for an answer without tools after a budget runs out; record that call as a step."""
        logger.warning(f"Stopped by {stop_reason}; requesting a final answer without tools")
        history.add_user(FINAL_ANSWER_PROMPT)
        response, llm_seconds = self._call_llm(history)
        steps.append(_step_record(len(steps) + 1, llm_seconds, response, ()))
        return response.choices[0].message.content
