import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

from pydantic import BaseModel, Field, ValidationError

from research_assistant.core.agent import ReActAgent, RunResult
from research_assistant.core.ledger import (
    Finding,
    Ledger,
    cited_finding_ids,
    unbacked_citations,
    write_answer_from_ledger,
)
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import BRIEF_PROMPT

logger = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 3  # one worker per sub-question: caps parallel work and cost


class ResearchBrief(BaseModel):
    """Sub-questions and a shared scope that divide one research question among parallel workers."""

    sub_questions: list[str] = Field(
        description=(
            "Sub-questions independently answerable from arXiv papers that collectively cover the research question."
        ),
        min_length=1,
        json_schema_extra={"maxItems": MAX_SUB_QUESTIONS},
    )
    scope: str = Field(
        description="One sentence stating what the research covers and excludes, shared with every worker."
    )


@dataclass(frozen=True)
class ResearchResult:
    """Outcome of one supervised research run: the answer, the brief, each worker's run, and the merged findings."""

    question: str
    answer: str
    brief: ResearchBrief
    worker_results: tuple[RunResult, ...]
    findings: tuple[Finding, ...]
    open_questions: tuple[str, ...]
    unbacked_citations: tuple[str, ...]
    seconds: float


def write_brief(llm: LLMClient, question: str) -> ResearchBrief:
    """Split a research question into a brief for parallel workers, falling back to the question itself."""
    messages = [
        {"role": "system", "content": BRIEF_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    response = llm.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "research_brief", "schema": ResearchBrief.model_json_schema()},
        },
    )
    try:
        brief = ResearchBrief.model_validate_json(response.choices[0].message.content)
    except ValidationError as error:
        logger.warning(f"Brief failed validation, falling back to the original question: {error}")
        return ResearchBrief(sub_questions=[question], scope="")

    brief.sub_questions = brief.sub_questions[:MAX_SUB_QUESTIONS]
    logger.info(f"Brief: {len(brief.sub_questions)} sub-questions; scope: {brief.scope}")
    return brief


def format_worker_task(sub_question: str, scope: str) -> str:
    """Combine a sub-question with the shared scope into the question one worker researches."""
    return f"{sub_question}\n\nScope: {scope}" if scope else sub_question


def merge_worker_ledgers(worker_results: list[RunResult]) -> Ledger:
    """Combine the workers' findings and open questions into one ledger, renumbering findings so every ID is unique."""
    merged_ledger = Ledger()

    for index, result in enumerate(worker_results, start=1):
        for finding in result.findings:
            new_id = len(merged_ledger.findings) + 1
            merged_ledger.findings.append(replace(finding, id=new_id))

        merged_ledger.open_questions.extend(result.open_questions)
        logger.info(
            f"Merged worker {index}: {len(result.findings)} findings; ledger now has {len(merged_ledger.findings)}"
        )

    merged_ledger.open_questions = list(dict.fromkeys(merged_ledger.open_questions))  # drop repeats, keep order
    return merged_ledger


class Supervisor:
    """Coordinator that briefs a question, runs workers on its sub-questions in parallel, and collects their results."""

    def __init__(self, llm: LLMClient, worker: ReActAgent) -> None:
        self._llm = llm
        self._worker = worker

    def research(self, question: str) -> ResearchResult:
        """Research a question end to end: brief, parallel workers, merged ledger, and one cited answer."""
        started = time.perf_counter()
        brief = write_brief(self._llm, question)
        worker_results = self._research_sub_questions(brief)
        merged_ledger = merge_worker_ledgers(worker_results)
        answer, flagged = self._answer_from_merged_ledger(question, merged_ledger)
        return ResearchResult(
            question=question,
            answer=answer,
            brief=brief,
            worker_results=tuple(worker_results),
            findings=tuple(merged_ledger.findings),
            open_questions=tuple(merged_ledger.open_questions),
            unbacked_citations=tuple(flagged),
            seconds=time.perf_counter() - started,
        )

    def _research_sub_questions(self, brief: ResearchBrief) -> list[RunResult]:
        """Run one worker per sub-question in parallel and return the successful results in the brief's order."""
        with ThreadPoolExecutor(max_workers=len(brief.sub_questions)) as pool:
            pending_results = [
                pool.submit(self._research_one_sub_question, index, sub_question, brief)
                for index, sub_question in enumerate(brief.sub_questions, start=1)
            ]
            worker_results = [pending.result() for pending in pending_results]
            return [result for result in worker_results if result is not None]

    def _research_one_sub_question(self, index: int, sub_question: str, brief: ResearchBrief) -> RunResult | None:
        """Run the worker on one scoped sub-question, or return None if the worker fails."""
        logger.info(f"Worker {index}/{len(brief.sub_questions)}: {sub_question}")
        task = format_worker_task(sub_question, brief.scope)
        try:
            result = self._worker.run(task)
        except Exception:
            logger.exception(f"Worker {index}/{len(brief.sub_questions)} failed; continuing without its findings")
            return None

        logger.info(
            f"Worker {index}/{len(brief.sub_questions)} finished: {result.stop_reason}, {len(result.findings)} findings"
        )
        return result

    def _answer_from_merged_ledger(self, question: str, merged_ledger: Ledger) -> tuple[str, list[str]]:
        """Write one answer from the merged ledger, check its citations, and append its Sources block."""
        if not merged_ledger.findings:
            return "No answer: the workers found no evidence in the papers they read.", []

        response = write_answer_from_ledger(self._llm, question, merged_ledger)
        answer = response.choices[0].message.content or ""
        flagged = unbacked_citations(answer, merged_ledger)
        if flagged:
            logger.warning(f"Answer cites sources not in the merged ledger: {flagged}")
        answer += merged_ledger.render_sources(cited_finding_ids(answer))
        return answer, flagged
