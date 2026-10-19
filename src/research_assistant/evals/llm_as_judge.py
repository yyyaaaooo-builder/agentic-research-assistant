import logging
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from research_assistant.core.ledger import cited_finding_ids
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import JUDGE_COMPARISON_PROMPT, JUDGE_FAITHFULNESS_PROMPT

logger = logging.getLogger(__name__)

Preference = Literal["A", "B", "tie"]
SOURCES_HEADING = "**Sources**"
PAIRWISE_METRICS = ("completeness", "specificity")


class RecordingLLMClient:
    """LLM client wrapper that records the verdict type, tokens, and latency of every judge call it makes."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.calls: list[dict] = []

    def generate(self, messages: list[dict], **options):
        """Send the request through the wrapped client and record the call."""
        started = time.perf_counter()
        response = self._llm.generate(messages, **options)
        usage = response.usage
        self.calls.append(
            {
                "verdict": options["response_format"]["json_schema"]["name"],
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "seconds": time.perf_counter() - started,
            }
        )
        return response


class ComparisonVerdict(BaseModel):
    """Judge's choice of the stronger answer on completeness and specificity, with a reason for each."""

    completeness_winner: Preference = Field(description="Which answer better covers the question's key requirements.")
    completeness_reason: str = Field(description="One sentence identifying the most important difference in coverage.")
    specificity_winner: Preference = Field(description="Which answer makes more concrete, verifiable claims.")
    specificity_reason: str = Field(
        description="One sentence identifying the most important difference in specificity."
    )


class FaithfulnessVerdict(BaseModel):
    """Judge's count of an answer's factual claims and how many its cited findings support."""

    total_claims: int = Field(ge=0, description="Total number of distinct factual claims.")
    supported_claims: int = Field(ge=0, description="Number of claims supported by their cited findings.")
    unsupported_claim_example: str = Field(
        description="One unsupported claim and why, or empty if all claims are supported."
    )


def read_final_answer(lines: list[dict]) -> str:
    """Return the run's final answer with its generated Sources block removed."""
    answer = lines[-1]["answer"]
    return answer.partition(SOURCES_HEADING)[0].strip()


def format_cited_findings(lines: list[dict], answer: str) -> str:
    """List each finding the answer cites, together with its evidence, as numbered text for the judge."""
    cited_ids = set(cited_finding_ids(answer))
    entries = []
    for finding in lines[-1]["findings"]:
        if finding["id"] not in cited_ids:
            continue

        entries.append(f"[{finding['id']}] {finding['claim']}")
        for evidence in finding["evidence"]:
            entries.append(f"    {evidence['relation']}: {evidence['text']}")

    return "\n".join(entries)


def judge_completeness_and_specificity(
    llm: LLMClient,
    question: str,
    answer_a: str,
    answer_b: str,
) -> ComparisonVerdict | None:
    """Ask the judge which of two answers is more complete and more specific, or return None for an invalid verdict."""
    messages = [
        {"role": "system", "content": JUDGE_COMPARISON_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer A:\n{answer_a}\n\nAnswer B:\n{answer_b}"},
    ]
    response = llm.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "comparison_verdict", "schema": ComparisonVerdict.model_json_schema()},
        },
    )
    try:
        return ComparisonVerdict.model_validate_json(response.choices[0].message.content or "")
    except ValidationError as error:
        logger.warning(f"Judge comparison reply failed validation: {error}")
        return None


def judge_completeness_and_specificity_both_orders(
    llm: LLMClient,
    question: str,
    single_answer: str,
    multi_answer: str,
) -> dict | None:
    """Compare the answers in both orders and keep each winner only when the two judgments agree."""
    single_first = judge_completeness_and_specificity(llm, question, single_answer, multi_answer)
    multi_first = judge_completeness_and_specificity(llm, question, multi_answer, single_answer)
    if single_first is None or multi_first is None:
        return None

    # Which mode each letter stands for, in each order.
    single_first_modes = {"A": "single", "B": "multi", "tie": "tie"}
    multi_first_modes = {"A": "multi", "B": "single", "tie": "tie"}

    outcome = {}
    for metric in PAIRWISE_METRICS:
        winner_single_first = single_first_modes[getattr(single_first, f"{metric}_winner")]
        winner_multi_first = multi_first_modes[getattr(multi_first, f"{metric}_winner")]
        agreed = winner_single_first == winner_multi_first
        outcome[metric] = winner_single_first if agreed else "tie"
        outcome[f"{metric}_agreed"] = agreed
    outcome["verdicts"] = [single_first.model_dump(), multi_first.model_dump()]
    return outcome


def judge_faithfulness(llm: LLMClient, answer: str, cited_findings: str) -> FaithfulnessVerdict | None:
    """Ask the judge how many claims the cited findings support, or return None for an invalid verdict."""
    messages = [
        {"role": "system", "content": JUDGE_FAITHFULNESS_PROMPT},
        {"role": "user", "content": f"Answer:\n{answer}\n\nCited findings:\n{cited_findings}"},
    ]
    response = llm.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "faithfulness_verdict", "schema": FaithfulnessVerdict.model_json_schema()},
        },
    )
    try:
        verdict = FaithfulnessVerdict.model_validate_json(response.choices[0].message.content or "")
    except ValidationError as error:
        logger.warning(f"Judge faithfulness reply failed validation: {error}")
        return None

    if verdict.supported_claims > verdict.total_claims:
        logger.warning(f"Judge counted {verdict.supported_claims} supported claims out of {verdict.total_claims}")
        return None
    return verdict


def judge_question(llm: LLMClient, question: str, single_lines: list[dict], multi_lines: list[dict]) -> dict:
    """Judge one question's single and multi answers on completeness, specificity, and faithfulness."""
    single_answer, multi_answer = read_final_answer(single_lines), read_final_answer(multi_lines)
    single_faithfulness = judge_faithfulness(llm, single_answer, format_cited_findings(single_lines, single_answer))
    multi_faithfulness = judge_faithfulness(llm, multi_answer, format_cited_findings(multi_lines, multi_answer))
    return {
        "comparison": judge_completeness_and_specificity_both_orders(llm, question, single_answer, multi_answer),
        "faithfulness": {
            "single": single_faithfulness.model_dump() if single_faithfulness else None,
            "multi": multi_faithfulness.model_dump() if multi_faithfulness else None,
        },
    }
