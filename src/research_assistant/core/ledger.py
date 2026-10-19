"""The evidence ledger: what the agent has learned, with sources and links between findings."""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from research_assistant.llm.client import LLMClient
from research_assistant.prompts import ANSWER_PROMPT, EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

Relation = Literal["supports", "contradicts", "qualifies"]
MAX_NEW_FINDINGS = 5  # per section: keeps each read's contribution to the notes small
EXTRACTION_MAX_TOKENS = 3000

CITATION_PATTERN = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?\]?(?:\s*§\s*(\w+(?:\.\w+)*))?")
# One cited ID or a range of IDs, e.g. 3 or 46–48 (hyphen or en dash).
FOOTNOTE_ITEM = r"\d+(?:\s*[-–]\s*\d+)?"
# A footnote of comma-separated items, e.g. [2], [1, 3], or [46–48].
FOOTNOTE_PATTERN = re.compile(rf"\[({FOOTNOTE_ITEM}(?:\s*,\s*{FOOTNOTE_ITEM})*)\]")
FOOTNOTE_RANGE_PATTERN = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


def format_citation(paper_id: str, section: str) -> str:
    """How a source is cited in an answer, e.g. arXiv:2410.18050 §5.1."""
    source = f"arXiv:{paper_id}"
    return f"{source} §{section}" if section else source


@dataclass(frozen=True)
class Evidence:
    """A later source that supports, contradicts, or qualifies an existing finding."""

    relation: Relation
    text: str
    paper_id: str
    section: str

    @property
    def citation(self) -> str:
        return format_citation(self.paper_id, self.section)


@dataclass
class Finding:
    """A claim learned from a paper, with the evidence gathered about it."""

    id: int
    claim: str
    paper_id: str
    section: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return format_citation(self.paper_id, self.section)


class Link(BaseModel):
    """Evidence from a paper section that supports, contradicts, or qualifies an existing finding."""

    target: int = Field(description="Number of the existing finding this is about, e.g. 2.")
    relation: Relation = Field(description="How the evidence in the current section relates to the target finding.")
    text: str = Field(description="Concise summary of the evidence this section provides about the finding.")


class ExtractionResult(BaseModel):
    """The findings, links to existing findings, and open questions extracted from one paper section."""

    new_findings: list[str] = Field(default_factory=list, json_schema_extra={"maxItems": MAX_NEW_FINDINGS})
    links: list[Link] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class Ledger:
    """The findings and open questions the agent has gathered during a run."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.open_questions: list[str] = []

    def add(self, result: ExtractionResult, paper_id: str, section: str) -> list[str]:
        """Record one section's extraction and return the changes made.

        At most MAX_NEW_FINDINGS findings are added, links to unknown findings are skipped,
        and open questions replace the previous list.
        """
        changes = []
        findings_by_id = {finding.id: finding for finding in self.findings}

        for link in result.links:
            finding = findings_by_id.get(link.target)
            if finding is None:
                logger.warning(f"Skipping link to unknown finding [{link.target}]")
                continue
            finding.evidence.append(
                Evidence(relation=link.relation, text=link.text, paper_id=paper_id, section=section)
            )
            changes.append(f"{link.relation} [{link.target}]")

        for claim in result.new_findings[:MAX_NEW_FINDINGS]:
            finding_id = len(self.findings) + 1
            self.findings.append(Finding(finding_id, claim, paper_id, section))
            changes.append(f"added [{finding_id}]")

        self.open_questions = list(result.open_questions)
        return changes

    def render(self) -> str:
        """Render the ledger as plain-text working notes for the model."""
        if not self.findings:
            return "Working notes: no findings yet."

        lines = ["Working notes (evidence ledger):"]
        for finding in self.findings:
            lines.append(f"[{finding.id}] {finding.claim} ({finding.citation})")
            for evidence in finding.evidence:
                lines.append(f"    {evidence.relation}: {evidence.text} ({evidence.citation})")

        if self.open_questions:
            lines.append("Open questions:")
            for question in self.open_questions:
                lines.append(f"- {question}")
        return "\n".join(lines)

    def render_sources(self, cited_ids: list[int]) -> str:
        """Render the sources of the findings an answer cites.
        Each line gives a finding's own citation, then the citation of each piece of linked evidence:

            **Sources**
            - [1] arXiv:2610.13504 §5
            - [2] arXiv:2610.13504 §5; contradicts: arXiv:2609.01234 §4.2
        """
        findings_by_id = {finding.id: finding for finding in self.findings}
        source_lines = []

        for finding_id in cited_ids:
            finding = findings_by_id.get(finding_id)
            if finding is None:
                continue

            line = f"- [{finding.id}] {finding.citation}"
            for evidence in finding.evidence:
                line += f"; {evidence.relation}: {evidence.citation}"
            source_lines.append(line)

        if not source_lines:
            return ""
        return "\n\n**Sources**\n" + "\n".join(source_lines)


def extract(llm: LLMClient, question: str, ledger: Ledger, section_text: str) -> ExtractionResult | None:
    """Extract what one paper section adds to the ledger, or None if the reply is unusable."""
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{ledger.render()}\n\nSection text:\n{section_text}"},
    ]
    response = llm.generate(
        messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": ExtractionResult.model_json_schema()},
        },
        max_tokens=EXTRACTION_MAX_TOKENS,
    )
    choice = response.choices[0]
    reply = choice.message.content or ""
    _log_extraction_reply(response, reply)

    if choice.finish_reason == "length":
        logger.warning(f"Extraction hit the {EXTRACTION_MAX_TOKENS}-token limit; keeping the raw section text")
        return None
    try:
        result = ExtractionResult.model_validate_json(reply)
    except ValidationError as error:
        logger.warning(f"Extraction failed validation: {error}")
        return None
    logger.info(
        f"Extracted {len(result.new_findings)} findings, {len(result.links)} links, "
        f"{len(result.open_questions)} open questions"
    )
    return result


def write_answer_from_ledger(llm: LLMClient, question: str, ledger: Ledger):
    """Answer the question from the ledger's findings alone and return the model's response."""
    messages = [
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n{ledger.render()}"},
    ]
    return llm.generate(messages)


def _log_extraction_reply(response, reply: str) -> None:
    """Log how long an extraction reply was, including any hidden reasoning tokens, and why it ended."""
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None)
    completion = usage.completion_tokens if usage else None
    logger.info(
        f"Extraction reply: {completion} tokens (reasoning {reasoning}), "
        f"finish={response.choices[0].finish_reason}, {len(reply):,} chars"
    )


def cited_finding_ids(answer: str) -> list[int]:
    """Finding IDs cited in the answer's footnotes, such as [2], [1, 3], or [46–48], sorted and without repeats."""
    cited_ids = set()
    for footnote in FOOTNOTE_PATTERN.findall(answer):
        for item in footnote.split(","):
            id_range = FOOTNOTE_RANGE_PATTERN.fullmatch(item.strip())
            if id_range:
                first_id, last_id = int(id_range.group(1)), int(id_range.group(2))
                cited_ids.update(range(first_id, last_id + 1))
            else:
                cited_ids.add(int(item))
    return sorted(cited_ids)


def unbacked_citations(answer: str, ledger: Ledger) -> list[str]:
    """Citations in the answer that the ledger can't back: unknown finding IDs, or sources no finding comes from."""
    ledger_sources = {(finding.paper_id, finding.section) for finding in ledger.findings}
    ledger_sources |= {
        (evidence.paper_id, evidence.section) for finding in ledger.findings for evidence in finding.evidence
    }
    ledger_papers = {paper_id for paper_id, _ in ledger_sources}
    finding_ids = {finding.id for finding in ledger.findings}
    unbacked = [f"[{finding_id}]" for finding_id in cited_finding_ids(answer) if finding_id not in finding_ids]

    for paper, section in CITATION_PATTERN.findall(answer):
        backed = (paper, section) in ledger_sources if section else paper in ledger_papers
        citation = format_citation(paper, section)
        if not backed and citation not in unbacked:
            unbacked.append(citation)
    return unbacked
