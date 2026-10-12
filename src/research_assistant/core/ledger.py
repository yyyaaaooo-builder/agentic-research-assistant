"""The evidence ledger: what the agent has learned, with sources and links between findings."""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from research_assistant.llm.client import LLMClient
from research_assistant.prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

Relation = Literal["supports", "contradicts", "qualifies"]
MAX_NEW_FINDINGS = 5  # per section: keeps each read's contribution to the notes small
EXTRACTION_MAX_TOKENS = 3000
CITATION_PATTERN = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?\]?(?:\s*§\s*(\w+(?:\.\w+)*))?")


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


def unbacked_citations(answer: str, ledger: Ledger) -> list[str]:
    """Citations in the answer that no finding or evidence in the ledger comes from."""
    sources = {(finding.paper_id, finding.section) for finding in ledger.findings}
    sources |= {(evidence.paper_id, evidence.section) for finding in ledger.findings for evidence in finding.evidence}
    papers = {paper_id for paper_id, _ in sources}

    unbacked: list[str] = []
    for paper, section in CITATION_PATTERN.findall(answer):
        backed = (paper, section) in sources if section else paper in papers
        citation = format_citation(paper, section)
        if not backed and citation not in unbacked:
            unbacked.append(citation)
    return unbacked
