import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from research_assistant.core.agent import RunResult


def write_trace(result: RunResult, trace_dir: Path, model: str) -> Path:
    """Write the run's record to a JSONL trace file in trace_dir and return its path."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_id = uuid.uuid4().hex[:6]
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{stamp}_{file_id}.jsonl"

    # step summary
    lines = [{"type": "step", **asdict(step)} for step in result.steps]
    # run summary
    lines.append(
        {
            "type": "run",
            "question": result.question,
            "answer": result.answer,
            "stop_reason": result.stop_reason,
            "model": model,
            "seconds": result.seconds,
            "steps": len(result.steps),
            "tool_calls": result.tool_call_count,
            "total_tokens": result.total_tokens,
            "peak_prompt_tokens": result.peak_prompt_tokens,
            "estimated_peak_without_ledger": result.estimated_peak_without_ledger,
            "findings": [asdict(finding) for finding in result.findings],
            "open_questions": list(result.open_questions),
            "unbacked_citations": list(result.unbacked_citations),
        }
    )
    with path.open("w", encoding="utf-8") as file:
        for line in lines:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
