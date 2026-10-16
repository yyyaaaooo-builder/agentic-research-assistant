import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from research_assistant.core.agent import RunResult
from research_assistant.core.supervisor import ResearchResult


def _create_path_for_tracing(trace_dir: Path) -> Path:
    """Unique trace file path in trace_dir, named by time and a short random ID."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_id = uuid.uuid4().hex[:6]
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"{stamp}_{file_id}.jsonl"
    return path


def build_agent_run_trace_lines(result: RunResult, model: str) -> list[dict]:
    """Trace lines for one agent run: a "step" line per model call, then a "run" summary line."""
    lines = [{"type": "step", **asdict(step)} for step in result.steps]
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
    return lines


def build_multi_agent_trace_lines(result: ResearchResult, model: str) -> list[dict]:
    """Trace lines for a multi-agent run: each worker's lines tagged with its number, then a "research" summary line."""
    lines = []

    for worker_number, worker_result in enumerate(result.worker_results, start=1):
        lines.extend({**line, "worker": worker_number} for line in build_agent_run_trace_lines(worker_result, model))
    lines.append(
        {
            "type": "research",
            "question": result.question,
            "answer": result.answer,
            "model": model,
            "seconds": result.seconds,
            "brief": result.brief.model_dump(),
            "workers_planned": len(result.brief.sub_questions),
            "workers_finished": len(result.worker_results),
            "worker_seconds": [worker_result.seconds for worker_result in result.worker_results],
            "findings": [asdict(finding) for finding in result.findings],
            "open_questions": list(result.open_questions),
            "unbacked_citations": list(result.unbacked_citations),
        }
    )
    return lines


def write_trace_file(trace_lines: list[dict], trace_dir: Path) -> Path:
    """Write trace lines to a new JSONL file in trace_dir, one JSON object per line, and return its path."""
    path = _create_path_for_tracing(trace_dir)
    with path.open("w", encoding="utf-8") as file:
        for line in trace_lines:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path
