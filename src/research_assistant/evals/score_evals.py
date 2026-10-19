import argparse
import hashlib
import json
from pathlib import Path

from research_assistant.config import Config
from research_assistant.core.ledger import cited_finding_ids
from research_assistant.evals.llm_as_judge import RecordingLLMClient, judge_question
from research_assistant.evals.run_evals import EVAL_TRACE_DIR, MODES, QUESTIONS_PATH, load_questions
from research_assistant.llm.client import LLMClient
from research_assistant.prompts import JUDGE_COMPARISON_PROMPT, JUDGE_FAITHFULNESS_PROMPT

RESULTS_PATH = QUESTIONS_PATH.parent / "eval_results_with_scores.md"


def read_trace_lines(path: Path) -> list[dict]:
    """Read all JSON lines from one trace file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def retrieve_sections_read(lines: list[dict]) -> set[tuple[str, str]]:
    """Distinct (paper ID, section ID) pairs the run read successfully with read_paper_section."""
    sections = set()
    for line in lines:
        if line["type"] != "step":
            continue

        for call in line["tool_calls"]:
            if call["name"] != "read_paper_section" or call["tool_result"].startswith("Error:"):
                continue
            arguments = json.loads(call["arguments"])
            sections.add((arguments["paper_id"], str(arguments["section_id"])))

    return sections


def count_tool_recovery(lines: list[dict]) -> tuple[int, int]:
    """Count failed tool calls and how many the same worker later retried successfully."""
    calls = []
    for line in lines:
        if line["type"] != "step":
            continue

        worker = line.get("worker")
        for call in line["tool_calls"]:
            failed = call["tool_result"].startswith("Error:")
            calls.append((worker, call["name"], failed))

    errors = recovered = 0
    for index, (worker, name, failed) in enumerate(calls):
        worker, name, failed = calls[index]
        if not failed:
            continue
        errors += 1
        later_success = any(
            later_worker == worker and later_name == name and not later_failed
            for later_worker, later_name, later_failed in calls[index + 1 :]
        )
        if later_success:
            recovered += 1
    return errors, recovered


def score_run(lines: list[dict]) -> dict:
    """Score one run's coverage, tokens, time, completion, and tool recovery."""
    summary = lines[-1]
    steps = [line for line in lines if line["type"] == "step"]

    sections = retrieve_sections_read(lines)
    papers = {paper_id for paper_id, _ in sections}

    errors, recoveries = count_tool_recovery(lines)
    worker_seconds = summary.get("worker_seconds")
    speedup = sum(worker_seconds) / max(worker_seconds) if worker_seconds else None

    return {
        "sections": len(sections),
        "papers": len(papers),
        "tokens": sum(step["prompt_tokens"] + step["completion_tokens"] for step in steps),
        "seconds": summary["seconds"],
        "completed": bool(cited_finding_ids(summary["answer"])),
        "tool_errors": errors,
        "tool_recoveries": recoveries,
        "speedup": speedup,
    }


def fingerprint_judge_inputs(single_path: Path, multi_path: Path, judge_model: str) -> str:
    """Hash both traces, the judge model, and the judge prompts to tell whether a saved judgment still applies."""
    digest = hashlib.sha256()
    for part in (single_path.read_bytes(), multi_path.read_bytes()):
        digest.update(part)
    for text in (judge_model, JUDGE_COMPARISON_PROMPT, JUDGE_FAITHFULNESS_PROMPT):
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def reuse_saved_judgment(scores_path: Path, fingerprint: str) -> dict | None:
    """Return the judgment saved in the question's scores file if it was made from the same inputs, or None."""
    if not scores_path.exists():
        return None
    saved_judgment = json.loads(scores_path.read_text(encoding="utf-8")).get("judgment")
    if saved_judgment and saved_judgment["input_fingerprint"] == fingerprint:
        return saved_judgment
    return None


def run_judgment(question: dict, single_path: Path, multi_path: Path, judge_model: str, fingerprint: str) -> dict:
    """Judge the question's two answers and return the verdicts together with the judge's recorded calls."""
    llm = RecordingLLMClient(LLMClient(Config(), model=judge_model))
    verdicts = judge_question(llm, question["question"], read_trace_lines(single_path), read_trace_lines(multi_path))
    return {"judge_model": judge_model, "input_fingerprint": fingerprint, **verdicts, "calls": llm.calls}


def compare_question_across_runs(question: dict, judge_model: str, rejudge: bool) -> dict | None:
    """Score and judge a question's two runs into one saved record, or return None if a trace is missing."""
    single_path = EVAL_TRACE_DIR / f"{question['id']}_single.jsonl"
    multi_path = EVAL_TRACE_DIR / f"{question['id']}_multi.jsonl"
    scores_path = EVAL_TRACE_DIR / f"{question['id']}_scores.json"

    if not single_path.exists() or not multi_path.exists():
        return None

    single = score_run(read_trace_lines(single_path))
    multi = score_run(read_trace_lines(multi_path))
    fingerprint = fingerprint_judge_inputs(single_path, multi_path, judge_model)
    judgment = None if rejudge else reuse_saved_judgment(scores_path, fingerprint)
    if judgment is None:
        judgment = run_judgment(question, single_path, multi_path, judge_model, fingerprint)

    comparison = {
        "id": question["id"],
        "breadth": question["breadth"],
        "single": single,
        "multi": multi,
        "coverage_ratio": multi["sections"] / single["sections"] if single["sections"] else None,
        "token_ratio": multi["tokens"] / single["tokens"] if single["tokens"] else None,
        "judgment": judgment,
    }
    scores_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    return comparison


def compute_average(values: list[float]) -> float | None:
    """Average the values, or return None if there are none."""
    return sum(values) / len(values) if values else None


def summarize_comparisons(comparisons: list[dict]) -> dict:
    """Average the single-vs-multi comparisons across questions."""
    runs = [comparison[mode] for comparison in comparisons for mode in MODES]
    tool_errors = sum(run["tool_errors"] for run in runs)
    tool_recoveries = sum(run["tool_recoveries"] for run in runs)

    return {
        "questions": len(comparisons),
        "coverage_ratio": compute_average(
            [c["coverage_ratio"] for c in comparisons if c["coverage_ratio"] is not None]
        ),
        "token_ratio": compute_average([c["token_ratio"] for c in comparisons if c["token_ratio"] is not None]),
        "speedup": compute_average([c["multi"]["speedup"] for c in comparisons if c["multi"]["speedup"] is not None]),
        "single_completed": compute_average([float(c["single"]["completed"]) for c in comparisons]),
        "multi_completed": compute_average([float(c["multi"]["completed"]) for c in comparisons]),
        "tool_errors": tool_errors,
        "tool_recovery_rate": tool_recoveries / tool_errors if tool_errors else None,
    }


def summarize_judgments(comparisons: list[dict]) -> dict:
    """Count the judge's winners per metric and pool each mode's faithfulness across questions."""
    verdicts = [c["judgment"]["comparison"] for c in comparisons if c["judgment"]["comparison"]]
    summary = {"judged": len(verdicts)}
    for metric in ("completeness", "specificity"):
        winners = [verdict[metric] for verdict in verdicts]
        summary[metric] = {mode: winners.count(mode) for mode in ("multi", "tie", "single")}
        summary[f"{metric}_agreed"] = sum(verdict[f"{metric}_agreed"] for verdict in verdicts)

    for mode in MODES:
        counts = [c["judgment"]["faithfulness"][mode] for c in comparisons if c["judgment"]["faithfulness"][mode]]
        total_claims = sum(count["total_claims"] for count in counts)
        supported_claims = sum(count["supported_claims"] for count in counts)
        summary[f"{mode}_faithfulness"] = supported_claims / total_claims if total_claims else None
    return summary


def format_wins(wins: dict) -> str:
    """Format a metric's winner counts as multi / tie / single."""
    return f"{wins['multi']} / {wins['tie']} / {wins['single']}"


def format_ratio(value: float | None) -> str:
    """Format a ratio like 2.0×, or – when it is missing."""
    return f"{value:.1f}×" if value is not None else "–"


def format_percentage(value: float | None) -> str:
    """Format a rate like 88%, or – when it is missing."""
    return f"{value:.0%}" if value is not None else "–"


def write_results(comparisons: list[dict], path: Path) -> None:
    """Write the summary and per-question comparison tables to a Markdown file."""
    result = [
        "# Eval results: single agent vs. multi-agent",
        "",
        "| Questions | n | Coverage (sections, multi ÷ single) | Tokens (multi ÷ single) "
        "| Parallel speed-up | Completed (single / multi) | Tool errors | Recovered |",
        "|---|---|---|---|---|---|---|---|",
    ]
    groups = {
        "all": comparisons,
        "broad": [c for c in comparisons if c["breadth"] == "broad"],
        "narrow": [c for c in comparisons if c["breadth"] == "narrow"],
    }
    for label, group in groups.items():
        summary = summarize_comparisons(group)
        result.append(
            f"| {label} | {summary['questions']} | {format_ratio(summary['coverage_ratio'])} "
            f"| {format_ratio(summary['token_ratio'])} | {format_ratio(summary['speedup'])} "
            f"| {format_percentage(summary['single_completed'])} / {format_percentage(summary['multi_completed'])} "
            f"| {summary['tool_errors']} | {format_percentage(summary['tool_recovery_rate'])} |"
        )

    judge_models = sorted({c["judgment"]["judge_model"] for c in comparisons})
    result += [
        "",
        f"## Answer quality (LLM judge: {', '.join(judge_models) or '–'})",
        "",
        "| Questions | Judged | Completeness (multi / tie / single) | Specificity (multi / tie / single) "
        "| Faithfulness (single / multi) |",
        "|---|---|---|---|---|",
    ]
    for label, group in groups.items():
        judged = summarize_judgments(group)
        result.append(
            f"| {label} | {judged['judged']} | {format_wins(judged['completeness'])} "
            f"| {format_wins(judged['specificity'])} "
            f"| {format_percentage(judged['single_faithfulness'])} "
            f"/ {format_percentage(judged['multi_faithfulness'])} |"
        )

    result += [
        "",
        "## Per question",
        "",
        "| ID | Breadth | Sections | Papers | Tokens | Seconds | Speed-up | More complete | More specific "
        "| Faithfulness (single / multi) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        single, multi, judgment = c["single"], c["multi"], c["judgment"]
        verdict = judgment["comparison"] or {"completeness": "invalid", "specificity": "invalid"}
        faithfulness = [
            f"{counts['supported_claims']}/{counts['total_claims']}" if counts else "invalid"
            for counts in (judgment["faithfulness"]["single"], judgment["faithfulness"]["multi"])
        ]
        result.append(
            f"| {c['id']} | {c['breadth']} | {single['sections']} → {multi['sections']} "
            f"| {single['papers']} → {multi['papers']} | {single['tokens']:,} → {multi['tokens']:,} "
            f"| {single['seconds']:.0f} → {multi['seconds']:.0f} | {format_ratio(multi['speedup'])} "
            f"| {verdict['completeness']} | {verdict['specificity']} | {faithfulness[0]} / {faithfulness[1]} |"
        )

    judge_calls = [call for c in comparisons for call in c["judgment"]["calls"]]
    judge_tokens = sum(call["prompt_tokens"] + call["completion_tokens"] for call in judge_calls)

    result += [
        "",
        "Tokens count the research model's calls; extraction and the supervisor's brief and answer calls "
        "are not included.",
        "Parallel speed-up = total worker time ÷ slowest worker (the research phase only).",
        "Judge winners count only when both answer orders agree; disagreements are ties. "
        "Faithfulness = claims supported by their cited findings ÷ all factual claims.",
        f"Judge calls: {len(judge_calls)} · {judge_tokens:,} tokens. Every metric, verdict, and judge call "
        "for a question is saved in traces/eval/<id>_scores.json.",
    ]
    path.write_text("\n".join(result) + "\n", encoding="utf-8")


def main() -> None:
    """Score and judge every eval question with both traces and write the results report."""
    parser = argparse.ArgumentParser(description="Score and judge the saved eval traces and write the results report.")
    parser.add_argument(
        "--output", type=Path, default=RESULTS_PATH, help="Markdown file to write. Default: %(default)s."
    )
    parser.add_argument("--rejudge", action="store_true", help="Judge again even when a saved judgment matches.")
    args = parser.parse_args()
    judge_model = Config().judge_model_id
    if not judge_model:
        parser.error("set JUDGE_MODEL_ID in .env to the model that judges answer quality")

    comparisons, missing = [], []
    for question in load_questions(QUESTIONS_PATH):
        comparison = compare_question_across_runs(question, judge_model, args.rejudge)
        if comparison is None:
            missing.append(question["id"])
        else:
            comparisons.append(comparison)

    write_results(comparisons, args.output)
    print(f"Scored {len(comparisons)} questions; wrote {args.output}")
    if missing:
        print(f"Missing a trace (not scored): {', '.join(missing)}")
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
