"""Eval runner: answers every eval question with ask in single- and multi-agent modes, saving one trace per run."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parents[3]
QUESTIONS_PATH = EVALS_DIR / "questions.jsonl"
EVAL_TRACE_DIR = PROJECT_DIR / "traces" / "eval"  # one trace per question and mode, e.g. q1_single.jsonl
EVAL_LOG_DIR = EVAL_TRACE_DIR / "logs"
MODES = ("single", "multi")
RUN_TIMEOUT_SECONDS = 45 * 60  # stop a stuck run rather than block the whole eval


def load_questions(path: Path) -> list[dict]:
    """Eval questions from a JSONL file, one {"id", "breadth", "question"} object per line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ask_command(question: str, mode: str, trace_path: Path) -> list[str]:
    """Command line that runs ask on a question and writes its trace to trace_path, with --single for single mode."""
    command = [sys.executable, "-m", "research_assistant.cli", "--trace-file", str(trace_path)]
    if mode == "single":
        command.append("--single")
    command.append(question)
    return command


def run_question(question: dict, mode: str, eval_trace_path: Path) -> bool:
    """Run ask on one question in one mode, writing its trace to eval_trace_path; return whether it succeeded."""
    log_path = EVAL_LOG_DIR / f"{question['id']}_{mode}.log"
    try:
        completed = subprocess.run(
            ask_command(question["question"], mode, eval_trace_path),
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
        EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")

        if completed.returncode != 0 or not eval_trace_path.exists():
            logger.error(f"{question['id']} {mode}: ask failed (exit {completed.returncode}); see {log_path}")
            return False

        logger.info(f"{question['id']} {mode}: saved {eval_trace_path}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"{question['id']} {mode}: timed out after {RUN_TIMEOUT_SECONDS}s")
        return False


def main() -> None:
    """Run every eval question in both modes, skipping runs whose trace already exists."""
    parser = argparse.ArgumentParser(description="Run the eval questions through ask in single- and multi-agent modes.")
    parser.add_argument("--question", action="append", help="Question id to run (repeatable). Default: all.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    questions = load_questions(QUESTIONS_PATH)
    if args.question:
        questions = [question for question in questions if question["id"] in args.question]

    succeeded = failed = skipped = 0
    for question in questions:
        for mode in MODES:
            eval_trace_path = EVAL_TRACE_DIR / f"{question['id']}_{mode}.jsonl"
            if eval_trace_path.exists():
                logger.info(f"{question['id']} {mode}: trace exists, skipping")
                skipped += 1
            elif run_question(question, mode, eval_trace_path):
                succeeded += 1
            else:
                failed += 1
    logger.info(f"Eval run finished: {succeeded} succeeded, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
