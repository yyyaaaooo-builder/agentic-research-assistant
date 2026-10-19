"""Command-line entry point: runs parallel research workers, or one agent with --single, and prints the answer."""

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path

from research_assistant.config import Config
from research_assistant.core.agent import ReActAgent, RunResult
from research_assistant.core.supervisor import ResearchResult, Supervisor
from research_assistant.llm.client import LLMClient
from research_assistant.mcp_servers import MCP_SERVERS
from research_assistant.prompts import RESEARCH_ASSISTANT_PROMPT
from research_assistant.tools.mcp_client import McpConnection, register_mcp_tools
from research_assistant.tools.registry import ToolRegistry
from research_assistant.tools.search import SearchTool
from research_assistant.tracing import build_agent_run_trace_lines, build_multi_agent_trace_lines, write_trace_file

TRACE_DIR = Path("traces")


def main() -> None:
    """Answer one research question from the command line and print the answer with a run summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Answer a research question from arXiv papers by running parallel research workers "
            "and merging their findings into one cited answer."
        )
    )
    parser.add_argument("question", help="The research question to answer")
    parser.add_argument(
        "--mcp", action="append", choices=list(MCP_SERVERS), help="MCP server to connect. Default: arxiv."
    )
    parser.add_argument("--single", action="store_true", help="Answer with one agent instead of parallel workers.")
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Path to the output trace file. Defaults to a new timestamped file in traces/.",
    )
    args = parser.parse_args()
    server_names = args.mcp or ["arxiv"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    config = Config()
    note_source: dict[str, tuple[str, str]] = {}
    with ExitStack() as stack:
        registry = ToolRegistry()
        registry.register(SearchTool())
        for name in server_names:
            server = MCP_SERVERS[name]
            connection = stack.enter_context(McpConnection(server))
            register_mcp_tools(registry, connection, server)
            note_source.update(server.note_source)

        llm = LLMClient(config)
        agent = ReActAgent(
            llm=llm,
            registry=registry,
            max_steps=config.max_steps,
            system_prompt=RESEARCH_ASSISTANT_PROMPT,
            max_tool_calls=config.max_tool_calls,
            max_run_seconds=config.max_run_seconds,
            note_source=note_source,
            extraction_llm=LLMClient(config, model=config.extraction_model_id),
        )

        if args.single:
            _report_single_agent_run(agent.run(args.question), config.model_id, args.trace_file)
        else:
            research = Supervisor(llm=llm, worker=agent).research(args.question)
            _report_multi_agent_research(research, config.model_id, args.trace_file)


def _report_single_agent_run(result: RunResult, model: str, trace_file: Path | None) -> None:
    """Save the run's trace, then print its answer and a one-line summary."""
    trace_path = write_trace_file(build_agent_run_trace_lines(result, model), TRACE_DIR, trace_file)
    tool_calls = result.tool_call_count

    print()
    print(result.answer)
    print("---")
    print(
        f"{result.stop_reason} · {len(result.steps)} steps · {tool_calls} tool call{'s' if tool_calls != 1 else ''}"
        f" · {result.total_tokens:,} tokens · {result.seconds:.1f}s"
    )
    print(f"Trace: {trace_path}")


def _report_multi_agent_research(result: ResearchResult, model: str, trace_file: Path | None) -> None:
    """Save the research trace, then print its answer and a one-line summary of its brief, workers, and findings."""
    trace_path = write_trace_file(build_multi_agent_trace_lines(result, model), TRACE_DIR, trace_file)

    print()
    print(result.answer)
    print("---")
    print(
        f"{len(result.brief.sub_questions)} sub-questions · {len(result.worker_results)} workers finished"
        f" · {len(result.findings)} findings · {result.seconds:.1f}s"
    )
    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    main()
