"""Command-line entry point: runs the research agent and saves a trace"""

import argparse
import logging
from contextlib import ExitStack
from pathlib import Path

from research_assistant.config import Config
from research_assistant.core.agent import ReActAgent
from research_assistant.llm.client import LLMClient
from research_assistant.mcp_servers import MCP_SERVERS
from research_assistant.prompts import RESEARCH_ASSISTANT_PROMPT
from research_assistant.tools.mcp_client import McpConnection, register_mcp_tools
from research_assistant.tools.registry import ToolRegistry
from research_assistant.tools.search import SearchTool
from research_assistant.tracing import write_trace

TRACE_DIR = Path("traces")


def main() -> None:
    """Answer one research question from the command line, then print where its trace was saved."""
    parser = argparse.ArgumentParser(description="Ask a research question and get an answer backed by arXiv papers.")
    parser.add_argument("question", help="The research question to answer")
    parser.add_argument(
        "--mcp", action="append", choices=list(MCP_SERVERS), help="MCP server to connect (repeatable). Default: arxiv."
    )
    args = parser.parse_args()
    server_names = args.mcp or ["arxiv"]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)

    config = Config()
    with ExitStack() as stack:
        registry = ToolRegistry()
        registry.register(SearchTool())
        for name in server_names:
            server = MCP_SERVERS[name]
            connection = stack.enter_context(McpConnection(server))
            register_mcp_tools(registry, connection, server)

        agent = ReActAgent(
            llm=LLMClient(config),
            registry=registry,
            max_steps=config.max_steps,
            system_prompt=RESEARCH_ASSISTANT_PROMPT,
            max_tool_calls=config.max_tool_calls,
            max_run_seconds=config.max_run_seconds,
        )

        result = agent.run(args.question)
    trace_path = write_trace(result=result, trace_dir=TRACE_DIR, model=config.model_id)
    tool_calls = result.tool_call_count
    print()
    print(result.answer)
    print("---")
    print(
        f"{result.stop_reason} · {len(result.steps)} steps · {tool_calls} tool call{'s' if tool_calls != 1 else ''}"
        f" · {result.total_tokens:,} tokens · {result.seconds:.1f}s"
    )
    print(f"Trace: {trace_path}")
