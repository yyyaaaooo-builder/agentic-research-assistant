"""Connect to MCP servers and expose their tools as ordinary Tools."""

import asyncio
import logging
import threading
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from research_assistant.tools.base import Tool
from research_assistant.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 60
CALL_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for launching an MCP server and exposing its tools.

    Defines the server's launch command and arguments, the tools to expose,
    and any arguments to supply as overrides for specific tools.
    """

    name: str
    command: str
    args: tuple[str, ...]
    tools: tuple[str, ...]
    pinned_args: dict[str, dict] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)
    note_source: dict[str, tuple[str, str]] = field(default_factory=dict)


class McpConnection:
    """Manages a persistent connection to an MCP server, allowing the synchronous agent to call MCP tools."""

    def __init__(self, config: McpServerConfig) -> None:
        self._config = config
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._serving = asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        if not self._ready.wait(timeout=CONNECT_TIMEOUT_SECONDS):
            raise RuntimeError(f"MCP server {self._config.name!r} did not start within {CONNECT_TIMEOUT_SECONDS}s.")
        logger.info(f"Connected to MCP server {self._config.name!r}")

    async def _serve(self) -> None:
        """Open the server session and hold it until close() is called."""
        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self._config.command, args=list(self._config.args))

        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            self._session = session
            self._ready.set()
            await self._stop.wait()

    def _run(self, coroutine):
        """Run a coroutine on the background loop and wait for its result."""
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=CALL_TIMEOUT_SECONDS)

    def list_tools(self):
        """Return the tools this server offers."""
        return self._run(self._session.list_tools()).tools

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Call a tool and return its text output and whether it failed."""
        result = self._run(self._session.call_tool(name, arguments))
        text = "".join(getattr(content, "text", "") for content in result.content)
        return (text, result.is_error)

    def close(self) -> None:
        """Close the session, shut down the server, and stop the background loop."""
        self._loop.call_soon_threadsafe(self._stop.set)
        self._serving.result(timeout=CALL_TIMEOUT_SECONDS)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def __enter__(self) -> "McpConnection":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class McpTool(Tool):
    """Wraps an MCP server tool so the agent can invoke it like any other tool."""

    def __init__(
        self, connection: McpConnection, remote_tool, pinned_args: dict, description: str | None = None
    ) -> None:
        self.name = remote_tool.name
        self.description = description or remote_tool.description or ""
        self.parameters = remote_tool.input_schema
        self._connection = connection
        self._pinned_args = pinned_args

    def run(self, **kwargs) -> str:
        """Call the remote tool with the model's arguments, applying any pinned values."""
        overridden = {
            key: value for key, value in kwargs.items() if key in self._pinned_args and value != self._pinned_args[key]
        }
        if overridden:
            logger.info(f"{self.name}: overriding model arguments {overridden}.")
        arguments = {**kwargs, **self._pinned_args}
        text, is_error = self._connection.call_tool(self.name, arguments)
        return f"Error: {text}" if is_error else text


def register_mcp_tools(registry: ToolRegistry, connection: McpConnection, config: McpServerConfig) -> None:
    """Register the server's allowlisted tools with the registry, failing fast if any are missing."""

    remote_tools = {tool.name: tool for tool in connection.list_tools()}
    missing = [name for name in config.tools if name not in remote_tools]
    if missing:
        raise ValueError(f"MCP server {config.name!r} has no tools named {missing}.")

    for name in config.tools:
        remote_tool = remote_tools[name]
        pinned_args = config.pinned_args.get(name, {})
        registry.register(McpTool(connection, remote_tool, pinned_args, config.descriptions.get(name)))

    logger.info(f"Registered {len(config.tools)} tools from MCP server {config.name!r}")
