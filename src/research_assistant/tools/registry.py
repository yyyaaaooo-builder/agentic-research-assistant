import inspect
import json
import logging

from research_assistant.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Holds the agent's tools, describes them to the model, and runs the one it requests."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool under its name; raise ValueError if that name is already taken."""
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")

        self._tools[tool.name] = tool

    def to_openai_tools(self) -> list[dict]:
        """Describe every registered tool in the format the OpenAI-compatible API expects."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def invoke(self, name: str, arguments: str) -> str:
        """Run the named tool with the model's JSON arguments.

        Always returns a string, either the tool's result or an error message the model
        can act on, and never raises, so a bad tool call can't crash the agent run.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(self._tools)}."

        try:
            parsed = json.loads(arguments)
            if not isinstance(parsed, dict):
                return "Error: arguments must be a JSON object."
            inspect.signature(tool.run).bind(**parsed)
        except json.JSONDecodeError as error:
            return f"Error: arguments are not valid JSON ({error})."
        except TypeError as error:
            return f"Error: invalid arguments for {name!r}: {error}."

        try:
            return tool.run(**parsed)
        except Exception:
            logger.exception("Tool %r crashed with arguments %r", name, parsed)
            return f"Error: tool {name!r} failed unexpectedly."
