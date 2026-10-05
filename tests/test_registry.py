import pytest

from research_assistant.tools.base import Tool
from research_assistant.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Return the given text unchanged."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str) -> str:
        return text


class BrokenTool(Tool):
    name = "broken"
    description = "Always crashes."
    parameters = {"type": "object", "properties": {}}

    def run(self) -> str:
        raise RuntimeError("tool bug")


@pytest.fixture
def registry():
    tool_registry = ToolRegistry()
    tool_registry.register(EchoTool())
    tool_registry.register(BrokenTool())
    return tool_registry


def test_invoke_runs_the_tool(registry):
    assert registry.invoke("echo", '{"text": "hello"}') == "hello"


@pytest.mark.parametrize(
    "name, arguments, expected",
    [
        pytest.param(
            "ecko",
            '{"text": "hi"}',
            "Error: unknown tool 'ecko'. Available tools: echo, broken.",
            id="unknown-tool",
        ),
        pytest.param(
            "echo",
            "{'text': 'hi'}",
            "Error: arguments are not valid JSON "
            "(Expecting property name enclosed in double quotes: line 1 column 2 (char 1)).",
            id="single-quoted-json",
        ),
        pytest.param(
            "echo",
            '{"txt": "hi"}',
            "Error: invalid arguments for 'echo': missing a required argument: 'text'.",
            id="wrong-argument-name",
        ),
    ],
)
def test_invoke_rejects_bad_calls(registry, name, arguments, expected):
    assert registry.invoke(name, arguments) == expected


def test_invoke_survives_a_tool_crash(registry, caplog):
    assert registry.invoke("broken", "{}") == "Error: tool 'broken' failed unexpectedly."
    assert "tool bug" in caplog.text


def test_register_rejects_duplicate_names(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(EchoTool())


def test_to_openai_tools(registry):
    tools = registry.to_openai_tools()

    assert [tool["function"]["name"] for tool in tools] == ["echo", "broken"]
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the given text unchanged.",
            "parameters": EchoTool.parameters,
        },
    }
