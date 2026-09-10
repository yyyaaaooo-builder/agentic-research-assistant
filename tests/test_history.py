import pytest
from research_assistant.core.message import Message
from research_assistant.core.history import History


@pytest.mark.parametrize(
    "message, expected", [
        pytest.param(
            Message(role="user", content ="What is 3 * 7?"),
            {"role":"user", "content":"What is 3 * 7?"},
            id="user_message"
            ),
        pytest.param(
            Message(role="assistant", content=""),
            {"role":"assistant", "content":""},
            id="empty_content"
            ),
        pytest.param(
            Message(role="assistant", tool_calls=[{"id":"call_1", "function":{"name":"calculator"}}]),
            {"role":"assistant", "tool_calls":[{"id":"call_1", "function":{"name":"calculator"}}]},
            id="assistant_tool_call"
            ),
        pytest.param(
            Message(role="tool", content="21", tool_call_id="call_1"),
            {"role":"tool", "content":"21", "tool_call_id":"call_1"},
            id="tool_result"
            ),
          
    ]
    )

def test_to_dict_renders(message, expected):
    assert message.to_dict() == expected


def test_history_preserves_order():
    history = History()
    assert history.to_openai_messages() == []

    history.add_system("You are a research assistant.")
    history.add_user("What is 12*7+5?")
    history.add_assistant(tool_calls=[{"id": "call_1", "function": {"name": "calculator"}}])
    history.add_tool_result(tool_call_id="call_1", content="89")
    history.add_assistant(content="The answer is 89.")

    roles=[m["role"] for m in history.to_openai_messages()]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_get_messages_returns_copy_of_history():
    history = History()
    history.add_user("hello world")

    returned = history.get_messages()
    returned.append("not a real message")

    assert len(history.get_messages()) == 1
