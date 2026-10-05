from research_assistant.core.message import Message


class History:
    def __init__(self):
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)

    def add_system(self, content: str) -> None:
        self.add(Message(role="system", content=content))

    def add_user(self, content: str) -> None:
        self.add(Message(role="user", content=content))

    def add_assistant(self, content: str | None = None, tool_calls: list[dict] | None = None) -> None:
        self.add(Message(role="assistant", content=content, tool_calls=tool_calls))

    def add_tool_result(self, content: str, tool_call_id: str) -> None:
        self.add(Message(role="tool", content=content, tool_call_id=tool_call_id))

    def to_openai_messages(self) -> list[dict]:
        return [m.to_dict() for m in self._messages]

    def get_messages(self) -> list[Message]:
        return self._messages.copy()
