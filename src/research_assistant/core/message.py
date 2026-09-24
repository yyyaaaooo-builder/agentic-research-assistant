from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict | None = None

    def to_dict(self) -> dict:
        data = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data
