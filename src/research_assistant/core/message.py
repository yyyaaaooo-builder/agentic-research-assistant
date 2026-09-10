from typing import Literal, Optional
from pydantic import Field, BaseModel
from datetime import datetime

MessageRole = Literal["user", "assistant", "system", "tool"]

class Message(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[list[dict]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        data = {
            "role": self.role
            }
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data



