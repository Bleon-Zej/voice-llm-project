from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        if "role" not in data or "content" not in data:
            raise ValueError("Message muss role und content enthalten")

        ts_data = data.get("timestamp")
        if isinstance(ts_data, (int, float)):
            timestamp = datetime.fromtimestamp(ts_data)
        elif isinstance(ts_data, str):
            timestamp = datetime.fromisoformat(ts_data)
        else:
            timestamp = datetime.now()  # Fallback

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
        )

    def to_ollama_format(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}
