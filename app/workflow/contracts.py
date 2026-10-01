from dataclasses import dataclass
from typing import Any


@dataclass
class AgentDependencies:
    model: Any
    tools: list
    executor: Any


@dataclass
class AgentResult:
    messages: list
    stop_reason: str
