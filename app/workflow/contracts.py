from dataclasses import dataclass
from typing import Any, Annotated, TypedDict, Literal
from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph.message import add_messages


@dataclass
class AgentDependencies:
    model: Any
    tools: list
    executor: Any


@dataclass
class AgentResult:
    messages: list
    stop_reason: str


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    conversation_id: int
    user_id: str
    turn_id: str
    raw_question: str
    question: str
    intent: str
    route: str
    evidence: list
    citations: list
    actions: list
    decision: dict
    final_text: str
    stop_reason: str
    budget_usage: dict
    tool_steps: int
    node_trace: list
    gate_allowed: bool
    gate_reason: str


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    args: dict
    id: str = Field(min_length=1, max_length=64)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tools", "answer", "clarify", "stop"]
    tool_calls: list[ToolRequest] = Field(default_factory=list, max_length=5)
    actions: list[Literal["handoff", "ticket"]] = Field(default_factory=list, max_length=2)
    reason: str = Field(min_length=1, max_length=300)


@dataclass
class TurnContext:
    agent: AgentDependencies
    budget: Any
    save_message: Any = None
    retrieve: Any = None
    record_low_confidence: Any = None
    issue_actions: Any = None
