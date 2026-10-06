from dataclasses import dataclass
from typing import Any, Annotated, TypedDict, Literal
from pydantic import BaseModel, ConfigDict, Field
from langgraph.graph.message import add_messages
from .routing_contracts import RefundAssessment


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
    message_db_ids: dict[str,int]
    context_anchors: dict
    conversation_id: int
    user_id: str
    turn_id: str
    raw_question: str
    question: str
    intent: str
    route: str
    history_recall: bool
    evidence: list
    evidence_snapshot: dict
    confidence: dict
    evidence_useful: bool | None
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
    resolution: dict
    intent_confidence: float
    intent_model_tier: str
    intent_upgraded: bool
    order_id: str | None
    order_snapshot: dict
    selection: dict
    expanded_queries: list[str]
    refund_assessment: dict
    ticket_request: dict
    ticket_confirmation: dict
    tool_cursor: int
    tool_observations: list


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=64)
    args: dict
    id: str = Field(min_length=1, max_length=64)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["tools", "answer", "clarify", "stop"]
    tool_calls: list[ToolRequest] = Field(default_factory=list, max_length=5)
    actions: list[Literal["handoff", "ticket", "refund"]] = Field(default_factory=list, max_length=3)
    reason: str = Field(min_length=1, max_length=300)
    refund_assessment: RefundAssessment | None = None


@dataclass
class TurnContext:
    agent: AgentDependencies
    budget: Any
    save_message: Any = None
    retrieve: Any = None
    record_low_confidence: Any = None
    issue_actions: Any = None
    retrieve_queries: Any = None
    intent_small_model: Any = None
    refund_defaults: Any = None
    message_db_ids: Any = None
    history_projection: Any = None
    context_config: Any = None
    confidence_config: Any = None
    evidence_check: Any = None
    observability: Any = None
