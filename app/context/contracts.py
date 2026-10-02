from dataclasses import dataclass
from langchain_core.messages import BaseMessage

@dataclass(frozen=True)
class ContextConfig:
    window: int = 1_000_000
    output: int = 4_000
    user_input: int = 4_000
    steps: int = 5
    tool_result: int = 2_400
    rerank_k: int = 10
    desired_turns: int = 40
    prefix: int = 1_600
    evidence_per_hit: int = 180
    summary_reserve: int = 600
    call_overhead: int = 400

@dataclass(frozen=True)
class BudgetBreakdown:
    fixed: int
    transient: int
    steady: int
    history: int
    layer1: int
    layer2: int
    summary: int
    margin: int

@dataclass(frozen=True)
class LayerAnchors:
    summary_upto: int = 0
    layer1_from: int = 0

@dataclass(frozen=True)
class HistoryRecord:
    message: BaseMessage
    owner_msg_id: int
    turn_id: str

@dataclass(frozen=True)
class SummarySegment:
    seq: int
    from_msg_id: int
    upto_msg_id: int
    content: str

@dataclass(frozen=True)
class HistoryProjection:
    anchors: LayerAnchors
    layer1: tuple[BaseMessage, ...]
    layer2: tuple[BaseMessage, ...]
    summary: str
    layer2_tokens: int
    omitted_ranges: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class SummaryBatch:
    conversation_id: int
    expected_summary_upto: int
    upto_msg_id: int
    source: tuple[HistoryRecord, ...]
    previous_segments: tuple[SummarySegment, ...]
