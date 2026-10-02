"""Validated model outputs; runtime diagnostics are kept outside their JSON."""
from dataclasses import dataclass
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Intent = Literal['物流','订单','商品咨询','退款退货','售后','投诉','闲聊','其他']


class StrictOutput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, str_strip_whitespace=True, allow_inf_nan=False)


class Resolution(StrictOutput):
    question: str = Field(min_length=1,max_length=4096)
    reference_status: Literal['resolved','clear','ambiguous']
    scope: Literal['general','order_specific']
    order_id: str | None = Field(default=None,min_length=1,max_length=64)
    order_source: Literal['current','history','none']

    @model_validator(mode='after')
    def coherent_slot(self):
        if (self.order_id is None) != (self.order_source == 'none'):
            raise ValueError('order id and source must agree')
        if self.reference_status == 'ambiguous' and self.order_id is not None:
            raise ValueError('ambiguous reference cannot choose an order')
        return self


class IntentDecision(StrictOutput):
    intent: Intent
    confidence: float = Field(ge=0,le=1)


class ExpandedQueries(StrictOutput):
    queries: list[Annotated[str,Field(min_length=1,max_length=512)]] = Field(min_length=1,max_length=3)


class RefundAssessment(StrictOutput):
    status: Literal['eligible','ineligible','needs_review']
    reason: str = Field(min_length=1,max_length=500)
    evidence_numbers: list[Annotated[int,Field(gt=0)]] = Field(max_length=10)

    @model_validator(mode='after')
    def eligibility_needs_evidence(self):
        if self.status == 'eligible' and not self.evidence_numbers:
            raise ValueError('eligibility requires evidence')
        self.evidence_numbers = list(dict.fromkeys(self.evidence_numbers))
        return self


@dataclass(frozen=True)
class IntentRun:
    decision: IntentDecision
    model_tier: Literal['large','small','rule']
    upgraded: bool
