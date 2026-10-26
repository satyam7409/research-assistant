from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str
    url: str
    title: str
    excerpt: str


class EvidenceList(BaseModel):
    items: list[Evidence]


class Insight(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class InsightList(BaseModel):
    insights: list[Insight]


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    claims: list[Claim]


class Report(BaseModel):
    title: str
    sections: list[Section]


class ClaimVerdict(BaseModel):
    claim: str
    status: Literal["supported", "partial", "unsupported"]
    note: str


class ReviewResult(BaseModel):
    verdict: Literal["approve", "revise"]
    claim_verdicts: list[ClaimVerdict]


class PipelineResult(BaseModel):
    topic: str
    report: Report
    markdown: str
    evidence: list[Evidence]
    insights: list[Insight]
    review: ReviewResult
    revision_rounds: int
    bad_citations: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    topic: str
    report: str
    evidence: list[Evidence]
    insights: list[Insight]
    bad_citations: list[str]
