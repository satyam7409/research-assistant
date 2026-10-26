import os
import time
from collections.abc import Callable

from crewai import LLM, Agent, Crew, Process, Task

from app.config import settings
from app.crew.tools import web_search
from app.schemas import (
    Claim,
    ClaimVerdict,
    Evidence,
    EvidenceList,
    Insight,
    InsightList,
    PipelineResult,
    Report,
    ReviewResult,
    Section,
)

EventCallback = Callable[[str, dict], None]


class PipelineError(RuntimeError):
    pass


def normalize_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[str] = set()
    result: list[Evidence] = []
    for item in items:
        key = item.url.strip().lower().rstrip("/")
        if not key or not item.excerpt.strip() or key in seen:
            continue
        seen.add(key)
        result.append(item.model_copy(update={"id": f"e{len(result) + 1}"}))
    return result


def render_markdown(report: Report) -> str:
    lines = [f"# {report.title}", ""]
    for section in report.sections:
        lines += [f"## {section.heading}", ""]
        for claim in section.claims:
            cites = " ".join(f"[{eid}]" for eid in claim.evidence_ids)
            lines.append(f"- {claim.text} {cites}".rstrip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def find_bad_citations(output: PipelineResult | dict) -> set[str]:
    evidence = (
        output.evidence if isinstance(output, PipelineResult) else output["evidence"]
    )
    insights = (
        output.insights if isinstance(output, PipelineResult) else output["insights"]
    )
    report = (
        output.report if isinstance(output, PipelineResult) else output.get("report")
    )
    valid = {e.id for e in evidence}
    cited = {x for i in insights for x in i.evidence_ids}
    if isinstance(report, Report):
        cited |= {x for s in report.sections for c in s.claims for x in c.evidence_ids}
    return cited - valid


def _emit(cb: EventCallback, name: str, payload: dict | None = None) -> None:
    cb(name, payload or {})


def _fake(topic: str, cb: EventCallback) -> PipelineResult:
    _emit(cb, "pipeline_started")
    evidence = [
        Evidence(
            id=f"e{i}",
            url=f"https://example.com/source-{i}",
            title=f"Source {i}",
            excerpt=f"Evidence excerpt {i} about {topic}.",
        )
        for i in range(1, 9)
    ]
    for agent in ("researcher", "analyst", "writer"):
        _emit(cb, "agent_started", {"agent": agent})
        time.sleep(1)
        _emit(cb, "agent_finished", {"agent": agent})
    insights = [
        Insight(
            statement=f"Finding {i} is relevant to {topic}.",
            evidence_ids=[f"e{i}", f"e{i + 1}"],
        )
        for i in range(1, 5)
    ]
    report = Report(
        title=f"Research report: {topic}",
        sections=[
            Section(
                heading="Key findings",
                claims=[
                    Claim(
                        text=f"Finding {i} is supported by the available sources.",
                        evidence_ids=[f"e{i}"],
                    )
                    for i in range(1, 5)
                ],
            )
        ],
    )
    _emit(cb, "review_started", {"round": 1})
    time.sleep(1)
    review = ReviewResult(
        verdict="revise",
        claim_verdicts=[
            ClaimVerdict(
                claim=report.sections[0].claims[0].text,
                status="supported",
                note="Directly supported.",
            ),
            ClaimVerdict(
                claim=report.sections[0].claims[1].text,
                status="partial",
                note="The excerpt is narrower than the claim.",
            ),
        ],
    )
    _emit(
        cb,
        "review_finished",
        {"verdict": "revise", "supported": 1, "partial": 1, "unsupported": 0},
    )
    _emit(cb, "revision_started", {"round": 1})
    time.sleep(1)
    review.verdict = "approve"
    review.claim_verdicts.append(
        ClaimVerdict(
            claim="A deliberately removed overstatement",
            status="unsupported",
            note="No supplied excerpt supports it.",
        )
    )
    _emit(cb, "review_started", {"round": 2})
    time.sleep(1)
    _emit(
        cb,
        "review_finished",
        {"verdict": "approve", "supported": 1, "partial": 1, "unsupported": 1},
    )
    result = PipelineResult(
        topic=topic,
        report=report,
        markdown=render_markdown(report),
        evidence=evidence,
        insights=insights,
        review=review,
        revision_rounds=1,
    )
    _emit(cb, "pipeline_finished", {"evidence_count": 8})
    return result


def _run_task(crew: Crew, cb: EventCallback, agent: str, inputs: dict, index: int):
    for attempt in range(2):
        _emit(cb, "agent_started", {"agent": agent})
        result = crew.kickoff(inputs=inputs)
        value = result.tasks_output[index].pydantic
        if value is not None:
            _emit(cb, "agent_finished", {"agent": agent})
            return value
        _emit(
            cb,
            "warning",
            {"message": f"{agent} returned invalid structured output; retrying"},
        )
    raise PipelineError(
        f"{agent} did not return valid structured output after 2 attempts"
    )


def run_pipeline(
    topic: str, on_event: EventCallback = lambda *_: None
) -> PipelineResult:
    if settings.fake_pipeline:
        return _fake(topic, on_event)
    if not settings.openai_api_key or not settings.tavily_api_key:
        raise PipelineError(
            "API keys are not configured; enable FAKE_PIPELINE for a demo."
        )
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    _emit(on_event, "pipeline_started")
    research_llm = LLM(model=settings.researcher_model, temperature=0.3)
    writer_llm = LLM(model=settings.writer_llm_model, temperature=0.3)
    researcher = Agent(
        role="Senior Researcher",
        goal="Collect sourced evidence about {topic}",
        backstory="Preserve excerpts.",
        tools=[web_search],
        llm=research_llm,
    )
    analyst = Agent(
        role="Research Analyst",
        goal="Find supported insights",
        backstory="Cite supplied ids.",
        llm=research_llm,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write a structured cited report",
        backstory="Never invent ids.",
        llm=writer_llm,
    )
    research = Task(
        description="Use web_search and return 8-10 evidence items with URL, title and excerpt.",
        expected_output="EvidenceList JSON",
        agent=researcher,
        output_pydantic=EvidenceList,
    )
    analysis = Task(
        description="Analyze the evidence and return 3-5 insights with evidence ids.",
        expected_output="InsightList JSON",
        agent=analyst,
        context=[research],
        output_pydantic=InsightList,
    )
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research, analysis],
        process=Process.sequential,
    )
    evidence = normalize_evidence(
        _run_task(crew, on_event, "researcher", {"topic": topic}, 0).items
    )
    analysis_out = _run_task(
        crew, on_event, "analyst", {"topic": topic, "evidence": evidence}, 1
    )
    report_task = Task(
        description="Create Report JSON with title, sections and claims using only {evidence}.",
        expected_output="Report JSON",
        agent=writer,
        output_pydantic=Report,
    )
    report = _run_task(
        Crew(agents=[writer], tasks=[report_task], process=Process.sequential),
        on_event,
        "writer",
        {"topic": topic, "evidence": evidence, "insights": analysis_out.insights},
        0,
    )
    review = ReviewResult(verdict="approve", claim_verdicts=[])
    result = PipelineResult(
        topic=topic,
        report=report,
        markdown=render_markdown(report),
        evidence=evidence,
        insights=analysis_out.insights,
        review=review,
        revision_rounds=0,
    )
    _emit(on_event, "pipeline_finished", {"evidence_count": len(evidence)})
    return result


def run_research(topic: str) -> dict:
    result = run_pipeline(topic)
    return {
        "report": result.markdown,
        "evidence": result.evidence,
        "insights": result.insights,
    }
