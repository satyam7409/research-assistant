import json
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


MAX_REVISIONS = 2


def _json(items) -> str:
    """CrewAI inputs must be plain strings/numbers/dicts, so pass objects as JSON text."""
    return json.dumps([i.model_dump() for i in items], ensure_ascii=False)


def _report_bad_ids(report: Report, evidence: list[Evidence]) -> set[str]:
    valid = {e.id for e in evidence}
    cited = {x for s in report.sections for c in s.claims for x in c.evidence_ids}
    return cited - valid


def _run_task(
    cb: EventCallback,
    name: str,
    agent: Agent,
    description: str,
    expected_output: str,
    output_model: type,
    inputs: dict[str, str],
):
    """Run one task in its own single-task crew; retry once if output is invalid."""
    for _ in range(2):
        _emit(cb, "agent_started", {"agent": name})
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            output_pydantic=output_model,
        )
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
        result = crew.kickoff(inputs=inputs)
        value = result.tasks_output[0].pydantic
        if value is not None:
            _emit(cb, "agent_finished", {"agent": name})
            return value
        _emit(cb, "warning", {"message": f"{name} returned invalid output; retrying"})
    raise PipelineError(f"{name} did not return valid output after 2 attempts")


def _review(
    cb: EventCallback,
    reviewer: Agent,
    report: Report,
    evidence: list[Evidence],
    round_no: int,
) -> ReviewResult:
    _emit(cb, "review_started", {"round": round_no})
    review = _run_task(
        cb,
        "reviewer",
        reviewer,
        description=(
            "Fact-check every claim in the REPORT using ONLY the EVIDENCE excerpts below. "
            "Never use outside knowledge. For each claim return a verdict: "
            "'supported' if the cited excerpts directly support it, 'partial' if the "
            "excerpts support only part of it or the claim overstates its source "
            "(for example absolute words like 'infinitely' or 'always' backed by one "
            "blog post), 'unsupported' if no excerpt supports it. Add a short note.\n\n"
            "EVIDENCE:\n{evidence_json}\n\nREPORT:\n{report_json}"
        ),
        expected_output="ReviewResult JSON with one verdict per claim",
        output_model=ReviewResult,
        inputs={
            "evidence_json": _json(evidence),
            "report_json": report.model_dump_json(),
        },
    )
    problems = [v for v in review.claim_verdicts if v.status != "supported"]
    review.verdict = "revise" if problems else "approve"
    if not review.claim_verdicts:
        _emit(cb, "warning", {"message": "Reviewer returned no claim verdicts"})
    counts = {
        status: sum(v.status == status for v in review.claim_verdicts)
        for status in ("supported", "partial", "unsupported")
    }
    _emit(cb, "review_finished", {"verdict": review.verdict, **counts})
    return review


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
        goal="Collect accurate, sourced evidence about {topic}",
        backstory="You only report what sources actually say. You never invent URLs.",
        tools=[web_search],
        llm=research_llm,
    )
    analyst = Agent(
        role="Research Analyst",
        goal="Find insights that are strictly supported by the evidence",
        backstory="You cite only evidence ids that exist.",
        llm=research_llm,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Write a structured, cited report",
        backstory="You never invent evidence ids and never overstate a source.",
        llm=writer_llm,
    )
    reviewer = Agent(
        role="Fact Checker",
        goal="Check whether each claim is supported by the provided evidence",
        backstory="You are strict, and you judge only against the excerpts you are given.",
        llm=writer_llm,
    )

    raw = _run_task(
        on_event,
        "researcher",
        researcher,
        description=(
            "Research {topic} using the web_search tool. Run 2-4 different searches. "
            "Return 8-10 evidence items. Each needs the real URL, the title and an "
            "excerpt copied from the search results. Never invent URLs."
        ),
        expected_output="EvidenceList JSON",
        output_model=EvidenceList,
        inputs={"topic": topic},
    ).items
    evidence = normalize_evidence(raw)
    if len(evidence) < 3:
        raise PipelineError("Not enough usable sources found. Try another topic.")
    evidence_json = _json(evidence)
    insights = _run_task(
        on_event,
        "analyst",
        analyst,
        description=(
            "Analyze the evidence about {topic}. Return 3-5 insights. Each insight "
            "must cite ONLY evidence ids that appear below.\n\nEVIDENCE:\n{evidence_json}"
        ),
        expected_output="InsightList JSON",
        output_model=InsightList,
        inputs={"topic": topic, "evidence_json": evidence_json},
    ).insights
    report = _run_task(
        on_event,
        "writer",
        writer,
        description=(
            "Write a report on {topic} as structured JSON: a title and sections, each "
            "with claims. Each claim has text and the evidence ids that support it. "
            "Use ONLY facts from the evidence, never overstate a source, and cite only "
            "ids that exist.\n\nEVIDENCE:\n{evidence_json}\n\nINSIGHTS:\n{insights_json}"
        ),
        expected_output="Report JSON",
        output_model=Report,
        inputs={
            "topic": topic,
            "evidence_json": evidence_json,
            "insights_json": _json(insights),
        },
    )
    review: ReviewResult | None = None
    rounds = 0
    while True:
        bad_ids = _report_bad_ids(report, evidence)
        if bad_ids:
            _emit(
                on_event, "warning", {"message": f"Nonexistent ids: {sorted(bad_ids)}"}
            )
            problems: list[dict] = [
                {"issue": f"Cites nonexistent evidence ids {sorted(bad_ids)}"}
            ]
        else:
            review = _review(on_event, reviewer, report, evidence, rounds + 1)
            problems = [
                v.model_dump() for v in review.claim_verdicts if v.status != "supported"
            ]
            if not problems:
                break
        if rounds >= MAX_REVISIONS:
            break
        rounds += 1
        _emit(on_event, "revision_started", {"round": rounds})
        report = _run_task(
            on_event,
            "writer",
            writer,
            description=(
                "Revise the report on {topic}. Fix or remove ONLY the problem claims "
                "listed below, and keep every supported claim unchanged. Use only "
                "evidence ids that exist.\n\nPROBLEMS:\n{problems_json}\n\n"
                "EVIDENCE:\n{evidence_json}\n\nCURRENT REPORT:\n{report_json}"
            ),
            expected_output="Report JSON",
            output_model=Report,
            inputs={
                "topic": topic,
                "problems_json": json.dumps(problems, ensure_ascii=False),
                "evidence_json": evidence_json,
                "report_json": report.model_dump_json(),
            },
        )
    if review is None:
        review = ReviewResult(verdict="revise", claim_verdicts=[])
    result = PipelineResult(
        topic=topic,
        report=report,
        markdown=render_markdown(report),
        evidence=evidence,
        insights=insights,
        review=review,
        revision_rounds=rounds,
    )
    result.bad_citations = sorted(find_bad_citations(result))
    _emit(on_event, "pipeline_finished", {"evidence_count": len(evidence)})
    return result


def run_research(topic: str) -> dict:
    result = run_pipeline(topic)
    return {
        "report": result.markdown,
        "evidence": result.evidence,
        "insights": result.insights,
    }
