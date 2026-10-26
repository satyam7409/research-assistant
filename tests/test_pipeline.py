from app.crew.research_crew import (
    find_bad_citations,
    normalize_evidence,
    render_markdown,
)
from app.schemas import (
    Claim,
    Evidence,
    Insight,
    PipelineResult,
    Report,
    ReviewResult,
    Section,
)


def test_evidence_is_deduped_and_renumbered():
    items = [
        Evidence(id="x", url="HTTPS://a.test/", title="a", excerpt="ok"),
        Evidence(id="x", url="https://a.test", title="dup", excerpt="ok"),
        Evidence(id="x", url="b", title="empty", excerpt=" "),
    ]
    assert [x.id for x in normalize_evidence(items)] == ["e1"]


def test_render_and_bad_citations():
    report = Report(
        title="T",
        sections=[
            Section(
                heading="S", claims=[Claim(text="Claim", evidence_ids=["e1", "e9"])]
            )
        ],
    )
    result = PipelineResult(
        topic="abc",
        report=report,
        markdown=render_markdown(report),
        evidence=[Evidence(id="e1", url="u", title="U", excerpt="e")],
        insights=[Insight(statement="i", evidence_ids=["e2"])],
        review=ReviewResult(verdict="approve", claim_verdicts=[]),
        revision_rounds=0,
    )
    assert "[e1]" in result.markdown and find_bad_citations(result) == {"e2", "e9"}
