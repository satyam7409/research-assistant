import asyncio
import json
import logging
import queue
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.crew.research_crew import PipelineError, run_pipeline
from app.schemas import PipelineResult

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
app = FastAPI(title="Research Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)
RUNS = Path("data/runs")
RUNS.mkdir(parents=True, exist_ok=True)
RUN_LOCK = threading.Lock()


@app.get("/health")
def health():
    return {"status": "ok"}


def save_run(run_id: str, topic: str, result: PipelineResult) -> None:
    payload = {
        "id": run_id,
        "topic": topic,
        "created_at": datetime.now(UTC).isoformat(),
        "result": result.model_dump(mode="json"),
    }
    (RUNS / f"{run_id}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


@app.get("/research/stream")
async def research_stream(topic: str = Query(min_length=3, max_length=200)):
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another research run is already in progress.")
    events: queue.Queue[tuple[str, dict]] = queue.Queue()
    run_id = str(uuid.uuid4())

    def work():
        try:
            result = run_pipeline(topic, lambda name, data: events.put((name, data)))
            save_run(run_id, topic, result)
            events.put(("result", result.model_dump(mode="json")))
        except PipelineError as exc:
            events.put(("error", {"message": str(exc)}))
        except Exception:
            log.exception("research run failed")
            events.put(
                ("error", {"message": "Research failed. Check the server logs."})
            )
        finally:
            RUN_LOCK.release()

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while True:
            try:
                name, data = await asyncio.to_thread(events.get, True, 1)
            except queue.Empty:
                yield {"comment": "heartbeat"}
                continue
            yield {"event": name, "data": json.dumps(data)}
            if name in {"result", "error"}:
                break

    return EventSourceResponse(stream())


@app.get("/runs")
def list_runs():
    values = []
    for path in RUNS.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            values.append(
                {
                    "id": item["id"],
                    "topic": item["topic"],
                    "created_at": item["created_at"],
                }
            )
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return sorted(values, key=lambda x: x["created_at"], reverse=True)[:20]


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    path = RUNS / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(404, "Run not found")
    return json.loads(path.read_text(encoding="utf-8"))
