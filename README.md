# Research Assistant

Small single-user FastAPI + CrewAI + React demo that researches a topic, preserves source excerpts, and renders a reviewed cited report.

```text
React/Vite ── SSE ──> FastAPI ── thread ──> CrewAI
   │                    │                    ├─ Researcher + Tavily
   └── /runs history ───┘                    ├─ Analyst
                                             ├─ Writer
                                             └─ Reviewer/revision
```

## Windows setup

Install Python 3.12, `uv`, Node.js, and npm. From PowerShell:

```powershell
uv sync
Copy-Item .env.example .env
cd frontend; npm install; cd ..
```

Use `scripts\dev.ps1 -Fake` for a no-key demo, then open http://localhost:5173. For real mode, put provider keys in `.env` and run `scripts\dev.ps1`.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI/CrewAI key |
| `TAVILY_API_KEY` | Web search key |
| `LLM_MODEL` | Default model |
| `RESEARCH_MODEL` | Optional researcher/analyst override |
| `WRITER_MODEL` | Optional writer/reviewer override |
| `FAKE_PIPELINE` | `true` for local demo mode |
| `CORS_ORIGIN` | Frontend origin |

Run tests with `uv run pytest -q`; lint with `uv run ruff check .`.

## Assumptions

- This is intentionally single-user and in-process; one pipeline run is allowed at a time.
- Finished JSON files under `data/runs` are local demo history and are not a database.
- A disconnected browser does not cancel the worker thread because CrewAI cannot be cancelled cleanly.

## Design decisions

In-process SSE is easy to demonstrate locally and is enough for one user. Production would add a durable job queue, shared state, cancellation, and multiple workers. Evidence IDs are assigned after URL deduplication in Python so an LLM cannot create duplicate or invalid identifiers. The Reviewer receives no search tool: it must judge only the supplied excerpts. Revisions are capped at two rounds so a bad model response cannot create an endless demo.
