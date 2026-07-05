# Agent Orchestration System with Tool Use, Memory, and Human in the Loop

A multi agent platform where a supervisor decomposes complex tasks,
specialist agents execute them with real tool calls, a reviewer validates
every output, persistent memory makes future planning smarter, and the
system escalates to a human operator whenever confidence is low or an
action is sensitive. Every decision lands in a full execution trace.

Live demo: https://mansimengde17.github.io/Agent-Orchestration-System-with-Tool-Use-Memory-and-Human-in-the-Loop/

## Why this exists

Most agent demos are a single model in a loop. Production agent systems
need the parts around the loop: permissioned tools, a reviewer that
catches bad output before it ships, memory that survives the session,
an operator who can approve, modify, or take over, and a trace of every
decision for debugging. That surrounding machinery is this project.

## Architecture

```
                      supervisor
            (memory recall -> plan -> confidence)
                 |            low confidence -> approve_plan
        +--------+---------+----------+
     research   analysis   writing   code        specialists
        |          |          |        |          (permissioned tools)
     web_search  query_db  write_file run_code
        +--------+---------+----------+
                    reviewer
        (score -> approve / feedback retry / escalate)
                       |
     sensitive step -> approve_action   double failure -> take_over
                       |
              synthesis -> delivery -> memory write
```

- `src/orchestrator/tools.py` tool registry with per agent permissions,
  per task rate limits, and full invocation logging
- `src/orchestrator/agents.py` supervisor planning with dependencies,
  four specialists, and a reviewer with a quality floor
- `src/orchestrator/memory.py` working memory per task plus long term
  semantic memory with importance scoring, consolidation, decay, and a
  delete endpoint for user data requests
- `src/orchestrator/hitl.py` escalation trigger table and the approval
  queue with four levels: notify, approve_action, approve_plan, take_over
- `src/orchestrator/graph.py` the state machine with conditional edges:
  retry on failure, feedback round on rejection, pause on escalation
- `src/orchestrator/tracing.py` trace tree with tokens, cost, latency,
  and status per node, plus a replay harness
- `src/orchestrator/api.py` FastAPI endpoints for tasks, approvals,
  traces, memory, and tools

## Quick start

```bash
pip install -r requirements.txt
python demo.py                       # full lifecycle showcase
python -m unittest discover tests
uvicorn orchestrator.api:app --app-dir src --port 8000
```

Drive an escalation manually through the API:

```bash
curl -s localhost:8000/v1/tasks -H "Content-Type: application/json" \
  -d '{"task": "Research the EV market and email the report"}'
# -> status waiting_for_human with an escalation id
curl -s localhost:8000/v1/approvals
curl -s localhost:8000/v1/approvals/esc-001 \
  -H "Content-Type: application/json" \
  -d '{"decision": "approved", "operator": "oncall"}'
```

## Docker

```bash
docker compose up --build
```

## Notes

Specialist reasoning and tools are deterministic simulations so the
orchestration logic, escalation paths, and trace tree are fully testable
offline. Swapping in live LLM calls touches only `agents.py`; LangGraph,
Redis, PostgreSQL, and ChromaDB map one to one onto the graph, working
memory, task store, and long term memory modules.
