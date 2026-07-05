"""FastAPI surface for the orchestration system."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import Orchestrator

app = FastAPI(title="Agent Orchestration System", version="1.0.0")
orchestrator = Orchestrator()


class TaskRequest(BaseModel):
    task: str
    user: str = "default"
    auto_approve: bool = False


class ResolveRequest(BaseModel):
    decision: str  # approved | rejected | modified
    operator: str
    note: str = ""
    replacement_output: str = ""


class MemoryRequest(BaseModel):
    kind: str
    text: str
    user: str


@app.post("/v1/tasks")
def create_task(request: TaskRequest):
    return orchestrator.run(request.task, request.user,
                            request.auto_approve)


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    state = orchestrator.tasks.get(task_id)
    if state is None:
        raise HTTPException(404, f"unknown task {task_id}")
    return orchestrator._snapshot(state)


@app.get("/v1/tasks/{task_id}/trace")
def get_trace(task_id: str):
    trace = orchestrator.traces.get(task_id)
    if trace is None:
        raise HTTPException(404, f"unknown task {task_id}")
    return {"task": trace.task, "tree": trace.root.to_dict(),
            "totals": trace.totals()}


@app.get("/v1/approvals")
def approvals():
    return [{"escalation_id": e.escalation_id, "task_id": e.task_id,
             "level": e.level, "reason": e.reason, "context": e.context}
            for e in orchestrator.approvals.waiting()]


@app.post("/v1/approvals/{escalation_id}")
def resolve(escalation_id: str, request: ResolveRequest):
    if escalation_id not in orchestrator.approvals.items:
        raise HTTPException(404, f"unknown escalation {escalation_id}")
    return orchestrator.resolve_and_continue(
        escalation_id, request.decision, request.operator,
        request.note, request.replacement_output)


@app.get("/v1/memory/{user}")
def memory_dashboard(user: str):
    return orchestrator.long_term.dashboard(user)


@app.post("/v1/memory")
def add_memory(request: MemoryRequest):
    memory = orchestrator.long_term.remember(request.kind, request.text,
                                             request.user)
    return {"memory_id": memory.memory_id}


@app.delete("/v1/memory/{user}")
def forget(user: str):
    return {"deleted": orchestrator.long_term.forget_user(user)}


@app.get("/v1/tools")
def tools():
    return [{"name": s.name, "description": s.description,
             "allowed_agents": s.allowed_agents,
             "rate_limit_per_task": s.rate_limit_per_task}
            for s in orchestrator.registry.tools.values()]
