"""
server/app.py — FastAPI server for the Legal Agent Environment.
OpenEnv-compatible API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

try:
    from ..models import LegalAction
    from .legal_environment import LegalEnvironment, grade_episode
except ImportError:
    from models import LegalAction
    from server.legal_environment import LegalEnvironment, grade_episode


# ── FastAPI Setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal Agent OpenEnv",
    description="Environment where AI agents learn legal reasoning tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance
env = LegalEnvironment()


# ── Request Models ───────────────────────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class GraderRequest(BaseModel):
    task_id: str
    issues_found: list = []
    false_positives: int = 0
    total_steps: int = 0
    strategy_submitted: bool = False


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Legal Agent OpenEnv running",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader"]
    }


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "env": "legal-agent-env", "version": "1.0.0"}


# ── Reset Environment ────────────────────────────────────────────────────────
@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    try:
        result = env.reset(task_id=req.task_id or "easy")
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Step ─────────────────────────────────────────────────────────────────────
@app.post("/step")
def step(action: LegalAction):
    try:
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── State ────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    try:
        return env.state().model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Episode Info ─────────────────────────────────────────────────────────────
@app.get("/episode")
def episode():
    return {
        "episode_id": env.episode_id,
        "task_id": env.task_id,
        "step": env.step_count,
        "total_reward": env.total_reward,
        "done": env.done,
    }


# ── Tasks Info ───────────────────────────────────────────────────────────────
@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "task_id": "easy",
                "name": "Contract Clause Review",
                "difficulty": "easy",
                "max_steps": 20,
                "actions": ["flag_issue", "approve_clause", "suggest_fix"],
            },
            {
                "task_id": "medium",
                "name": "Legal Issue Spotting",
                "difficulty": "medium",
                "max_steps": 15,
                "actions": ["flag_issue", "identify_law"],
            },
            {
                "task_id": "hard",
                "name": "Case Strategy Building",
                "difficulty": "hard",
                "max_steps": 25,
                "actions": ["flag_issue", "identify_law", "submit_strategy"],
            },
        ],
        "action_schema": {
            "action_type": "string",
            "clause_id": "int",
            "issue_type": "string",
            "area_of_law": "string",
            "doctrine": "string",
            "suggested_fix": "string",
            "reasoning": "string",
        },
    }


# ── Grader ───────────────────────────────────────────────────────────────────
@app.post("/grader")
def grader(req: GraderRequest):
    try:
        score = grade_episode(
            task_id=req.task_id,
            issues_found=req.issues_found,
            false_positives=req.false_positives,
            total_steps=req.total_steps,
            strategy_submitted=req.strategy_submitted,
        )
        return {"task_id": req.task_id, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))