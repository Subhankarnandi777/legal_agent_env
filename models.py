"""
models.py — Typed Pydantic models for the Legal Agent Environment.
These are shared between server and client.
"""

from __future__ import annotations
from typing import Optional, List, Any
from pydantic import BaseModel, Field


# ── ACTION ────────────────────────────────────────────────────────────────────
class LegalAction(BaseModel):
    """
    What the agent can DO in the environment.

    action_type options:
      - "flag_issue"     : mark a clause / fact as having a legal problem
      - "approve_clause" : mark a clause as legally sound
      - "suggest_fix"    : propose a corrected version of a clause
      - "identify_law"   : name the area of law / doctrine that applies
      - "submit_strategy": final case strategy submission (Task 3)
    """
    action_type: str                        # required — one of the above
    clause_id: Optional[int] = None        # which clause (Task 1)
    issue_type: Optional[str] = None       # e.g. "missing_liability_cap"
    area_of_law: Optional[str] = None      # e.g. "contract_law"
    doctrine: Optional[str] = None         # e.g. "consideration"
    suggested_fix: Optional[str] = None    # corrected clause text
    reasoning: Optional[str] = None        # agent's explanation (optional)


# ── OBSERVATION ───────────────────────────────────────────────────────────────
class ClauseStatus(BaseModel):
    """Status of a single contract clause."""
    id: int
    text: str
    status: str = "unreviewed"   # unreviewed | flagged | approved | fixed


class LegalObservation(BaseModel):
    """
    What the agent SEES at each step.
    """
    task_id: str                            # "easy" | "medium" | "hard"
    task_description: str                   # plain-English instruction
    document_text: str                      # full contract / fact pattern
    clauses: List[ClauseStatus] = Field(default_factory=list)
    issues_found: List[str] = Field(default_factory=list)
    issues_remaining: int = 0
    last_action_feedback: str = ""
    reward_last_step: float = 0.0
    step_count: int = 0
    max_steps: int = 20
    done: bool = False
    hint: str = ""


# ── STATE ─────────────────────────────────────────────────────────────────────
class LegalState(BaseModel):
    """
    Internal episode metadata (returned by GET /state).
    """
    episode_id: str
    task_id: str
    step: int
    total_reward: float
    done: bool
    issues_total: int
    issues_found_count: int
    false_positives: int


# ── STEP RESULT (what env.step() returns) ─────────────────────────────────────
class StepResult(BaseModel):
    observation: LegalObservation
    reward: float
    done: bool
    info: dict = {}