"""
server/legal_environment.py — Core Legal Agent Environment logic.
OpenEnv-compatible environment for legal reasoning agents.
"""

from __future__ import annotations
import uuid
from typing import Optional

from models import LegalAction, LegalObservation, LegalState, StepResult, ClauseStatus


# ── TASK DATA ────────────────────────────────────────────────────────────────

TASK_EASY = {
    "task_id": "easy",
    "task_description": "Review contract and identify problematic clauses.",
    "document_text": "Software Services Agreement with multiple clauses...",
    "clauses": [
        {"id": 1, "text": "PARTIES clause — defines the two parties.", "has_issue": False},
        {"id": 2, "text": "SERVICES clause — no written scope required.", "has_issue": True, "issue_type": "vague_scope"},
        {"id": 3, "text": "PAYMENT clause — 50% per day penalty.", "has_issue": True, "issue_type": "unenforceable_penalty"},
        {"id": 4, "text": "IP clause — work product belongs to Provider.", "has_issue": True, "issue_type": "ip_ownership_risk"},
        {"id": 5, "text": "LIABILITY clause — blanket exclusion.", "has_issue": True, "issue_type": "overbroad_disclaimer"},
        {"id": 6, "text": "TERMINATION clause — no notice period.", "has_issue": True, "issue_type": "missing_notice_period"},
        {"id": 7, "text": "GOVERNING LAW vague.", "has_issue": True, "issue_type": "missing_governing_law"},
    ],
    "max_steps": 20,
    "real_issue_count": 6,
}

TASK_MEDIUM = {
    "task_id": "medium",
    "task_description": "Identify legal issues in fact pattern.",
    "document_text": "Rivera v. Northgate Properties fact pattern...",
    "real_issues": [
        {"issue_type": "implied_warranty_of_habitability", "area_of_law": "landlord_tenant_law"},
        {"issue_type": "retaliatory_eviction", "area_of_law": "landlord_tenant_law"},
        {"issue_type": "unlawful_entry", "area_of_law": "landlord_tenant_law"},
        {"issue_type": "unenforceable_waiver_clause", "area_of_law": "contract_law"},
        {"issue_type": "promissory_estoppel", "area_of_law": "contract_law"},
    ],
    "max_steps": 15,
    "real_issue_count": 5,
}

TASK_HARD = {
    "task_id": "hard",
    "task_description": "Build defense strategy for litigation case.",
    "document_text": "Harmon v. DataStream case file...",
    "key_defenses": [
        "material_breach_by_plaintiff",
        "merger_clause_bars_misrepresentation",
        "opinion_privilege_tortious_interference",
        "statute_of_limitations",
    ],
    "max_steps": 25,
    "real_issue_count": 4,
}

TASKS = {
    "easy": TASK_EASY,
    "medium": TASK_MEDIUM,
    "hard": TASK_HARD,
}


# ── ENVIRONMENT ──────────────────────────────────────────────────────────────

class LegalEnvironment:

    def __init__(self):
        self.reset()

    # ── RESET ────────────────────────────────────────────────────────────────
    def reset(self, task_id: str = "easy") -> StepResult:
        self.task_id = task_id if task_id in TASKS else "easy"
        self.task = TASKS[self.task_id]

        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

        self.issues_found = set()
        self.false_positives = 0
        self.strategy_submitted = False

        obs = self._build_observation("Episode started.")
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    # ── STEP ─────────────────────────────────────────────────────────────────
    def step(self, action: LegalAction) -> StepResult:
        if self.done:
            obs = self._build_observation("Episode already finished.")
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        self.step_count += 1

        reward, feedback = self._process_action(action)
        self.total_reward += reward

        # End conditions
        if self.task_id != "hard" and len(self.issues_found) >= self.task["real_issue_count"]:
            self.done = True
            feedback += " ✓ All issues identified."

        if self.step_count >= self.task["max_steps"]:
            self.done = True
            feedback += " Max steps reached."

        if self.task_id == "hard" and self.strategy_submitted:
            self.done = True
            feedback += " Strategy submitted."

        obs = self._build_observation(feedback)

        return StepResult(
            observation=obs,
            reward=round(reward, 3),
            done=self.done,
            info={
                "total_reward": round(self.total_reward, 3),
                "step": self.step_count
            },
        )

    # ── STATE ────────────────────────────────────────────────────────────────
    def state(self) -> LegalState:
        return LegalState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step=self.step_count,
            total_reward=self.total_reward,
            done=self.done,
            issues_total=self.task["real_issue_count"],
            issues_found_count=len(self.issues_found),
            false_positives=self.false_positives,
        )

    # ── PROCESS ACTION ───────────────────────────────────────────────────────
    def _process_action(self, action: LegalAction):

        if self.task_id == "easy":
            return self._process_easy(action)

        if self.task_id == "medium":
            return self._process_medium(action)

        if self.task_id == "hard":
            return self._process_hard(action)

        return 0.0, "Invalid task."

    # ── EASY TASK ────────────────────────────────────────────────────────────
    def _process_easy(self, action: LegalAction):
        clauses = {c["id"]: c for c in self.task["clauses"]}

        if action.action_type != "flag_issue":
            return -0.05, "Use flag_issue for this task."

        cid = action.clause_id

        if cid is None or cid not in clauses:
            self.false_positives += 1
            return -0.1, "Invalid clause."

        clause = clauses[cid]

        if not clause["has_issue"]:
            self.false_positives += 1
            return -0.15, "Clause has no issue."

        if cid in self.issues_found:
            return -0.05, "Already flagged."

        self.issues_found.add(cid)

        # Bonus for correct issue type
        if action.issue_type:
            expected = clause.get("issue_type")
            if expected and action.issue_type.lower() == expected.lower():
                return 0.30, f"Correct issue and type: {expected}"

        return 0.20, "Correct issue flagged."

    # ── MEDIUM TASK ──────────────────────────────────────────────────────────
    def _process_medium(self, action: LegalAction):
        real = {i["issue_type"] for i in self.task["real_issues"]}

        if action.action_type == "flag_issue":
            if action.issue_type in real:
                if action.issue_type in self.issues_found:
                    return -0.05, "Already identified."

                self.issues_found.add(action.issue_type)
                return 0.25, "Correct legal issue."

            self.false_positives += 1
            return -0.15, "Incorrect issue."

        return -0.05, "Unknown action."

    # ── HARD TASK ────────────────────────────────────────────────────────────
    def _process_hard(self, action: LegalAction):
        defenses = set(self.task["key_defenses"])

        if action.action_type == "flag_issue":
            if action.issue_type in defenses:
                if action.issue_type in self.issues_found:
                    return -0.05, "Already identified."
                self.issues_found.add(action.issue_type)
                return 0.25, "Defense identified."
            return -0.1, "Wrong defense."

        if action.action_type == "submit_strategy":
            self.strategy_submitted = True
            coverage = len(self.issues_found) / len(defenses)
            return 0.3 * coverage, "Strategy submitted."

        return -0.05, "Unknown action."

    # ── OBSERVATION ──────────────────────────────────────────────────────────
    def _build_observation(self, feedback):
        clauses = []
        if self.task_id == "easy":
            clauses = [
                ClauseStatus(id=c["id"], text=c["text"], status="unreviewed")
                for c in self.task["clauses"]
            ]

        return LegalObservation(
            task_id=self.task["task_id"],
            task_description=self.task["task_description"],
            document_text=self.task["document_text"],
            clauses=clauses,
            issues_found=[str(i) for i in self.issues_found],
            issues_remaining=self.task["real_issue_count"] - len(self.issues_found),
            last_action_feedback=feedback,
            reward_last_step=0.0,
            step_count=self.step_count,
            max_steps=self.task["max_steps"],
            done=self.done,
            hint="",
        )


# ── GRADER ───────────────────────────────────────────────────────────────────
def grade_episode(task_id: str, issues_found: list, false_positives: int,
                  total_steps: int, strategy_submitted: bool = False) -> float:
    """
    Returns a score strictly in the open interval (0, 1).
    The Meta/OpenEnv Phase-2 validator requires 0 < score < 1
    (not 0.0 and not 1.0).
    """
    task = TASKS.get(task_id, TASK_EASY)
    total = task["real_issue_count"]

    precision = max(0.0, len(issues_found) - false_positives) / max(1, len(issues_found))
    recall = len(issues_found) / total

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    efficiency = max(0.0, (task["max_steps"] - total_steps) / task["max_steps"]) * 0.1

    raw_score = f1 + efficiency

    # Clamp to open interval (0.01, 0.99) — never exactly 0.0 or 1.0
    score = max(0.01, min(0.99, raw_score))
    return round(score, 3)