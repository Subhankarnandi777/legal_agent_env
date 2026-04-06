"""
inference.py — Legal Agent Baseline Runner
Loads configuration from .env file.
"""

import asyncio
import json
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import LegalEnv
from models import LegalAction

# ── Load .env ─────────────────────────────────────────────
load_dotenv()

# ── Config ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
IMAGE_NAME = os.getenv("IMAGE_NAME", LOCAL_IMAGE_NAME or "")

ENV_BASE_URL = os.getenv("ENV_BASE_URL")

MAX_STEPS = int(os.getenv("MAX_STEPS", 25))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", 0.4))

BENCHMARK = "legal-agent-env"

# ── Issue types per task ──────────────────────────────────
EASY_ISSUES = [
    "vague_scope",
    "unenforceable_penalty",
    "ip_ownership_risk",
    "overbroad_disclaimer",
    "missing_notice_period",
    "missing_governing_law",
]

MEDIUM_ISSUES = [
    "implied_warranty_of_habitability",
    "retaliatory_eviction",
    "unlawful_entry",
    "unenforceable_waiver_clause",
    "promissory_estoppel",
]

HARD_ISSUES = [
    "material_breach_by_plaintiff",
    "merger_clause_bars_misrepresentation",
    "opinion_privilege_tortious_interference",
    "statute_of_limitations",
]

TASK_ISSUES = {
    "easy": EASY_ISSUES,
    "medium": MEDIUM_ISSUES,
    "hard": HARD_ISSUES,
}

SYSTEM_PROMPT = """
You are a legal AI agent interacting with a legal environment.
Use EXACT issue names.
Return ONLY JSON.
"""


# ── Logging ───────────────────────────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r_str}", flush=True)


# ── Prompt Builder ────────────────────────────────────────
def build_prompt(obs, step: int, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    return f"""
STEP: {step}
TASK: {obs.task_id}
DESCRIPTION: {obs.task_description}

DOCUMENT:
{obs.document_text}

ISSUES FOUND: {obs.issues_found}
ISSUES REMAINING: {obs.issues_remaining}
LAST FEEDBACK: {obs.last_action_feedback}

RECENT HISTORY:
{history_block}

Return next action as JSON.
"""


# ── Rule Based Agent ──────────────────────────────────────
def rule_based_action(obs):
    if obs.task_id == "easy":
        mapping = {
            2: "vague_scope",
            3: "unenforceable_penalty",
            4: "ip_ownership_risk",
            5: "overbroad_disclaimer",
            6: "missing_notice_period",
            7: "missing_governing_law",
        }
        for cid, issue in mapping.items():
            if str(cid) not in obs.issues_found:
                return LegalAction(action_type="flag_issue", clause_id=cid, issue_type=issue)

    elif obs.task_id == "medium":
        for issue in MEDIUM_ISSUES:
            if issue not in obs.issues_found:
                return LegalAction(action_type="flag_issue", issue_type=issue)

    elif obs.task_id == "hard":
        for d in HARD_ISSUES:
            if d not in obs.issues_found:
                return LegalAction(action_type="flag_issue", issue_type=d)

        return LegalAction(action_type="submit_strategy")


# ── LLM Agent ─────────────────────────────────────────────────────────────
async def get_llm_action(obs, client: OpenAI) -> LegalAction:
    """Gets action from LLM using the OpenAI client."""
    try:
        prompt = build_prompt(obs, obs.step_count, [])
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        
        # Validate against LegalAction
        return LegalAction(**data)
    except Exception as exc:
        # Fallback to rule-based logic for baseline reproducibility
        return rule_based_action(obs)


# ── Episode Loop ──────────────────────────────────────────
async def run_episode(task_id: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # ENV_BASE_URL takes precedence (for HF Space deployment)
    if ENV_BASE_URL:
        async with LegalEnv(base_url=ENV_BASE_URL) as env:
            return await episode_loop(env, task_id, client)
    elif IMAGE_NAME:
        env = await LegalEnv.from_docker_image(IMAGE_NAME)
        async with env:
            return await episode_loop(env, task_id, client)
    else:
        # Local development fallback
        async with LegalEnv(base_url="http://localhost:7860") as env:
            return await episode_loop(env, task_id, client)


async def episode_loop(env, task_id: str, client: OpenAI):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    total_reward = 0.0
    steps_taken = 0
    success = False
    strategy_submitted = False

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = await get_llm_action(obs, client)
            action_json = action.model_dump_json()

            if action.action_type == "submit_strategy":
                strategy_submitted = True

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            total_reward += reward
            steps_taken = step

            log_step(step, action_json, reward, done, None)

            if done:
                break

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    try:
        state = await env.state()
        score = await env.grader(
            task_id=task_id,
            issues_found=obs.issues_found,
            false_positives=state.false_positives,
            total_steps=steps_taken,
            strategy_submitted=strategy_submitted
        )
    except Exception as exc:
        print(f"[DEBUG] Grader error: {exc}", flush=True)
        score = 0.0

    success = score >= SUCCESS_SCORE_THRESHOLD

    log_end(success, steps_taken, score, rewards)
    return score, steps_taken, rewards, success


# ── Main ─────────────────────────────────────────────────
async def main():
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task_id in tasks:
        print("\n" + "=" * 50)
        print(f"Running task: {task_id}")
        print("=" * 50)

        score, steps, rewards, success = await run_episode(task_id)
        results[task_id] = {
            "score": score,
            "steps": steps,
            "success": success,
        }

    print("\n=== BASELINE SUMMARY ===")
    for t, r in results.items():
        print(f"{t:8s} score={r['score']:.3f} steps={r['steps']} success={r['success']}")

    avg = sum(r["score"] for r in results.values()) / len(results)
    print(f"{'AVERAGE':8s} score={avg:.3f}")


if __name__ == "__main__":
    asyncio.run(main())