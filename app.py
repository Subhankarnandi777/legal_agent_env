"""
server/app.py — FastAPI server for the Legal Agent Environment.
OpenEnv-compatible API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import LegalAction
from legal_environment import LegalEnvironment, grade_episode


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


from fastapi.responses import HTMLResponse

LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚖️ Legal Agent OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0e1a;
    color: #e2e8f0;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated gradient background */
  body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 20% 50%, rgba(59,130,246,0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(139,92,246,0.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 80%, rgba(16,185,129,0.06) 0%, transparent 50%);
    animation: bgShift 20s ease-in-out infinite;
    z-index: 0;
  }
  @keyframes bgShift { 0%,100%{transform:rotate(0deg)} 50%{transform:rotate(3deg)} }

  .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; position: relative; z-index: 1; }

  /* Header */
  .header { text-align: center; margin-bottom: 3rem; }
  .header .badge {
    display: inline-flex; align-items: center; gap: .5rem;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.3);
    color: #34d399; padding: .35rem 1rem; border-radius: 999px;
    font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em;
    margin-bottom: 1.25rem;
  }
  .badge .dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .header h1 { font-size: 2.8rem; font-weight: 800; letter-spacing: -.02em;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .6rem;
  }
  .header p { font-size: 1.1rem; color: #94a3b8; max-width: 650px; margin: 0 auto; line-height: 1.6; }

  /* Stats bar */
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2.5rem; }
  .stat {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.25rem; text-align: center;
    backdrop-filter: blur(10px); transition: all .3s;
  }
  .stat:hover { border-color: rgba(96,165,250,0.3); transform: translateY(-2px); }
  .stat .num { font-size: 1.8rem; font-weight: 800; color: #60a5fa; }
  .stat .label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; margin-top: .25rem; }

  /* Glass card */
  .card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 20px; padding: 2rem; margin-bottom: 1.5rem;
    backdrop-filter: blur(12px); transition: border-color .3s;
  }
  .card:hover { border-color: rgba(139,92,246,0.25); }
  .card h2 { font-size: 1.3rem; font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; gap: .6rem; }

  /* Task cards */
  .tasks-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
  .task-card {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.5rem; transition: all .3s; position: relative; overflow: hidden;
  }
  .task-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 16px 16px 0 0;
  }
  .task-card.easy::before { background: linear-gradient(90deg, #34d399, #6ee7b7); }
  .task-card.medium::before { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
  .task-card.hard::before { background: linear-gradient(90deg, #f87171, #ef4444); }
  .task-card:hover { transform: translateY(-3px); border-color: rgba(255,255,255,0.12); }

  .task-card .diff {
    display: inline-block; padding: .2rem .7rem; border-radius: 999px;
    font-size: .65rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; margin-bottom: .75rem;
  }
  .easy .diff { background: rgba(16,185,129,0.15); color: #34d399; }
  .medium .diff { background: rgba(245,158,11,0.15); color: #fbbf24; }
  .hard .diff { background: rgba(239,68,68,0.15); color: #f87171; }

  .task-card h3 { font-size: 1rem; font-weight: 700; margin-bottom: .4rem; }
  .task-card p { font-size: .8rem; color: #94a3b8; line-height: 1.5; margin-bottom: .75rem; }
  .task-card .meta { font-size: .7rem; color: #64748b; }

  /* Endpoints table */
  .ep-table { width: 100%; border-collapse: collapse; }
  .ep-table th { text-align: left; font-size: .7rem; color: #64748b; text-transform: uppercase;
    letter-spacing: .08em; padding: .6rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .ep-table td { padding: .75rem 1rem; font-size: .85rem; border-bottom: 1px solid rgba(255,255,255,0.04); }
  .ep-table tr:hover td { background: rgba(255,255,255,0.02); }
  .method {
    display: inline-block; padding: .15rem .55rem; border-radius: 6px;
    font-size: .65rem; font-weight: 700; letter-spacing: .05em;
  }
  .method.get { background: rgba(16,185,129,0.15); color: #34d399; }
  .method.post { background: rgba(59,130,246,0.15); color: #60a5fa; }
  .ep-path { font-family: 'Courier New', monospace; color: #a78bfa; font-weight: 600; }

  /* Reward table */
  .rw-table { width: 100%; border-collapse: collapse; }
  .rw-table th { text-align: left; font-size: .7rem; color: #64748b; text-transform: uppercase;
    letter-spacing: .08em; padding: .6rem .75rem; border-bottom: 1px solid rgba(255,255,255,0.06); }
  .rw-table td { padding: .6rem .75rem; font-size: .82rem; border-bottom: 1px solid rgba(255,255,255,0.04); }
  .rw-pos { color: #34d399; font-weight: 600; }
  .rw-neg { color: #f87171; font-weight: 600; }

  /* Try it section */
  .try-section { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
  .try-btn {
    padding: .65rem 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.04); color: #e2e8f0; font-size: .85rem; font-weight: 600;
    cursor: pointer; transition: all .3s; font-family: 'Inter', sans-serif;
  }
  .try-btn:hover { background: rgba(96,165,250,0.15); border-color: rgba(96,165,250,0.4); color: #60a5fa; }
  .try-btn.primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6); border: none; color: #fff; }
  .try-btn.primary:hover { opacity: .9; transform: translateY(-1px); }

  #result-box {
    margin-top: 1rem; padding: 1.25rem; border-radius: 12px;
    background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.06);
    font-family: 'Courier New', monospace; font-size: .8rem; color: #94a3b8;
    max-height: 300px; overflow-y: auto; white-space: pre-wrap; display: none;
  }

  /* Footer */
  .footer { text-align: center; margin-top: 3rem; padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.05); color: #475569; font-size: .8rem; }
  .footer a { color: #60a5fa; text-decoration: none; }

  /* Responsive */
  @media (max-width: 768px) {
    .stats { grid-template-columns: repeat(2, 1fr); }
    .tasks-grid { grid-template-columns: 1fr; }
    .header h1 { font-size: 2rem; }
  }
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <div class="badge"><span class="dot"></span> OpenEnv Compliant · Running</div>
    <h1>⚖️ Legal Agent OpenEnv</h1>
    <p>A reinforcement learning environment for training AI agents on complex legal reasoning — contract review, issue spotting, and litigation strategy.</p>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat"><div class="num">3</div><div class="label">Tasks</div></div>
    <div class="stat"><div class="num">6</div><div class="label">API Endpoints</div></div>
    <div class="stat"><div class="num">5</div><div class="label">Action Types</div></div>
    <div class="stat"><div class="num">F1</div><div class="label">Scoring Method</div></div>
  </div>

  <!-- Tasks -->
  <div class="card">
    <h2>🎯 Available Tasks</h2>
    <div class="tasks-grid">
      <div class="task-card easy">
        <span class="diff">Easy</span>
        <h3>Contract Clause Review</h3>
        <p>Identify "red flag" clauses in a SaaS agreement — vague scope, unenforceable penalties, IP risks.</p>
        <div class="meta">Max 20 steps · 6 issues · flag_issue, approve_clause, suggest_fix</div>
      </div>
      <div class="task-card medium">
        <span class="diff">Medium</span>
        <h3>Legal Issue Spotting</h3>
        <p>Find legal issues in a landlord-tenant fact pattern — habitability, retaliatory eviction, estoppel.</p>
        <div class="meta">Max 15 steps · 5 issues · flag_issue, identify_law</div>
      </div>
      <div class="task-card hard">
        <span class="diff">Hard</span>
        <h3>Case Strategy Building</h3>
        <p>Build a defense strategy identifying key affirmative defenses in a commercial litigation case.</p>
        <div class="meta">Max 25 steps · 4 defenses · flag_issue, submit_strategy</div>
      </div>
    </div>
  </div>

  <!-- Endpoints -->
  <div class="card">
    <h2>🌐 API Endpoints</h2>
    <table class="ep-table">
      <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td><span class="method get">GET</span></td><td class="ep-path">/health</td><td>Health check</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="ep-path">/reset</td><td>Start a new episode</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="ep-path">/step</td><td>Take an action in the environment</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="ep-path">/state</td><td>Get current episode state</td></tr>
        <tr><td><span class="method get">GET</span></td><td class="ep-path">/tasks</td><td>List available tasks</td></tr>
        <tr><td><span class="method post">POST</span></td><td class="ep-path">/grader</td><td>Score a completed episode</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Reward System -->
  <div class="card">
    <h2>🏆 Reward System</h2>
    <table class="rw-table">
      <thead><tr><th>Action Result</th><th>Reward</th></tr></thead>
      <tbody>
        <tr><td>Correct issue + type</td><td class="rw-pos">+0.30</td></tr>
        <tr><td>Correct issue flagged</td><td class="rw-pos">+0.20</td></tr>
        <tr><td>Correct legal issue (medium)</td><td class="rw-pos">+0.25</td></tr>
        <tr><td>Defense identified (hard)</td><td class="rw-pos">+0.25</td></tr>
        <tr><td>Incorrect issue</td><td class="rw-neg">−0.15</td></tr>
        <tr><td>Duplicate / invalid action</td><td class="rw-neg">−0.05</td></tr>
      </tbody>
    </table>
  </div>

  <!-- Try It -->
  <div class="card">
    <h2>🚀 Try It Live</h2>
    <p style="color:#94a3b8;font-size:.85rem;margin-bottom:1rem;">Test the environment directly from here.</p>
    <div class="try-section">
      <button class="try-btn primary" onclick="tryEndpoint('/health')">🏥 Health Check</button>
      <button class="try-btn" onclick="tryEndpoint('/tasks')">📋 List Tasks</button>
      <button class="try-btn" onclick="tryReset('easy')">🟢 Reset Easy</button>
      <button class="try-btn" onclick="tryReset('medium')">🟡 Reset Medium</button>
      <button class="try-btn" onclick="tryReset('hard')">🔴 Reset Hard</button>
      <button class="try-btn" onclick="tryEndpoint('/state')">📊 Get State</button>
      <a class="try-btn" href="/docs" target="_blank">📖 Swagger Docs</a>
    </div>
    <div id="result-box"></div>
  </div>

  <!-- Footer -->
  <div class="footer">
    <p>Legal Agent OpenEnv v1.0.0 · Apache 2.0 · Built for the <a href="#">OpenEnv Hackathon</a></p>
  </div>

</div>

<script>
  const box = document.getElementById('result-box');
  async function tryEndpoint(path) {
    box.style.display = 'block';
    box.textContent = '⏳ Loading...';
    try {
      const r = await fetch(path);
      const d = await r.json();
      box.textContent = JSON.stringify(d, null, 2);
    } catch(e) { box.textContent = '❌ Error: ' + e.message; }
  }
  async function tryReset(task) {
    box.style.display = 'block';
    box.textContent = '⏳ Resetting ' + task + '...';
    try {
      const r = await fetch('/reset', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task_id:task}) });
      const d = await r.json();
      box.textContent = JSON.stringify(d, null, 2);
    } catch(e) { box.textContent = '❌ Error: ' + e.message; }
  }
</script>
</body>
</html>
"""


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return LANDING_PAGE


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
        import traceback
        traceback.print_exc()
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