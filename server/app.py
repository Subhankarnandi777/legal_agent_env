"""
server/app.py — FastAPI server for the Legal Agent Environment.
OpenEnv-compatible API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

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


from fastapi.responses import HTMLResponse

LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>⚖️ Legal Agent OpenEnv | Premium Environment</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #030712;
    --card-bg: rgba(17, 24, 39, 0.7);
    --border: rgba(255, 255, 255, 0.08);
    --primary: #3b82f6;
    --secondary: #8b5cf6;
    --accent: #10b981;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --glass: rgba(255, 255, 255, 0.03);
    --glass-hover: rgba(255, 255, 255, 0.05);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg);
    color: var(--text-main);
    min-height: 100vh;
    overflow-x: hidden;
    line-height: 1.6;
  }

  h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; }

  /* Premium Animated Background */
  .bg-glow {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -1;
    background: radial-gradient(circle at 50% 50%, #0f172a 0%, #030712 100%);
    overflow: hidden;
  }

  .blob {
    position: absolute;
    width: 500px; height: 500px;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.15));
    filter: blur(80px);
    border-radius: 50%;
    animation: move 25s infinite alternate;
  }
  .blob.one { top: -100px; left: -100px; animation-delay: 0s; }
  .blob.two { bottom: -100px; right: -100px; animation-delay: -5s; width: 600px; height: 600px; background: rgba(16, 185, 129, 0.1); }
  .blob.three { top: 20%; right: 10%; animation-delay: -10s; width: 300px; height: 300px; background: rgba(239, 68, 68, 0.05); }

  @keyframes move {
    0% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(100px, 50px) scale(1.1); }
    66% { transform: translate(-50px, 120px) scale(0.9); }
    100% { transform: translate(0, 0) scale(1); }
  }

  .container { max-width: 1200px; margin: 0 auto; padding: 4rem 2rem; position: relative; }

  /* Animated Header */
  .header { text-align: center; margin-bottom: 5rem; animation: fadeInDown 1s ease-out; }
  @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }

  .badge-container { display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem; }
  .badge {
    padding: 0.5rem 1.25rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em; display: flex; align-items: center; gap: 0.5rem;
    backdrop-filter: blur(8px);
  }
  .badge.status { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: #34d399; }
  .badge.version { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); color: #60a5fa; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; box-shadow: 0 0 12px currentColor; animation: pulse 2s infinite; }
  @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }

  .header h1 {
    font-size: 4rem; font-weight: 800; margin-bottom: 1rem; line-height: 1.1;
    background: linear-gradient(to right, #fff, #94a3b8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header p { font-size: 1.25rem; color: var(--text-muted); max-width: 700px; margin: 0 auto; font-weight: 300; }

  /* Glossy Grid Cards */
  .grid-layout { display: grid; grid-template-columns: 1.5fr 1fr; gap: 2rem; margin-bottom: 2rem; }

  .hero-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 32px;
    padding: 3rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    position: relative;
    overflow: hidden;
  }
  .hero-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  }

  .stats-strip { display: flex; gap: 2rem; margin-top: 3rem; }
  .stat-item { border-left: 1px solid var(--border); padding-left: 1.5rem; }
  .stat-item .val { font-size: 2rem; font-weight: 800; color: #fff; display: block; }
  .stat-item .lab { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; }

  /* Task Section */
  .section-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 2rem; display: flex; align-items: center; gap: 0.75rem; }
  
  .task-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 4rem; }
  .task-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 2rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    cursor: default;
  }
  .task-card:hover {
    transform: translateY(-8px);
    background: var(--glass-hover);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.4);
  }
  .task-card .icon-box {
    width: 56px; height: 56px; border-radius: 16px; 
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; margin-bottom: 1.5rem;
  }
  .task-card.easy .icon-box { background: rgba(16, 185, 129, 0.1); color: #10b981; }
  .task-card.medium .icon-box { background: rgba(245, 158, 11, 0.1); color: #f59e0b; }
  .task-card.hard .icon-box { background: rgba(239, 68, 68, 0.1); color: #ef4444; }

  .task-card h3 { font-size: 1.25rem; margin-bottom: 0.75rem; color: #fff; }
  .task-card p { font-size: 0.875rem; color: var(--text-muted); margin-bottom: 1.5rem; min-height: 4rem; }
  
  .tag { padding: 0.25rem 0.75rem; border-radius: 8px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }
  .easy .tag { background: rgba(16, 185, 129, 0.1); color: #10b981; }
  .medium .tag { background: rgba(245, 158, 11, 0.1); color: #f59e0b; }
  .hard .tag { background: rgba(239, 68, 68, 0.1); color: #ef4444; }

  /* API Section with Interactive Terminal */
  .api-card {
    background: #0f172a; border: 1px solid var(--border); border-radius: 24px;
    padding: 0; overflow: hidden; height: 100%;
  }
  .api-header { background: rgba(255,255,255,0.03); padding: 1.25rem 2rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .dots { display: flex; gap: 6px; }
  .dots span { width: 10px; height: 10px; border-radius: 50%; background: var(--border); }

  .endpoint-list { padding: 1rem; }
  .endpoint {
    display: flex; align-items: center; gap: 1rem; padding: 0.85rem 1.5rem;
    border-radius: 12px; margin-bottom: 0.5rem; cursor: pointer; transition: all 0.2s;
  }
  .endpoint:hover { background: rgba(255,255,255,0.05); }
  .method { font-size: 0.65rem; font-weight: 800; padding: 0.2rem 0.5rem; border-radius: 4px; min-width: 45px; text-align: center; }
  .method.get { background: rgba(16, 185, 129, 0.15); color: #10b981; }
  .method.post { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
  .path { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #94a3b8; }

  /* Result Box */
  #result-container { padding: 2rem; background: #020617; min-height: 200px; }
  #result-box {
    font-family: 'Courier New', monospace; font-size: 0.85rem; color: #60a5fa;
    white-space: pre-wrap; margin: 0; height: 100%;
  }
  .blink { animation: blink 1s step-end infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  /* Buttons */
  .action-bar { display: flex; gap: 1rem; margin-bottom: 1.5rem; overflow-x: auto; padding-bottom: 0.5rem; }
  .btn {
    padding: 0.75rem 1.5rem; border-radius: 14px; border: 1px solid var(--border);
    background: var(--glass); color: #fff; font-size: 0.875rem; font-weight: 600;
    cursor: pointer; transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
    white-space: nowrap; display: flex; align-items: center; gap: 0.5rem;
  }
  .btn:hover { background: #fff; color: #000; transform: translateY(-2px); box-shadow: 0 10px 20px -5px rgba(255,255,255,0.1); }
  .btn.primary { background: var(--primary); border: none; }
  .btn.primary:hover { background: #60a5fa; color: #fff; box-shadow: 0 10px 20px -5px rgba(59,130,246,0.3); }

  /* Custom Scrollbar */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

  footer { text-align: center; margin-top: 6rem; padding-bottom: 3rem; color: var(--text-muted); font-size: 0.85rem; }
</style>
</head>
<body>

<div class="bg-glow">
  <div class="blob one"></div>
  <div class="blob two"></div>
  <div class="blob three"></div>
</div>

<div class="container">
  
  <header class="header">
    <div class="badge-container">
      <div class="badge status"><span class="dot"></span> Environment Live</div>
      <div class="badge version">v1.1.0 · OpenEnv 0.2.0</div>
    </div>
    <h1>Premium Legal Agent<br>Training Environment</h1>
    <p>A sophisticated reward-shaping engine designed to evaluate AI agents across diverse legal reasoning trajectories.</p>
  </header>

  <div class="grid-layout">
    <div class="hero-card">
      <div class="section-title">✨ System Overview</div>
      <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 2rem;">This Space serves as an OpenEnv compliant backend. It simulates real-world legal workflows, providing agents with structured observations and immediate reward feedback.</p>
      
      <div class="action-bar">
        <button class="btn primary" onclick="callApi('/health')">🏥 Health Check</button>
        <button class="btn" onclick="callApi('/tasks')">📋 Inspect Tasks</button>
        <a class="btn" href="/docs" target="_blank">📖 API Reference</a>
      </div>

      <div class="stats-strip">
        <div class="stat-item"><span class="val">3</span><span class="lab">Scenarios</span></div>
        <div class="stat-item"><span class="val">6</span><span class="lab">Endpoints</span></div>
        <div class="stat-item"><span class="val">F1+</span><span class="lab">Scoring</span></div>
      </div>
    </div>

    <div class="api-card">
      <div class="api-header">
        <div class="dots"><span></span><span></span><span></span></div>
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Real-time Monitor</div>
      </div>
      <div class="endpoint-list">
        <div class="endpoint" onclick="callApi('/health')">
          <span class="method get">GET</span><span class="path">/health</span>
        </div>
        <div class="endpoint" onclick="resetTask('easy')">
          <span class="method post">POST</span><span class="path">/reset (easy)</span>
        </div>
        <div class="endpoint" onclick="resetTask('medium')">
          <span class="method post">POST</span><span class="path">/reset (medium)</span>
        </div>
        <div class="endpoint" onclick="callApi('/tasks')">
          <span class="method get">GET</span><span class="path">/tasks</span>
        </div>
      </div>
      <div id="result-container">
        <pre id="result-box">> Waiting for interaction...<span class="blink">_</span></pre>
      </div>
    </div>
  </div>

  <div class="section-title" style="margin-top: 4rem;">🎯 Active Training Tracks</div>
  <div class="task-grid">
    <div class="task-card easy">
      <div class="icon-box">📄</div>
      <span class="tag">Beginner</span>
      <h3>Contract Audit</h3>
      <p>Identify critical risk factors like vague liability caps and missing termination notices in SaaS agreements.</p>
      <div style="font-size: 0.75rem; color: #64748b;">REWARD: +0.30 per correct flag</div>
    </div>
    
    <div class="task-card medium">
      <div class="icon-box">🔍</div>
      <span class="tag">Pro</span>
      <h3>Issue Spotting</h3>
      <p>Extract complex doctrines like Promissory Estoppel and Habitability from real-world litigation fact patterns.</p>
      <div style="font-size: 0.75rem; color: #64748b;">REWARD: +0.25 per identification</div>
    </div>

    <div class="task-card hard">
      <div class="icon-box">⚖️</div>
      <span class="tag">Expert</span>
      <h3>Defense Strategy</h3>
      <p>Analyze claims to build multi-layered affirmative defenses. The ultimate test of professional judgment.</p>
      <div style="font-size: 0.75rem; color: #64748b;">REWARD: Weighted Strategy Score</div>
    </div>
  </div>

  <footer>
    <p>Powered by OpenEnv · Scaler School of Technology Hackathon · 2026</p>
  </footer>
</div>

<script>
  const resultBox = document.getElementById('result-box');
  
  async function callApi(path) {
    resultBox.innerHTML = `> Executing GET ${path}...\n<span class="blink">_</span>`;
    try {
      const response = await fetch(path);
      const data = await response.json();
      resultBox.innerHTML = `> GET ${path} SUCCESS\n\n${JSON.stringify(data, null, 2)}`;
    } catch (e) {
      resultBox.innerHTML = `> ERROR\n\n${e.message}`;
    }
  }

  async function resetTask(taskId) {
    resultBox.innerHTML = `> Executing POST /reset {task_id: "${taskId}"}...\n<span class="blink">_</span>`;
    try {
      const response = await fetch('/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId })
      });
      const data = await response.json();
      resultBox.innerHTML = `> POST /reset SUCCESS\n\n${JSON.stringify(data, null, 2)}`;
    } catch (e) {
      resultBox.innerHTML = `> ERROR\n\n${e.message}`;
    }
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

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()