# ⚖️ Legal Agent OpenEnv Environment
### 🤖 Reinforcement Learning Environment for Legal Reasoning Agents

---

## 📌 Problem Statement

Legal document review, issue spotting, and litigation strategy development require significant time and expertise from legal professionals. Many legal tasks such as contract review, risk identification, and case analysis are repetitive and structured, making them suitable for automation using AI.

However, there is a lack of structured environments where AI agents can **learn legal reasoning through interaction, feedback, and rewards** rather than only static datasets.

Most AI models are trained on text but are not trained to **reason step-by-step like a lawyer**.

### 🎯 This project solves the problem by:
Building a **Reinforcement Learning Environment** where AI agents learn legal reasoning by interacting with legal documents, taking actions, receiving rewards, and improving strategies over time.

This environment can be used to train AI agents for:
- 📄 Contract review
- ⚖️ Legal issue spotting
- 🧠 Litigation strategy planning
- 📚 Legal reasoning
- 🏛️ Legal decision making
- 🔐 Compliance checking
- 🤖 Legal AI assistants

---

## 💡 Solution Overview

This project implements an **OpenEnv-compatible legal reasoning environment** where an AI agent interacts with legal tasks using structured actions and receives rewards based on legal correctness.

The environment simulates real legal workflows such as:
1. Reviewing contracts
2. Identifying legal issues
3. Suggesting clause fixes
4. Building litigation strategies

The environment is designed similar to reinforcement learning environments like **OpenAI Gym**, but for legal reasoning.

---

## 🏗️ Environment Architecture
```
    Agent (inference.py)
            ↓
    client.py (API wrapper)
            ↓
    FastAPI server (server/app.py)
            ↓
    Legal Environment Logic (legal_environment.py)
            ↓
    Reward + State + Done
```

The agent interacts with the environment using:
- `/reset`
- `/step`
- `/state`
- `/grader`

This follows the standard reinforcement learning interaction loop.

---

## 📚 Tasks in the Environment

The environment contains **three tasks with increasing difficulty**.

### 🟢 Easy — Contract Clause Review
The agent reviews a software services agreement and must:
- Identify problematic clauses
- Approve legally sound clauses
- Suggest fixes for problematic clauses

**Goal:** Identify all legal risks in the contract.

---

### 🟡 Medium — Legal Issue Spotting
The agent analyzes a legal fact pattern and must:
- Identify legal issues
- Specify area of law
- Provide reasoning

**Goal:** Identify all legal doctrines present in the case.

---

### 🔴 Hard — Case Strategy Building
The agent acts as defense counsel and must:
- Identify weaknesses in plaintiff claims
- Identify legal defenses
- Submit a final defense strategy

**Goal:** Build a complete legal defense strategy.

---

## 📥 Observation Space

The agent receives structured observations containing:
- Task ID
- Task description
- Legal document text
- Clause status
- Issues found
- Issues remaining
- Feedback from last action
- Step count
- Hint

---

## 🎮 Action Space

| Action | Description |
|-------|-------------|
| flag_issue | Identify a legal issue |
| approve_clause | Approve a legally sound clause |
| suggest_fix | Suggest a corrected clause |
| identify_law | Identify legal doctrine |
| submit_strategy | Submit final legal strategy |

---

## 🏆 Reward System

The environment uses **reward shaping** to guide the agent toward correct legal reasoning.

| Action Result | Reward |
|---------------|--------|
| Correct issue identification | +0.20 |
| Correct issue type | +0.10 |
| Suggest valid fix | +0.15 |
| Approve valid clause | +0.05 |
| Incorrect issue | -0.15 |
| Duplicate action | -0.05 |
| Invalid action | -0.05 |
| Strategy submission | Reward based on defense coverage |

Final score is calculated using:
- Precision
- Recall
- F1 Score
- Step efficiency bonus

Score range: **0.0 – 1.0**

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Take an action |
| /state | GET | Current episode state |
| /tasks | GET | List available tasks |
| /grader | POST | Score completed episode |

---

## 📁 Project Structure
```
    legal-agent-env/
    ├── Dockerfile
    ├── openenv.yaml
    ├── requirements.txt
    ├── README.md
    ├── inference.py
    ├── client.py
    ├── models.py
    └── server/
    ├── app.py
    └── legal_environment.py
```


---

## ▶️ Running Locally

Start the FastAPI server:

> uvicorn server.app:app --port 7860


Test health endpoint:

http://localhost:7860/docs#/default/health_health_get


Run baseline agent:

> python inference.py


---

## 🐳 Running with Docker

Build Docker image:
> docker build -t legal-env .


Run container:
> docker run -p 7860:7860 legal-env


---

## 🧠 Reinforcement Learning Usage

This environment can be used to train reinforcement learning agents using:
- Q-learning
- PPO
- DQN
- RLlib
- Stable-Baselines3
- LLM agents

Standard RL loop:
```
state = reset()
while not done:
action = agent(state)
state, reward, done = step(action)
```

---

## 🚀 Use Cases

This environment can be used for:
- Legal AI research
- Contract risk detection
- Legal document automation
- Litigation strategy generation
- Legal reasoning benchmarks
- Compliance checking
- Legal assistants
- AI lawyer training environments

---

## ✅ Evaluation Criteria

The environment satisfies the following evaluation criteria:
- Runtime correctness
- OpenEnv interface compliance
- Clear and realistic task design
- Meaningful reward system
- Deterministic grading
- Docker reproducibility
- Agent compatibility

---

## 📜 License
MIT License