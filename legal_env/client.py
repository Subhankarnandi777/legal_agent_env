"""
client.py — Python client for the Legal Agent Environment.

Usage:
    # Sync (simple)
    with LegalEnv(base_url="http://localhost:7860").sync() as env:
        result = env.reset(task_id="easy")
        result = env.step(LegalAction(action_type="flag_issue", clause_id=3, issue_type="vague_scope"))

    # Async
    async with LegalEnv(base_url="http://localhost:7860") as env:
        result = await env.reset(task_id="easy")
        result = await env.step(LegalAction(...))
"""

from __future__ import annotations
import asyncio
import httpx
from typing import Optional

try:
    from .models import LegalAction, LegalObservation, LegalState, StepResult
except ImportError:
    from models import LegalAction, LegalObservation, LegalState, StepResult


class _SyncLegalEnv:
    """Synchronous wrapper — use inside a `with` statement."""

    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._client.close()

    def reset(self, task_id: str = "easy") -> StepResult:
        resp = self._client.post(f"{self._base}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return StepResult(**resp.json())

    def step(self, action: LegalAction) -> StepResult:
        resp = self._client.post(f"{self._base}/step", json=action.model_dump())
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> LegalState:
        resp = self._client.get(f"{self._base}/state")
        resp.raise_for_status()
        return LegalState(**resp.json())

    def close(self):
        self._client.close()


class LegalEnv:
    """
    Async client for the Legal Agent Environment.
    Mirrors the OpenEnv HTTPEnvClient interface.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self._base = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ── async context manager ─────────────────────────────────────────────────
    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30)
        return self

    async def __aexit__(self, *_):
        await self.close()

    # ── factory: connect to a Docker image (mirrors OpenEnv pattern) ──────────
    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "LegalEnv":
        """
        Start a local Docker container and connect to it.
        Used in inference.py when IMAGE_NAME env var is set.
        """
        import subprocess, time
        container_name = "legal-env-container"
        subprocess.Popen(
            ["docker", "run", "--rm", "--name", container_name,
             "-p", f"{port}:7860", image_name],
        )
        # Wait for container to be ready
        instance = cls(base_url=f"http://localhost:{port}")
        for _ in range(30):
            try:
                async with httpx.AsyncClient() as c:
                    r = await c.get(f"http://localhost:{port}/health", timeout=2)
                    if r.status_code == 200:
                        return instance
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("Container did not start in time")

    # ── API methods ───────────────────────────────────────────────────────────
    async def reset(self, task_id: str = "easy") -> StepResult:
        r = await self._client.post(f"{self._base}/reset", json={"task_id": task_id})
        r.raise_for_status()
        return StepResult(**r.json())

    async def step(self, action: LegalAction) -> StepResult:
        r = await self._client.post(f"{self._base}/step", json=action.model_dump())
        r.raise_for_status()
        return StepResult(**r.json())

    async def state(self) -> LegalState:
        r = await self._client.get(f"{self._base}/state")
        r.raise_for_status()
        return LegalState(**r.json())

    async def close(self):
        if self._client:
            await self._client.aclose()

    def sync(self) -> _SyncLegalEnv:
        """Return a synchronous wrapper for simple scripts."""
        return _SyncLegalEnv(self._base)