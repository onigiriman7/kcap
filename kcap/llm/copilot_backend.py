"""GitHub Copilot SDK backend: plain chat completions through a Copilot subscription.

One CopilotClient runs on a background event loop; each call is a fresh tool-less session.
"""
import asyncio
import threading

_lock = threading.Lock()
_backend = None


class CopilotBackend:
    def __init__(self, pc: dict):
        from copilot import CopilotClient

        from kcap.io import RUNS

        self.timeout = float(pc.get("timeout", 300))
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        # "empty" mode: no CLI agent defaults (tools, skills, memory, custom instructions)
        base = RUNS / ".copilot"
        base.mkdir(parents=True, exist_ok=True)
        self.client = CopilotClient(mode=pc.get("mode", "empty"), base_directory=str(base.resolve()),
                                    github_token=pc.get("github_token") or None)
        self._run(self.client.start())

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    async def _chat(self, model: str, system: str, prompt: str) -> str:
        kw = {"model": model, "available_tools": [], "infinite_sessions": {"enabled": False}}
        if system:
            kw["system_message"] = {"mode": "replace", "content": system}
        session = await self.client.create_session(**kw)
        try:
            ev = await session.send_and_wait(prompt, timeout=self.timeout)
            return getattr(getattr(ev, "data", None), "content", None) or ""
        finally:
            await session.disconnect()

    def chat(self, model: str, system: str, messages: list[dict]) -> str:
        return self._run(self._chat(model, system, _flatten(messages)))

    def list_models(self) -> list[str]:
        return [getattr(m, "id", str(m)) for m in self._run(self.client.list_models())]


def _flatten(messages: list[dict]) -> str:
    """Sessions are single-turn here, so earlier turns are inlined as a transcript."""
    if len(messages) == 1:
        return messages[0]["content"]
    parts = [f"[{m['role'].upper()}]\n{m['content']}" for m in messages[:-1]]
    return ("Conversation so far:\n\n" + "\n\n".join(parts) +
            f"\n\n[USER]\n{messages[-1]['content']}\n\nRespond to the last USER message.")


def get_backend(pc: dict) -> CopilotBackend:
    global _backend
    with _lock:
        if _backend is None:
            _backend = CopilotBackend(pc)
        return _backend
