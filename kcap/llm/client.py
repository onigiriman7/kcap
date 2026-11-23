"""Provider-agnostic LLM client: disk cache, token/cost log, hard budget cap.

Every call names a `role` (interviewer, persona, judge, ...) which maps to a model in config,
and a `task` tag (e.g. "persona.gate") used for cost accounting and for test fakes.
"""
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from kcap.io import RUNS, append_jsonl

try:
    import truststore  # corporate TLS proxies

    truststore.inject_into_ssl()
except Exception:
    pass


class BudgetExceeded(RuntimeError):
    pass


FakeFn = Callable[[str, str, list[dict]], str]


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def parse_json(text: str):
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError(f"no JSON in response: {text[:200]}")
    s = min(starts)
    e = max(text.rfind("}"), text.rfind("]"))
    return json.loads(text[s:e + 1])


class LLM:
    def __init__(self, cfg: dict, cache_dir: Path | None = None, cost_log: Path | None = None,
                 fake: FakeFn | None = None, use_cache: bool = True):
        self.cfg = cfg
        self.fake = fake
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or RUNS / ".cache")
        self.cost_log = Path(cost_log or RUNS / "cost_log.jsonl")
        self.max_usd = float(cfg.get("budget", {}).get("max_usd", 1e9))
        self.max_calls = int(cfg.get("budget", {}).get("max_calls", 10**9))
        self.calls = 0
        self.spent_usd = 0.0
        self.tokens_by_task: dict[str, int] = {}
        self._lock = threading.Lock()

    # ---------- public API ----------
    def chat(self, role: str, task: str, prompt: str | list[dict], system: str = "",
             seed: int | None = None, json_mode: bool = False, temperature: float | None = None) -> str:
        rc = dict(self.cfg["roles"][role])
        if temperature is not None:
            rc["temperature"] = temperature
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        seed = self.cfg.get("seed", 0) if seed is None else seed
        key_obj = {"p": rc["provider"], "m": rc["model"], "t": rc.get("temperature"), "mt": rc.get("max_tokens"),
                   "s": system, "msgs": messages, "seed": seed, "json": json_mode,
                   "fake": self.fake is not None}
        key = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()
        cpath = self.cache_dir / key[:2] / f"{key}.json"
        if self.use_cache and cpath.exists():
            hit = json.loads(cpath.read_text(encoding="utf-8"))
            # cached calls cost no money but still count as logical compute (e.g. D's search tokens)
            with self._lock:
                self.tokens_by_task[task] = self.tokens_by_task.get(task, 0) + sum(hit.get("usage", [0, 0]))
            return hit["text"]

        in_tok = _est_tokens(system + "".join(m["content"] for m in messages))
        pin, pout = self._price(rc["model"])
        est = (in_tok * pin + rc.get("max_tokens", 1000) * pout) / 1e6
        with self._lock:
            if self.spent_usd + est > self.max_usd:
                raise BudgetExceeded(f"budget cap ${self.max_usd:.2f} would be exceeded (spent ${self.spent_usd:.2f})")
            if self.calls >= self.max_calls:
                raise BudgetExceeded(f"call cap {self.max_calls} reached (uncached calls this process)")
            self.calls += 1

        if self.fake is not None:
            text, usage = self.fake(task, system, messages), None
        else:
            text, usage = self._call(rc, system, messages, seed, json_mode)
        it, ot = usage or (in_tok, _est_tokens(text))
        cost = (it * pin + ot * pout) / 1e6
        with self._lock:
            self.spent_usd += cost
            self.tokens_by_task[task] = self.tokens_by_task.get(task, 0) + it + ot
        append_jsonl(self.cost_log, {"ts": time.time(), "role": role, "task": task, "model": rc["model"],
                                     "in": it, "out": ot, "usd": round(cost, 6)})
        if self.use_cache:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps({"text": text, "usage": [it, ot], "key": key_obj}, ensure_ascii=False), encoding="utf-8")
        return text

    def json(self, role: str, task: str, prompt, system: str = "", seed: int | None = None, **kw):
        sys2 = (system + "\n\nRespond with valid JSON only.").strip()
        seed = self.cfg.get("seed", 0) if seed is None else seed
        last = None
        for attempt in range(3):
            text = self.chat(role, task, prompt, system=sys2, seed=seed + 1000 * attempt, json_mode=True, **kw)
            try:
                return parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last = e
        raise ValueError(f"[{task}] could not parse JSON after 3 attempts: {last}")

    def tokens(self, prefix: str = "") -> int:
        return sum(v for k, v in self.tokens_by_task.items() if k.startswith(prefix))

    # ---------- internals ----------
    def _price(self, model: str):
        p = self.cfg.get("pricing", {}).get(model, [0.0, 0.0])
        return float(p[0]), float(p[1])

    def _call(self, rc, system, messages, seed, json_mode):
        prov = rc["provider"]
        pc = self.cfg["providers"][prov]
        if prov == "copilot":
            from kcap.llm.copilot_backend import get_backend
            return get_backend(pc).chat(rc["model"], system, messages), None
        key = os.environ.get(pc["api_key_env"])
        if not key:
            raise RuntimeError(f"environment variable {pc['api_key_env']} is not set (provider {prov})")
        if prov == "anthropic":
            url = pc.get("base_url", "https://api.anthropic.com/v1") + "/messages"
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            body = {"model": rc["model"], "max_tokens": rc.get("max_tokens", 1000),
                    "temperature": rc.get("temperature", 0.7), "messages": messages}
            if system:
                body["system"] = system
            data = self._post(url, headers, body)
            text = "".join(b.get("text", "") for b in data["content"])
            u = data.get("usage", {})
            return text, (u.get("input_tokens", 0), u.get("output_tokens", 0))
        if prov == "azure":
            ep = os.environ[pc["endpoint_env"]].rstrip("/")
            url = f"{ep}/openai/deployments/{rc['model']}/chat/completions?api-version={pc['api_version']}"
            headers = {"api-key": key}
        else:  # openai-compatible
            url = pc.get("base_url", "https://api.openai.com/v1") + "/chat/completions"
            headers = {"Authorization": f"Bearer {key}"}
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        body = {"model": rc["model"], "messages": msgs, "temperature": rc.get("temperature", 0.7),
                "max_tokens": rc.get("max_tokens", 1000), "seed": seed}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post(url, headers, body)
        u = data.get("usage", {})
        return data["choices"][0]["message"]["content"], (u.get("prompt_tokens", 0), u.get("completion_tokens", 0))

    def _post(self, url, headers, body):
        req_body = json.dumps(body).encode()
        for attempt in range(5):
            req = urllib.request.Request(url, data=req_body, method="POST",
                                         headers={"Content-Type": "application/json", **headers})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {e.read()[:500]!r}") from e
            except urllib.error.URLError:
                if attempt < 4:
                    time.sleep(2 ** attempt * 2)
                    continue
                raise
