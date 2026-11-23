"""Local browser GUI for owner steps: blinded Exp 3 interview sessions and scenario sealing. Binds to 127.0.0.1 only."""
import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kcap.human import scenarios_cli
from kcap.io import SEALED, read_json, write_json
from kcap.schemas import Scenario

DRAFT = SEALED / "scenarios_draft.json"
MAX_BODY = 2_000_000


def _status() -> dict:
    sealed = scenarios_cli.SEALED_FILE.exists()
    return {"sealed": sealed,
            "sha256": scenarios_cli.sha256_file(scenarios_cli.SEALED_FILE) if sealed else None,
            "count": len(json.loads(scenarios_cli.SEALED_FILE.read_text(encoding="utf-8"))) if sealed else 0,
            # never send sealed content back to the browser
            "draft": json.loads(DRAFT.read_text(encoding="utf-8")) if DRAFT.exists() and not sealed else []}


def _clean(rows) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("scenarios must be a list")
    out = []
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("bad row")
        out.append({k: str(r.get(k, ""))[:20000] for k in ("topic", "prompt", "reference_answer")})
    return out


class _Stop(Exception):
    pass


class Live:
    """One running interview session, bridged between a worker thread and the browser."""

    def __init__(self):
        self.cv = threading.Condition()
        self.llm = self.cfg = None
        self.reset()

    def reset(self):
        self.state, self.label, self.arm, self.question, self.answer = "idle", None, None, "", None
        self.q_shown, self.spent, self.limit, self.n, self.stop, self.error = 0.0, 0.0, 0.0, 0, False, ""

    def public(self) -> dict:
        with self.cv:
            used = self.spent + (time.monotonic() - self.q_shown if self.state == "question" else 0)
            # never expose the arm (B/C/D): the owner stays blind to the strategy
            return {"state": self.state, "label": self.label, "question": self.question if self.state == "question" else "",
                    "time_left": max(0, self.limit - used), "answered": self.n, "error": self.error}

    def answer_fn(self, q: str) -> str:
        with self.cv:
            if self.stop:
                raise _Stop
            self.state, self.question, self.answer, self.q_shown = "question", q, None, time.monotonic()
            self.cv.notify_all()
            while self.answer is None and not self.stop:
                self.cv.wait(1)
            if self.answer is None:
                raise _Stop
            self.spent += time.monotonic() - self.q_shown
            self.n += 1
            self.state = "thinking"
            return self.answer

    def run(self, arm: str):
        from kcap.components.interviewer.base import run_interview
        from kcap.runners import exp3
        sess = exp3.EXP3 / "sessions"
        sess.mkdir(parents=True, exist_ok=True)
        (sess / f"{arm}.jsonl").unlink(missing_ok=True)
        log: list = []
        try:
            iv, n = exp3.build_arm(self.llm, self.cfg, arm, log)
            run_interview(iv, self.answer_fn, n, sess / f"{arm}.jsonl", time_limit_s=self.limit)
        except _Stop:
            pass
        except Exception as e:
            traceback.print_exc()
            with self.cv:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
            return
        write_json(sess / f"search_log_{arm}.json", log)
        with self.cv:
            self.state = "rating"


LIVE = Live()


def _sessions() -> list[dict]:
    from kcap.runners import exp3
    order = read_json(exp3.EXP3 / "split.json")["session_order"]
    sess = exp3.EXP3 / "sessions"
    return [{"label": f"Session {i + 1}",
             "status": "done" if (sess / f"burden_{a}.json").exists() else
             "partial" if (sess / f"{a}.jsonl").exists() else "new"} for i, a in enumerate(order)]


def _arm_for(label: str) -> str:
    from kcap.runners import exp3
    order = read_json(exp3.EXP3 / "split.json")["session_order"]
    i = int(label.rsplit(" ", 1)[-1]) - 1
    if not 0 <= i < len(order):
        raise ValueError("bad session")
    return order[i]


def _order() -> list[str]:
    from kcap.runners import exp3
    return read_json(exp3.EXP3 / "split.json")["session_order"]


def _notes() -> dict:
    """Notes per interview. Keys are 'interview_number:note_id' so the browser never sees the strategy letter."""
    from kcap.io import read_jsonl
    from kcap.runners import exp3
    tags_p = exp3.EXP3 / "tags.json"
    tags = read_json(tags_p) if tags_p.exists() else {}
    out = []
    for i, arm in enumerate(_order(), 1):
        p = exp3.EXP3 / f"kus_{arm}.json"
        tp = exp3.EXP3 / "sessions" / f"{arm}.jsonl"
        turns = {t["idx"]: t for t in read_jsonl(tp)} if tp.exists() else {}
        notes = []
        for k in (read_json(p) if p.exists() else []):
            notes.append({"key": f"{i}:{k['id']}", "id": k["id"], "condition": k.get("condition"), "action": k.get("action"),
                          "reason": k.get("reason"), "scope": k.get("scope"), "confidence": k.get("confidence"),
                          "examples": k.get("examples") or [],
                          "sources": [{"q": turns[n]["question"], "a": turns[n]["answer"]} for n in k.get("source_turns", [])
                                      if n in turns],
                          "tag": tags.get(f"{arm}:{k['id']}")})
        out.append({"label": f"Interview {i}", "notes": notes})
    return {"interviews": out}


def _save_tag(body: dict) -> dict:
    from kcap.runners import exp3
    i, kid = str(body["key"]).split(":", 1)
    arm = _order()[int(i) - 1]
    tags_p = exp3.EXP3 / "tags.json"
    tags = read_json(tags_p) if tags_p.exists() else {}
    t = tags.get(f"{arm}:{kid}", {})
    for k in ("true", "documented", "unprompted"):
        if k in body:
            if body[k] not in (True, False):
                raise ValueError(f"{k} must be true/false")
            t[k] = body[k]
    if "importance" in body:
        if body["importance"] not in (1, 2, 3, 4, 5):
            raise ValueError("importance must be 1-5")
        t["importance"] = body["importance"]
    # partial answers are kept; surprise is only defined once all three are answered
    if all(k in t for k in ("true", "documented", "unprompted")):
        t["surprise"] = t["true"] and not t["documented"] and not t["unprompted"]
    tags[f"{arm}:{kid}"] = t
    write_json(tags_p, tags)
    return {"ok": True, "tag": t}


def _grade_order(n: int) -> list[int]:
    import random
    order = list(range(n))
    random.Random(0).shuffle(order)
    return order


def _grade_items() -> dict:
    """Shuffled apprentice answers with condition hidden; addressed by position only."""
    from kcap.config import load_prereg
    from kcap.human.scenarios_cli import load_sealed
    from kcap.runners import exp3
    p = exp3.EXP3 / "apprentice.json"
    if not p.exists():
        return {"ready": False, "items": []}
    answers = read_json(p)
    scen = {s.id: s for s in load_sealed((load_prereg() or {}).get("exp3", {}).get("sealed_scenarios_sha256"))}
    gp = exp3.EXP3 / "owner_grades.json"
    grades = read_json(gp) if gp.exists() else {}
    items = []
    for pos, i in enumerate(_grade_order(len(answers))):
        a = answers[i]
        items.append({"n": pos, "situation": scen[a["scenario_id"]].prompt, "answer": a["answer"],
                      "grade": grades.get(f"{a['scenario_id']}|{a['condition']}")})
    return {"ready": True, "items": items}


def _save_grade(body: dict) -> dict:
    from kcap.runners import exp3
    answers = read_json(exp3.EXP3 / "apprentice.json")
    pos, score = int(body["n"]), body.get("score")
    if score not in (0, 1, 2):
        raise ValueError("score must be 0, 1 or 2")
    a = answers[_grade_order(len(answers))[pos]]
    gp = exp3.EXP3 / "owner_grades.json"
    grades = read_json(gp) if gp.exists() else {}
    grades[f"{a['scenario_id']}|{a['condition']}"] = score
    write_json(gp, grades)
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self._send(200, INTERVIEW_PAGE.encode("utf-8"), "text/html")
        elif self.path == "/seal":
            self._send(200, PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/status":
            self._json(200, _status())
        elif self.path == "/api/sessions":
            self._json(200, {"sessions": _sessions(), "minutes": LIVE.cfg["exp3"]["session_minutes"] if LIVE.cfg else None})
        elif self.path == "/api/live":
            self._json(200, LIVE.public())
        elif self.path == "/notes":
            self._send(200, NOTES_PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/notes":
            self._json(200, _notes())
        elif self.path == "/grade":
            self._send(200, GRADE_PAGE.encode("utf-8"), "text/html")
        elif self.path == "/api/grade":
            self._json(200, _grade_items())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        # same-origin only: blocks other websites from posting to this local server
        if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            return self._json(403, {"error": "bad origin"})
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._json(413, {"error": "too large"})
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": str(e)})
        if self.path.startswith("/api/session/"):
            return self._session(self.path.rsplit("/", 1)[-1], body)
        if self.path == "/api/notes/tag":
            try:
                return self._json(200, _save_tag(body))
            except (ValueError, KeyError) as e:
                return self._json(400, {"error": str(e)})
        if self.path == "/api/grade":
            try:
                return self._json(200, _save_grade(body))
            except (ValueError, KeyError, IndexError) as e:
                return self._json(400, {"error": str(e)})
        try:
            rows = _clean(body.get("scenarios"))
        except ValueError as e:
            return self._json(400, {"error": str(e)})
        if scenarios_cli.SEALED_FILE.exists():
            return self._json(409, {"error": "already sealed"})
        if self.path == "/api/draft":
            DRAFT.parent.mkdir(parents=True, exist_ok=True)
            DRAFT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            return self._json(200, {"ok": True})
        if self.path == "/api/seal":
            done = [r for r in rows if r["topic"].strip() and r["prompt"].strip() and r["reference_answer"].strip()]
            if len(done) != len(rows) or not done:
                return self._json(400, {"error": "every scenario needs topic, situation and answer"})
            scen = [Scenario(id=f"H{i + 1:02d}", topic=r["topic"].strip(), prompt=r["prompt"].strip(),
                             reference_answer=r["reference_answer"].strip()) for i, r in enumerate(done)]
            h = scenarios_cli.seal(scen)
            DRAFT.unlink(missing_ok=True)
            return self._json(200, {"ok": True, "sha256": h, "count": len(scen)})
        self._json(404, {"error": "not found"})

    def _session(self, action: str, body: dict):
        if LIVE.llm is None:
            return self._json(503, {"error": "interviews need the LLM; start with `kcap exp3 gui`"})
        with LIVE.cv:
            if action == "start":
                if LIVE.state in ("preparing", "question", "thinking", "rating"):
                    return self._json(409, {"error": "a session is already running"})
                try:
                    arm = _arm_for(str(body.get("label", "")))
                except (ValueError, KeyError) as e:
                    return self._json(400, {"error": str(e)})
                LIVE.reset()
                LIVE.state, LIVE.label, LIVE.arm = "preparing", body["label"], arm
                LIVE.limit = float(LIVE.cfg["exp3"]["session_minutes"]) * 60
                threading.Thread(target=LIVE.run, args=(arm,), daemon=True).start()
                return self._json(200, {"ok": True})
            if action == "answer":
                if LIVE.state != "question":
                    return self._json(409, {"error": "no open question"})
                LIVE.answer = str(body.get("text", ""))[:20000].strip() or "(no answer)"
                LIVE.cv.notify_all()
                return self._json(200, {"ok": True})
            if action == "stop":
                LIVE.stop = True
                LIVE.cv.notify_all()
                return self._json(200, {"ok": True})
            if action == "rate":
                if LIVE.state != "rating":
                    return self._json(409, {"error": "nothing to rate"})
                r = body.get("rating")
                if r not in (1, 2, 3, 4, 5):
                    return self._json(400, {"error": "rating must be 1-5"})
                from kcap.runners import exp3
                write_json(exp3.EXP3 / "sessions" / f"burden_{LIVE.arm}.json",
                           {"rating": r, "note": str(body.get("note", ""))[:5000]})
                LIVE.reset()
                return self._json(200, {"ok": True})
        self._json(404, {"error": "not found"})


def serve(port: int = 8765, open_browser: bool = True, llm=None, cfg=None):
    LIVE.llm, LIVE.cfg = llm, cfg
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"kcap GUI at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>kcap - Exp 3 scenarios</title>
<style>
:root{--bg:#0f1420;--card:#182033;--ink:#e7ecf5;--mute:#9aa6bd;--d:#5ac8f2;--ok:#7bd88f;--bad:#e86a92}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
.wrap{display:grid;grid-template-columns:260px 1fr;min-height:100vh}
aside{background:#121a2b;padding:16px;overflow:auto;max-height:100vh;position:sticky;top:0}
aside h2{font-size:15px;margin:0 0 10px}.item{padding:7px 9px;border-radius:6px;cursor:pointer;margin:2px 0;font-size:13px;display:flex;gap:6px}
.item:hover{background:#1d2740}.item.on{background:#24314f}.dot{width:9px;height:9px;border-radius:50%;margin-top:4px;flex:none;background:#3b4868}.dot.full{background:var(--ok)}.dot.part{background:#f2a65a}
main{padding:24px 32px;max-width:900px}h1{margin:0 0 4px;font-size:22px}.sub{color:var(--mute);font-size:14px}
label{display:block;margin:16px 0 6px;font-weight:600}.hint{font-weight:400;color:var(--mute);font-size:13px}
input,textarea{width:100%;background:#0b1120;color:var(--ink);border:1px solid #2e3a58;border-radius:6px;padding:10px;font:inherit}
textarea{min-height:140px;resize:vertical}input:focus,textarea:focus{outline:2px solid var(--d)}
.bar{display:flex;gap:8px;margin-top:18px;flex-wrap:wrap;align-items:center}
button{background:#2a3654;color:var(--ink);border:0;padding:9px 16px;border-radius:6px;cursor:pointer;font:inherit}
button:hover{background:#36446a}button.p{background:var(--d);color:#04121c;font-weight:600}button.danger{background:var(--bad);color:#200010;font-weight:600}
button:disabled{opacity:.4;cursor:default}.box{background:var(--card);border-radius:10px;padding:16px 20px;margin:16px 0}
.saved{color:var(--mute);font-size:13px}.prog{height:8px;background:#222c44;border-radius:4px;overflow:hidden;margin:10px 0}.prog div{height:100%;background:var(--ok)}
.ok{color:var(--ok)}.err{color:var(--bad)}code{background:#0b1120;padding:2px 6px;border-radius:4px;color:#ffd58a}
</style></head><body>
<div class="wrap">
<aside><h2>Scenarios</h2><div id="list"></div><div class="bar"><button id="add">+ Add</button></div></aside>
<main>
  <h1>Experiment 3 - Sealed test scenarios</h1>
  <div class="sub">Write realistic situations a successor in <b>your area</b> would face, and the answer <b>you</b> would give.
  Prefer ones where the documents alone would lead to a wrong or incomplete answer. Drafts auto-save; nothing is final until you press <b>Seal</b>.</div>
  <div class="prog"><div id="pg"></div></div><div class="sub" id="pgt"></div>
  <div id="editor" class="box">
    <h2 id="title" style="margin:0"></h2>
    <label>Topic label <span class="hint">a few words, e.g. "voice wake-word at speed"</span></label>
    <input id="topic" maxlength="200">
    <label>Situation <span class="hint">what the successor faces - concrete, with the relevant context</span></label>
    <textarea id="prompt"></textarea>
    <label>Your reference answer <span class="hint">what to do AND why; include conditions / exceptions you'd expect a good answer to mention</span></label>
    <textarea id="ref"></textarea>
    <div class="bar"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button>
      <button id="del" style="margin-left:auto">Delete this scenario</button><span class="saved" id="saved"></span></div>
  </div>
  <div class="box">
    <h2 style="margin-top:0">Seal</h2>
    <p class="sub">Sealing writes the scenarios to <code>sealed/scenarios.json</code>, records its SHA-256 in <code>config/prereg.yaml</code>, and locks them. The tool's components cannot read this folder. You cannot edit them afterwards.</p>
    <div class="bar"><button class="danger" id="seal">Seal scenarios</button><span id="msg"></span></div>
  </div>
  <div id="done" class="box" style="display:none"></div>
</main></div>
<script>
const $=s=>document.querySelector(s);let S=[],cur=0,t=null;
const blank=()=>({topic:"",prompt:"",reference_answer:""});
const state=r=>[r.topic,r.prompt,r.reference_answer].filter(x=>x.trim()).length;
function list(){ $('#list').innerHTML=S.map((r,i)=>{const n=state(r);return `<div class="item ${i===cur?'on':''}" data-i="${i}"><span class="dot ${n===3?'full':n?'part':''}"></span><span>${i+1}. ${esc(r.topic)||'<i style="color:#9aa6bd">untitled</i>'}</span></div>`}).join('');
  document.querySelectorAll('.item').forEach(e=>e.onclick=()=>{cur=+e.dataset.i;show()});
  const full=S.filter(r=>state(r)===3).length;$('#pg').style.width=Math.min(100,full/20*100)+'%';$('#pgt').textContent=`${full} complete of ${S.length} (target 20)`;}
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function show(){const r=S[cur];$('#title').textContent=`Scenario ${cur+1} of ${S.length}`;$('#topic').value=r.topic;$('#prompt').value=r.prompt;$('#ref').value=r.reference_answer;
  $('#prev').disabled=cur===0;list();}
function edit(){const r=S[cur];r.topic=$('#topic').value;r.prompt=$('#prompt').value;r.reference_answer=$('#ref').value;list();
  $('#saved').textContent='saving...';clearTimeout(t);t=setTimeout(save,600);}
async function api(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});return [r.ok,await r.json()];}
async function save(){const [ok,j]=await api('/api/draft',{scenarios:S});$('#saved').textContent=ok?'draft saved '+new Date().toLocaleTimeString():('save failed: '+j.error);}
['#topic','#prompt','#ref'].forEach(i=>$(i).oninput=edit);
$('#prev').onclick=()=>{cur=Math.max(0,cur-1);show()};
$('#next').onclick=()=>{if(cur===S.length-1)S.push(blank());cur++;show();save()};
$('#add').onclick=()=>{S.push(blank());cur=S.length-1;show();save()};
$('#del').onclick=()=>{if(!confirm('Delete this scenario?'))return;S.splice(cur,1);if(!S.length)S.push(blank());cur=Math.min(cur,S.length-1);show();save()};
$('#seal').onclick=async()=>{const rows=S.filter(r=>state(r)>0);const bad=rows.findIndex(r=>state(r)<3);
  if(!rows.length)return $('#msg').innerHTML='<span class="err">Nothing to seal.</span>';
  if(bad>=0){cur=S.indexOf(rows[bad]);show();return $('#msg').innerHTML='<span class="err">This scenario is incomplete.</span>';}
  if(!confirm(`Seal ${rows.length} scenarios? This cannot be undone.`))return;
  const [ok,j]=await api('/api/seal',{scenarios:rows});
  if(!ok)return $('#msg').innerHTML=`<span class="err">${esc(j.error)}</span>`;locked(j.count,j.sha256);};
function locked(n,h){document.querySelectorAll('#editor,#seal,#add').forEach(e=>e.style.display='none');$('#list').innerHTML='';
  $('#done').style.display='block';$('#done').innerHTML=`<h2 class="ok" style="margin-top:0">Sealed ${n} scenarios</h2><p>SHA-256: <code>${h}</code></p>
  <p>Next: commit the hash (<code>git add config/prereg.yaml &amp;&amp; git commit</code>), then tell Copilot to run <code>kcap exp3 split</code>. You can close this tab.</p>`;}
(async()=>{const j=await (await fetch('/api/status')).json();if(j.sealed)return locked(j.count,j.sha256);
  S=j.draft&&j.draft.length?j.draft:Array.from({length:20},blank);show();})();
</script></body></html>"""

INTERVIEW_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>kcap - Exp 3 interviews</title>
<style>
:root{--bg:#0f1420;--card:#182033;--ink:#e7ecf5;--mute:#9aa6bd;--d:#5ac8f2;--ok:#7bd88f;--bad:#e86a92;--warn:#f2a65a}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:860px;margin:auto;padding:28px 24px 60px}h1{margin:0 0 6px;font-size:24px}.sub{color:var(--mute);font-size:14px;line-height:1.55}
.box{background:var(--card);border-radius:12px;padding:18px 22px;margin:16px 0}
.sess{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #26314c}.sess:last-child{border:0}
.sess b{min-width:100px}.tag{font-size:12px;padding:2px 9px;border-radius:10px;background:#2a3654}.tag.done{background:var(--ok);color:#002008}.tag.partial{background:var(--warn);color:#201000}
button{background:#2a3654;color:var(--ink);border:0;padding:9px 16px;border-radius:6px;cursor:pointer;font:inherit}
button:hover{background:#36446a}button.p{background:var(--d);color:#04121c;font-weight:600}button.danger{background:var(--bad);color:#200010}
button:disabled{opacity:.4;cursor:default}
.q{font-size:19px;line-height:1.5;background:#121a2b;border-left:4px solid var(--d);padding:14px 18px;border-radius:6px;white-space:pre-wrap}
textarea{width:100%;min-height:170px;margin-top:12px;background:#0b1120;color:var(--ink);border:1px solid #2e3a58;border-radius:6px;padding:12px;font:inherit;font-size:15px;resize:vertical}
textarea:focus{outline:2px solid var(--d)}.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
.timer{font-size:28px;font-weight:700;font-variant-numeric:tabular-nums}.timer.low{color:var(--warn)}
.top{display:flex;justify-content:space-between;align-items:center}.spin{display:inline-block;width:16px;height:16px;border:3px solid #2e3a58;border-top-color:var(--d);border-radius:50%;animation:s 1s linear infinite;vertical-align:-3px;margin-right:8px}
@keyframes s{to{transform:rotate(360deg)}}.rate button{min-width:46px}.rate button.on{background:var(--d);color:#04121c}
.err{color:var(--bad)}.ok{color:var(--ok)}
</style></head><body><main>
<h1>Experiment 3 - Interview sessions</h1>
<div class="sub">Three sessions, each run by a <b>different interviewer strategy</b>. You are not told which is which, and the order is random.
Each session covers a different random third of the topics. Only <b>your answering time</b> counts toward the limit; the interviewer's thinking time between questions does not.
"I don't know" is a perfectly good answer. Take a break between sessions if you like.</div>

<div id="list" class="box"></div>

<div id="run" class="box" style="display:none">
  <div class="top"><h2 id="slabel" style="margin:0"></h2><div><span class="sub">your time left&nbsp;</span><span class="timer" id="timer">--:--</span></div></div>
  <p class="sub" id="count"></p>
  <div id="wait"><p><span class="spin"></span><span id="waitmsg"></span></p></div>
  <div id="ask" style="display:none">
    <div class="q" id="q"></div>
    <textarea id="a" placeholder="Type your answer. English or Japanese. Ctrl+Enter to send."></textarea>
    <div class="bar"><button class="p" id="send">Send answer (Ctrl+Enter)</button><button id="idk">I don't know</button>
      <button class="danger" id="stop" style="margin-left:auto">End session</button></div>
  </div>
</div>

<div id="rating" class="box" style="display:none">
  <h2 style="margin-top:0">Session finished</h2>
  <p>How burdensome was this session? 1 = effortless ... 5 = exhausting</p>
  <div class="bar rate" id="rate"></div>
  <textarea id="note" style="min-height:80px" placeholder="Optional note (what was good or annoying about this interviewer?)"></textarea>
  <div class="bar"><button class="p" id="saverate" disabled>Save and finish</button></div>
</div>
<p id="msg"></p>
</main>
<script>
const $=s=>document.querySelector(s);let live={},rating=0,poll=null,lastQ="";
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function post(p,b){const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});const j=await r.json();if(!r.ok)$('#msg').innerHTML=`<span class="err">${esc(j.error)}</span>`;return r.ok;}
async function sessions(){const j=await (await fetch('/api/sessions')).json();const busy=['preparing','question','thinking','rating'].includes(live.state);
  $('#list').innerHTML=`<h2 style="margin-top:0">Sessions (${j.minutes} min of your time each)</h2>`+j.sessions.map(s=>`<div class="sess"><b>${s.label}</b>
  <span class="tag ${s.status}">${{new:'not started',partial:'interrupted - restart to redo',done:'done'}[s.status]}</span>
  <span style="margin-left:auto"></span>${s.status==='done'?'':`<button class="p" ${busy?'disabled':''} data-l="${s.label}">${s.status==='partial'?'Restart':'Start'}</button>`}</div>`).join('');
  document.querySelectorAll('#list button[data-l]').forEach(b=>b.onclick=async()=>{
    if(b.textContent==='Restart'&&!confirm('Restarting discards the interrupted answers of this session. Continue?'))return;
    $('#msg').textContent='';if(await post('/api/session/start',{label:b.dataset.l}))tick();});}
function fmt(s){s=Math.ceil(s);return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`}
async function tick(){live=await (await fetch('/api/live')).json();const st=live.state;
  $('#run').style.display=['preparing','question','thinking'].includes(st)?'block':'none';
  $('#rating').style.display=st==='rating'?'block':'none';
  if(st==='error')$('#msg').innerHTML=`<span class="err">Session error: ${esc(live.error)}. You can restart it.</span>`;
  $('#slabel').textContent=live.label||'';$('#count').textContent=`${live.answered} answered`;
  $('#timer').textContent=fmt(live.time_left);$('#timer').classList.toggle('low',live.time_left<180);
  const asking=st==='question';$('#ask').style.display=asking?'block':'none';$('#wait').style.display=asking?'none':'block';
  $('#waitmsg').textContent=st==='preparing'?'Preparing the interview (reading the documents). This can take a minute or two - not counted against your time.':'The interviewer is choosing the next question - not counted against your time.';
  if(asking&&live.question!==lastQ){lastQ=live.question;$('#q').textContent=live.question;$('#a').value='';$('#a').focus();}
  clearTimeout(poll);if(['preparing','question','thinking'].includes(st))poll=setTimeout(tick,1000);
  await sessions();}
async function send(t){$('#send').disabled=$('#idk').disabled=true;if(await post('/api/session/answer',{text:t})){lastQ='';}
  $('#send').disabled=$('#idk').disabled=false;tick();}
$('#send').onclick=()=>{const t=$('#a').value.trim();if(!t)return $('#a').focus();send(t)};
$('#idk').onclick=()=>send("I don't know.");
$('#a').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();$('#send').click();}});
$('#stop').onclick=async()=>{if(!confirm('End this session now? Answers so far are kept.'))return;await post('/api/session/stop');tick();};
$('#rate').innerHTML=[1,2,3,4,5].map(n=>`<button data-r="${n}">${n}</button>`).join('');
document.querySelectorAll('#rate button').forEach(b=>b.onclick=()=>{rating=+b.dataset.r;document.querySelectorAll('#rate button').forEach(x=>x.classList.toggle('on',x===b));$('#saverate').disabled=false;});
$('#saverate').onclick=async()=>{if(await post('/api/session/rate',{rating,note:$('#note').value})){rating=0;$('#note').value='';$('#saverate').disabled=true;
  document.querySelectorAll('#rate button').forEach(x=>x.classList.remove('on'));$('#msg').innerHTML='<span class="ok">Saved. Take a break, then start the next session.</span>';}tick();};
tick();
</script></body></html>"""

_STYLE = r"""<style>
:root{--bg:#0f1420;--card:#182033;--ink:#e7ecf5;--mute:#9aa6bd;--d:#5ac8f2;--ok:#7bd88f;--bad:#e86a92;--warn:#f2a65a}
*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,system-ui,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:980px;margin:auto;padding:28px 24px 60px}h1{margin:0 0 6px;font-size:24px}.sub{color:var(--mute);font-size:14px;line-height:1.55}
.box{background:var(--card);border-radius:12px;padding:16px 20px;margin:14px 0}
.tabs{display:flex;gap:8px;margin:16px 0}.tabs button.on{background:var(--d);color:#04121c;font-weight:600}
button{background:#2a3654;color:var(--ink);border:0;padding:7px 14px;border-radius:6px;cursor:pointer;font:inherit}
button:hover{background:#36446a}.yn button.on{background:var(--d);color:#04121c;font-weight:600}
.f{display:grid;grid-template-columns:150px 1fr;gap:4px 12px;margin:6px 0 10px}.f b{color:var(--mute);font-weight:600}
.q{display:flex;gap:10px;align-items:center;margin:5px 0;flex-wrap:wrap}.q span{min-width:330px}
details{margin-top:8px}summary{cursor:pointer;color:var(--mute)}.src{background:#121a2b;border-radius:6px;padding:8px 12px;margin:6px 0;white-space:pre-wrap;font-size:14px}
.pill{font-size:12px;padding:2px 9px;border-radius:10px;background:#2a3654}.pill.s{background:var(--ok);color:#002008}.pill.c{background:var(--bad);color:#200010}
.prog{height:8px;background:#222c44;border-radius:4px;overflow:hidden;margin:8px 0}.prog div{height:100%;background:var(--ok)}
.big{font-size:17px;line-height:1.55;white-space:pre-wrap}.grade button{min-width:150px}.grade button.on{background:var(--d);color:#04121c;font-weight:600}
</style>"""

NOTES_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>kcap - Exp 3 notes</title>""" + _STYLE + r"""</head><body><main>
<h1>Experiment 3 - Principles distilled from your interviews</h1>
<div class="sub">Each card is a reusable principle the AI abstracted from your answers, with the past decisions it is based on. Open "from your answers" to see the exact Q&amp;A.
For each principle answer three questions. One that is <b>true</b>, <b>not documented</b>, and that you <b>would not have written down unprompted</b> counts as a <b>surprise</b>. Answers save immediately.</div>
<div class="prog"><div id="pg"></div></div><div class="sub" id="pgt"></div>
<div class="tabs" id="tabs"></div><div id="list"></div>
</main><script>
const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let D={interviews:[]},cur=0;const Q=[['true','Is this true?'],['documented','Is it already documented somewhere?'],['unprompted','Would you have written this down unprompted?']];
const done=t=>t&&Q.every(([q])=>typeof t[q]==='boolean')&&t.importance;
async function load(){D=await (await fetch('/api/notes')).json();draw();}
function draw(){const all=D.interviews.flatMap(i=>i.notes),nd=all.filter(n=>done(n.tag)).length;
  $('#pg').style.width=(all.length?nd/all.length*100:0)+'%';$('#pgt').textContent=`${nd} of ${all.length} principles fully rated`;
  $('#tabs').innerHTML=D.interviews.map((iv,k)=>`<button class="${k===cur?'on':''}" data-k="${k}">${iv.label} (${iv.notes.filter(n=>done(n.tag)).length}/${iv.notes.length})</button>`).join('');
  document.querySelectorAll('#tabs button').forEach(b=>b.onclick=()=>{cur=+b.dataset.k;draw();});
  const iv=D.interviews[cur];if(!iv){$('#list').innerHTML='<p class="sub">No notes yet.</p>';return;}
  $('#list').innerHTML=iv.notes.length?iv.notes.map((n,j)=>{const t=n.tag||{};return `<div class="box" id="n${j}">
   <div style="display:flex;justify-content:space-between"><b>${esc(n.id)}</b><span>${n.tag&&n.tag.surprise?'<span class="pill s">surprise</span>':done(n.tag)?'<span class="pill">rated</span>':''}
   ${n.confidence==='conflicts_with_records'?' <span class="pill c">conflicts with documents</span>':''}</span></div>
   <div class="f"><b>Situation type</b><div>${esc(n.condition)}</div><b>Principle</b><div>${esc(n.action)}</div>
   ${n.reason?`<b>Why</b><div>${esc(n.reason)}</div>`:''}${n.scope?`<b>Exceptions / signals</b><div>${esc(n.scope)}</div>`:''}
   ${n.examples.length?`<b>Past examples</b><div>${n.examples.map(esc).join('<br>')}</div>`:''}</div>
   ${Q.map(([k,l])=>`<div class="q yn"><span>${l}</span><button data-j="${j}" data-k="${k}" data-v="1" class="${t[k]===true?'on':''}">Yes</button><button data-j="${j}" data-k="${k}" data-v="0" class="${t[k]===false?'on':''}">No</button></div>`).join('')}
   <div class="q yn"><span>How important is this principle for a successor? (1 = minor, 5 = critical)</span>${[1,2,3,4,5].map(v=>`<button data-j="${j}" data-imp="${v}" class="${t.importance===v?'on':''}">${v}</button>`).join('')}</div>
   <details><summary>from your answers (${n.sources.length})</summary>${n.sources.map(s=>`<div class="src"><b>Q:</b> ${esc(s.q)}\n<b>A:</b> ${esc(s.a)}</div>`).join('')}</details></div>`}).join(''):'<p class="sub">No notes were extracted from this interview.</p>';
  document.querySelectorAll('.yn button').forEach(b=>b.onclick=()=>b.dataset.imp?pick(+b.dataset.j,'importance',+b.dataset.imp):pick(+b.dataset.j,b.dataset.k,b.dataset.v==='1'));}
async function pick(j,k,v){const n=D.interviews[cur].notes[j];
  const r=await fetch('/api/notes/tag',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:n.key,[k]:v})});
  if(r.ok){n.tag=(await r.json()).tag;draw();}}
load();
</script></body></html>"""

GRADE_PAGE = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>kcap - Exp 3 blind grading</title>""" + _STYLE + r"""</head><body><main>
<h1>Experiment 3 - Blind grading</h1>
<div class="sub">Each card is a hidden test situation and an AI answer to it. You are not told what information the AI had, and the order is shuffled.
Grade by your own judgment as the expert. Grades save immediately; you can stop and come back.</div>
<div class="prog"><div id="pg"></div></div><div class="sub" id="pgt"></div>
<div id="card" class="box"></div>
<div style="display:flex;gap:8px"><button id="prev">&larr; Prev</button><button id="next">Next &rarr;</button><button id="nextun" style="margin-left:auto">Next ungraded</button></div>
</main><script>
const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let D={items:[]},i=0;const G=[[0,'0 - wrong or harmful'],[1,'1 - partly right (misses a key condition)'],[2,'2 - what I would actually do']];
async function load(){D=await (await fetch('/api/grade')).json();if(!D.ready){$('#card').innerHTML='<p>The AI answers are not ready yet.</p>';return;}draw();}
function draw(){const it=D.items[i],done=D.items.filter(x=>x.grade!=null).length;$('#pg').style.width=done/D.items.length*100+'%';
  $('#pgt').textContent=`${done} of ${D.items.length} graded - card ${i+1}`;
  $('#card').innerHTML=`<h3 style="margin-top:0">Situation</h3><div class="big">${esc(it.situation)}</div><h3>AI answer</h3><div class="big">${esc(it.answer)}</div>
   <div class="grade" style="display:flex;gap:8px;flex-wrap:wrap;margin-top:16px">${G.map(([v,l])=>`<button data-v="${v}" class="${it.grade===v?'on':''}">${l}</button>`).join('')}</div>`;
  document.querySelectorAll('.grade button').forEach(b=>b.onclick=async()=>{const r=await fetch('/api/grade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n:it.n,score:+b.dataset.v})});
   if(r.ok){it.grade=+b.dataset.v;const nx=D.items.findIndex((x,k)=>k>i&&x.grade==null);if(nx>=0)i=nx;draw();}});}
$('#prev').onclick=()=>{i=Math.max(0,i-1);draw()};$('#next').onclick=()=>{i=Math.min(D.items.length-1,i+1);draw()};
$('#nextun').onclick=()=>{const k=D.items.findIndex(x=>x.grade==null);if(k>=0){i=k;draw();}};
load();
</script></body></html>"""
