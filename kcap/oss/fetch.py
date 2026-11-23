"""Download a repo's docs (public corpus) and merged-PR review comments (private ground-truth source)."""
import json
import re
import subprocess
from pathlib import Path

from kcap.io import RUNS, write_json, write_jsonl
from kcap.schemas import Document

OSS = RUNS / "oss"
DOC_EXT = {".md", ".rst", ".txt", ".adoc"}
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def repo_dir(repo: str) -> Path:
    return OSS / repo.replace("/", "__")


def _gh(args: list[str]) -> str:
    r = subprocess.run(["gh", "api", *args], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"gh api failed: {r.stderr[:300]}  (run `gh auth login` first)")
    return r.stdout


def _chunks(text: str, size: int = 1500) -> list[str]:
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur)
            cur = ""
        cur += p + "\n\n"
    if cur.strip():
        out.append(cur)
    return out


def _structure(root: Path, depth: int = 3) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or len(rel.parts) > depth:
            continue
        lines.append("  " * (len(rel.parts) - 1) + rel.name + ("/" if p.is_dir() else ""))
    return "\n".join(lines[:1500])


def fetch(repo: str, max_pages: int = 10, log=print) -> Path:
    if not REPO_RE.match(repo):
        raise ValueError("repo must look like OWNER/REPO")
    out = repo_dir(repo)
    src = out / "src"
    if not src.exists():
        log(f"cloning {repo}")
        subprocess.run(["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(src)], check=True)
    docs, n = [], 0
    files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXT and ".git" not in p.parts
             and (p.parent == src or "doc" in p.relative_to(src).parts[0].lower() or "design" in str(p).lower())]
    for p in sorted(files):
        rel = p.relative_to(src).as_posix()
        for i, ch in enumerate(_chunks(p.read_text(encoding="utf-8", errors="ignore"))):
            n += 1
            docs.append(Document(id=f"DOC{n:04d}", doc_type="doc", date="", text=f"{rel} (part {i + 1})\n{ch}"))
    for i, ch in enumerate(_chunks(_structure(src), 3000)):
        n += 1
        docs.append(Document(id=f"DOC{n:04d}", doc_type="code_structure", date="", text=f"Repository tree (part {i + 1})\n{ch}"))
    write_jsonl(out / "public" / "corpus.jsonl", docs)
    meta = json.loads(_gh([f"repos/{repo}"]))
    write_json(out / "public" / "domain.json", {"name": repo, "description": meta.get("description") or repo})
    log(f"{len(docs)} doc chunks")

    rows = []
    for page in range(1, max_pages + 1):
        batch = json.loads(_gh([f"repos/{repo}/pulls/comments?per_page=100&page={page}&sort=created&direction=desc"]))
        if not batch:
            break
        for c in batch:
            body = (c.get("body") or "").strip()
            if len(body) >= 80 and c.get("user", {}).get("type") != "Bot":
                rows.append({"id": f"C{c['id']}", "body": body[:2000], "path": c.get("path"),
                             "pr_url": c.get("pull_request_url"), "author": c.get("user", {}).get("login"),
                             "created_at": c.get("created_at")})
        log(f"review comments page {page}: total kept {len(rows)}")
    write_jsonl(out / "private" / "reviews.jsonl", rows)
    return out
