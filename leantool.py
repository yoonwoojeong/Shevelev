"""
leantool — on-demand Leanstral, with a fast compiler gate
═════════════════════════════════════════════════════════

The batch modes in `leanstral.py` (--simplify / --bridge / --inspect) are slow
for two structural reasons, both fixed here:

  1. They re-ran a full `lake build` (≈75 s: recompiles Shevelev.lean) after
     *every* candidate. Here a candidate is checked in a throwaway file that
     `import`s the already-compiled library, so only the snippet is elaborated.

  2. They sent each theorem to Leanstral with `context=""`, so the model had
     never seen `F`, `H`, `K`, `mark` … and duly hallucinated or returned
     `sorry`. Here every request carries an auto-extracted API summary
     (signatures only, no proof bodies) of the development.

Use it three ways:

    # 1. CLI, one shot
    python leantool.py ask   "theorem foo (n : ℕ) : H n 0 0 = 1 := by sorry"
    python leantool.py check snippet.lean
    python leantool.py solve "theorem foo (n : ℕ) : H n 0 0 = 1 := by sorry" -n 3

    # 2. MCP server, so an agent can call it as a tool
    python leantool.py mcp          # (registered via .mcp.json)

    # 3. As a library
    from leantool import lean_check, solve
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time

# Console on this machine is cp949; printing `ℤ` otherwise kills the process.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leanstral import query_leanstral, get_api_key  # noqa: E402

# An MCP server is spawned by the editor, not from your shell, so it may not
# inherit MISTRAL_API_KEY. Fall back to a git-ignored key file.
if not os.environ.get("MISTRAL_API_KEY"):
    _kf = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mistral_key")
    if os.path.exists(_kf):
        with open(_kf, "r", encoding="utf-8") as _f:
            os.environ["MISTRAL_API_KEY"] = _f.read().strip()

# ── Configuration ─────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRATCH_DIR = os.path.join(ROOT, ".leanscratch")
DEFAULT_IMPORT = "Proof.Shevelev"
# Everything lives in `namespace Shevelev`, which itself opens `Finset`. Without
# these the scratch file cannot even see `H`, `F` or `mark`.
DEFAULT_OPENS = ["Shevelev", "Finset"]
DEFAULT_SOURCE = os.path.join(ROOT, "Proof", "Shevelev.lean")
CHECK_TIMEOUT = 300


# ── The fast compiler gate ────────────────────────────────────────

_DIAG = re.compile(r"^(?P<file>.*?):(?P<line>\d+):(?P<col>\d+):\s*"
                   r"(?P<sev>error|warning|info):\s*(?P<msg>.*)$")


def lean_check(snippet: str, imports: list[str] | None = None,
               timeout: int = CHECK_TIMEOUT) -> dict:
    """Elaborate `snippet` against the compiled library and report diagnostics.

    The snippet is written to a file under `.leanscratch/` (outside the `Proof`
    lib glob, so `lake build` never picks it up) with the `import` lines
    prepended, then run through `lake env lean`. Because the library's `.olean`
    is already on disk, this costs seconds rather than a full rebuild.

    Returns {ok, errors, warnings, elapsed, output, path}. `ok` means the file
    elaborated with no error *and* no `sorry`.
    """
    imports = imports if imports is not None else [DEFAULT_IMPORT]
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    header = "".join(f"import {m}\n" for m in imports) + f"open {' '.join(DEFAULT_OPENS)}\n"
    path = os.path.join(SCRATCH_DIR, f"Check_{os.getpid()}_{int(time.time()*1000)}.lean")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + snippet.rstrip() + "\n")

    started = time.time()
    try:
        r = subprocess.run(["lake", "env", "lean", os.path.relpath(path, ROOT)],
                           cwd=ROOT, capture_output=True, text=False, timeout=timeout)
        raw = (r.stdout or b"") + (r.stderr or b"")
        out = raw.decode("utf-8", errors="replace")
        rc = r.returncode
    except subprocess.TimeoutExpired:
        out, rc = f"TIMEOUT after {timeout}s", 1
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    elapsed = time.time() - started

    # `lean` reports line numbers of the scratch file; shift them back so they
    # refer to the snippet the caller actually handed us. The header is the
    # import lines, the `open` line, then one blank line.
    offset = len(imports) + 2
    errors, warnings = [], []
    for line in out.splitlines():
        m = _DIAG.match(line.strip())
        if not m:
            if errors and m is None and line.startswith(" "):
                errors[-1]["msg"] += "\n" + line
            continue
        d = {"line": int(m["line"]) - offset, "col": int(m["col"]), "msg": m["msg"]}
        (errors if m["sev"] == "error" else
         warnings if m["sev"] == "warning" else []).append(d)

    uses_sorry = any("sorry" in w["msg"] for w in warnings) or "sorry" in snippet
    return {
        "ok": rc == 0 and not errors and not uses_sorry,
        "errors": errors,
        "warnings": warnings,
        "uses_sorry": uses_sorry,
        "elapsed": round(elapsed, 1),
        "output": out.strip(),
    }


# ── The *fast* gate: one persistent Lean server ───────────────────
#
# `lean_check` above is correct but pays ~5 min per call, because Shevelev.lean
# does `import Mathlib` and a fresh `lean` process reloads every olean. The
# language server instead keeps one worker alive per file: the imports are
# loaded once, and each subsequent edit re-elaborates only the snippet.
#
# So the first check costs minutes and every later one costs seconds. That is
# only worth it in a long-lived process — which is exactly what the MCP server
# is, so tool calls after the first are fast.

import queue        # noqa: E402
import threading    # noqa: E402


class LeanServer:
    """A `lake serve` process with one scratch file held open."""

    FIRST_TIMEOUT = 2400  # cold start: lake may rebuild imports, then load Mathlib
    EDIT_TIMEOUT = 180    # warm: re-elaborating the snippet only

    def __init__(self, imports: list[str] | None = None):
        self.imports = imports if imports is not None else [DEFAULT_IMPORT]
        self.header = ("".join(f"import {m}\n" for m in self.imports)
                       + f"open {' '.join(DEFAULT_OPENS)}\n\n")
        self.path = os.path.join(SCRATCH_DIR, "Live.lean")
        self.uri = "file:///" + self.path.replace("\\", "/").lstrip("/")
        self.version = 1
        self._id = 0
        self._q: queue.Queue = queue.Queue()
        self.proc = None
        self.ready = False

    # -- transport ------------------------------------------------
    def _send(self, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()

    def _request(self, method: str, params: dict) -> int:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
        return self._id

    def _notify(self, method: str, params: dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _reader(self) -> None:
        """Parse Content-Length framed messages onto the queue until EOF."""
        f = self.proc.stdout
        while True:
            length = None
            while True:
                line = f.readline()
                if not line:
                    self._q.put(None)
                    return
                line = line.strip()
                if not line:
                    break
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
            if length is None:
                continue
            try:
                self._q.put(json.loads(f.read(length).decode("utf-8", errors="replace")))
            except Exception:
                pass

    # -- lifecycle ------------------------------------------------
    def start(self) -> None:
        os.makedirs(SCRATCH_DIR, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(self.header)
        self.proc = subprocess.Popen(
            ["lake", "serve"], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._reader, daemon=True).start()

        self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": "file:///" + ROOT.replace("\\", "/").lstrip("/"),
            "capabilities": {"textDocument": {"publishDiagnostics": {}}},
        })
        self._await_id(self._id, timeout=120)
        self._notify("initialized", {})
        self._notify("textDocument/didOpen", {"textDocument": {
            "uri": self.uri, "languageId": "lean4",
            "version": self.version, "text": self.header}})
        self._drain_until_idle(self.FIRST_TIMEOUT, self.version)
        self.ready = True

    def _await_id(self, want: int, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                msg = self._q.get(timeout=max(0.1, end - time.time()))
            except queue.Empty:
                break
            if msg is None:
                raise RuntimeError("lean server exited")
            if msg.get("id") == want and "method" not in msg:
                return msg
        raise TimeoutError(f"no response to request {want}")

    QUIET = 3.0   # seconds of silence that mean "the server is done talking"

    def _drain_stale(self) -> None:
        """Discard anything queued from a previous version before we edit."""
        while True:
            try:
                if self._q.get_nowait() is None:
                    raise RuntimeError("lean server exited")
            except queue.Empty:
                return

    def _drain_until_idle(self, timeout: float, version: int) -> list[dict]:
        """Collect the diagnostics *for `version`* and return once elaboration ends.

        Matching the version is not optional. Without it the first diagnostics
        to arrive — typically the previous snippet's, still queued — are taken
        as this snippet's, and a check reports whatever the last one did. That
        made a provably false `by decide` goal come back clean.
        """
        end = time.time() + timeout
        diags: list[dict] = []

        while time.time() < end:
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg is None:
                raise RuntimeError("lean server exited")
            params = msg.get("params") or {}
            method = msg.get("method")

            if method == "textDocument/publishDiagnostics" and params.get("uri") == self.uri:
                v = params.get("version")
                if v is not None and v != version:
                    continue          # stale: belongs to an earlier snippet
                diags = params.get("diagnostics", [])
            elif method == "$/lean/fileProgress":
                v = (params.get("textDocument") or {}).get("version")
                if v is not None and v != version:
                    continue
                if params.get("processing", []):
                    continue
                # Elaboration finished. Diagnostics are published *during*
                # processing too — an empty set before this point means "nothing
                # wrong yet", not "nothing wrong" — so only now are they final.
                grace = time.time() + 2.0
                while time.time() < grace:
                    try:
                        m2 = self._q.get(timeout=max(0.05, grace - time.time()))
                    except queue.Empty:
                        break
                    if m2 and m2.get("method") == "textDocument/publishDiagnostics" \
                            and (m2.get("params") or {}).get("uri") == self.uri:
                        v2 = m2["params"].get("version")
                        if v2 is None or v2 == version:
                            diags = m2["params"].get("diagnostics", [])
                return diags

        raise TimeoutError(
            f"elaboration of version {version} did not finish within {timeout}s")

    def check(self, snippet: str) -> dict:
        """Replace the scratch file's body with `snippet` and report diagnostics."""
        if self.proc is None:
            started = time.time()
            self.start()
            cold = round(time.time() - started, 1)
        else:
            cold = 0.0

        self._drain_stale()
        self.version += 1
        text = self.header + snippet.rstrip() + "\n"
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(text)
        started = time.time()
        self._notify("textDocument/didChange", {
            "textDocument": {"uri": self.uri, "version": self.version},
            "contentChanges": [{"text": text}]})
        raw = self._drain_until_idle(self.EDIT_TIMEOUT, self.version)
        elapsed = round(time.time() - started, 1)

        offset = len(self.imports) + 2   # imports + `open` line + blank line
        errors, warnings = [], []
        for d in raw:
            entry = {"line": d["range"]["start"]["line"] - offset + 1,
                     "col": d["range"]["start"]["character"],
                     "msg": d.get("message", "").strip()}
            sev = d.get("severity", 1)
            if sev == 1:
                errors.append(entry)
            elif sev == 2:
                warnings.append(entry)
        uses_sorry = "sorry" in snippet or any("sorry" in w["msg"] for w in warnings)
        return {"ok": not errors and not uses_sorry, "errors": errors,
                "warnings": warnings, "uses_sorry": uses_sorry,
                "elapsed": elapsed, "cold_start": cold}

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
            except OSError:
                pass
            self.proc = None


# A checker that can silently pass a false goal is worse than no checker: it
# launders garbage into "VERIFIED". These two probes run once per server and
# `solve` refuses to report a verdict until the harness has told them apart.
#
# They are deliberately trivial. An earlier version used a real counterexample
# to the mis-transcribed `paper_theorem_2_H`, but `decide` on a `Finset.sum`
# over `ℤ` runs for minutes — so a self-test failure could not be distinguished
# from a slow one. The self-test checks the *harness*, not the mathematics.
_PROBE_TRUE = "example : (1 : ℤ) = 1 := by decide"
_PROBE_FALSE = "example : (1 : ℤ) = 0 := by decide"


def verify_harness(srv: "LeanServer") -> None:
    """Raise unless the server accepts a true goal and rejects a false one."""
    if getattr(srv, "_trusted", False):
        return
    if not srv.check(_PROBE_TRUE)["ok"]:
        raise RuntimeError("harness self-test failed: a TRUE goal was rejected")
    if srv.check(_PROBE_FALSE)["ok"]:
        raise RuntimeError(
            "harness self-test failed: a FALSE goal was accepted — diagnostics "
            "are not being matched to the submitted snippet; results cannot be trusted")
    srv._trusted = True


_SERVER: LeanServer | None = None


def lean_check_fast(snippet: str) -> dict:
    """Check via the shared persistent server, falling back to a one-shot run."""
    global _SERVER
    if _SERVER is None:
        _SERVER = LeanServer()
    try:
        verify_harness(_SERVER)
        return _SERVER.check(snippet)
    except Exception as e:
        _SERVER = None
        r = lean_check(snippet)
        r["note"] = f"persistent server unavailable ({e}); used one-shot lake env lean"
        return r


# ── Context: an API summary of the development ────────────────────

_DECL_KW = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?"
    r"(noncomputable\s+)?(theorem|lemma|def|abbrev|instance|structure|inductive)\b")


def api_summary(source: str = DEFAULT_SOURCE, max_chars: int = 12000) -> str:
    """Signatures of everything in `source`, with proof bodies stripped.

    This is what the batch modes were missing: given only a bare theorem,
    Leanstral cannot know that `H`, `K` and `mark` exist, so it invents lemma
    names. Feeding it the available API is the cheapest quality win there is.

    `def`/`abbrev` bodies are kept (they *are* the meaning); theorem proofs are
    replaced by `:= ..` since only the statement matters for reuse.
    """
    try:
        with open(source, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return ""

    starts = [i for i, l in enumerate(lines) if _DECL_KW.match(l)]
    out: list[str] = []
    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        block = "\n".join(lines[s:end]).rstrip()
        kind = _DECL_KW.match(lines[s]).group(2)

        if kind in ("def", "abbrev", "structure", "inductive", "instance"):
            out.append(block)
            continue
        # Theorem/lemma: keep the statement, drop the proof.
        depth = 0
        cut = None
        for i, c in enumerate(block):
            if c in "([{⟨":
                depth += 1
            elif c in ")]}⟩":
                depth -= 1
            elif depth == 0 and c == ":" and i + 1 < len(block) and block[i + 1] == "=":
                cut = i
                break
        out.append((block[:cut].rstrip() + " := ..") if cut else block)

    text = "\n\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n-- [truncated]"
    return text


# ── Asking Leanstral ──────────────────────────────────────────────

_FENCE_OPEN = re.compile(r"^```(?:lean4?)?\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def _unfence(s: str) -> str:
    return _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", s.strip())).strip()


def ask(goal: str, context: str | None = None, extra: str = "",
        api_key: str = "") -> str:
    """One Leanstral call. `context=None` auto-loads the API summary."""
    ctx = api_summary() if context is None else context
    prompt = goal if not extra else f"{extra}\n\n{goal}"
    return _unfence(query_leanstral(prompt, context=ctx, api_key=api_key or get_api_key()))


def solve(goal: str, attempts: int = 3, context: str | None = None,
          imports: list[str] | None = None, api_key: str = "",
          verbose: bool = True) -> dict:
    """Ask → compile → feed the errors back → ask again, up to `attempts`.

    This is the loop the batch modes never had: a rejected candidate was simply
    discarded, so Leanstral never learned *why* it failed. Returns the first
    candidate that compiles clean, else the last attempt plus its diagnostics.
    """
    api_key = api_key or get_api_key()
    ctx = api_summary() if context is None else context
    history: list[dict] = []
    candidate = ""

    # Leanstral will otherwise "think out loud" in `--` comments and run into the
    # token limit mid-expression, producing a truncated block that cannot parse.
    RULES = ("Return ONLY the finished Lean 4 declaration: no markdown fences, no "
             "imports, no prose, and no explanatory `--` comments. Keep the statement "
             "exactly as given and replace `sorry` with a real proof.")

    for i in range(1, attempts + 1):
        if i == 1:
            prompt = f"{RULES}\n\nComplete this declaration:\n```lean\n{goal}\n```"
        else:
            last = history[-1]
            diag = "\n".join(f"  line {e['line']}: {e['msg']}" for e in last["errors"][:6]) \
                or "  (no errors reported, but the proof still uses sorry)"
            prompt = (f"{RULES}\n\nYour previous attempt did NOT compile.\n\n"
                      f"Previous attempt:\n```lean\n{last['candidate']}\n```\n\n"
                      f"Compiler errors:\n{diag}\n\n"
                      f"Original goal:\n```lean\n{goal}\n```")

        if verbose:
            print(f"  [attempt {i}/{attempts}] querying Leanstral…", flush=True)
        candidate = _unfence(query_leanstral(prompt, context=ctx, api_key=api_key))

        if verbose:
            print(f"  [attempt {i}/{attempts}] compiling…", flush=True)
        res = lean_check_fast(candidate)
        history.append({"candidate": candidate, **res})

        if verbose:
            status = "✓ compiles" if res["ok"] else \
                     f"✗ {len(res['errors'])} error(s)" + (" +sorry" if res["uses_sorry"] else "")
            print(f"  [attempt {i}/{attempts}] {status} ({res['elapsed']}s)", flush=True)

        if res["ok"]:
            return {"ok": True, "proof": candidate, "attempts": i, "history": history}

    return {"ok": False, "proof": candidate, "attempts": attempts, "history": history}


# ── MCP server (hand-rolled stdio JSON-RPC — no extra dependencies) ─

TOOLS = [
    {
        "name": "lean_check",
        "description": (
            "Compile a Lean 4 snippet against the already-built Proof library "
            "(imports Proof.Shevelev by default) and return errors/warnings. "
            "Seconds, not a full rebuild. Use to verify any Lean code before editing a file."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "snippet": {"type": "string", "description": "Lean 4 code, without import lines."},
                "imports": {"type": "array", "items": {"type": "string"},
                            "description": "Modules to import (default: [\"Proof.Shevelev\"])."},
            },
            "required": ["snippet"],
        },
    },
    {
        "name": "leanstral_ask",
        "description": (
            "Ask Leanstral 1.5 for a proof. Fast, unverified — the reply is a "
            "candidate only. Automatically includes an API summary of Shevelev.lean "
            "as context. Use leanstral_solve if you want it compiler-checked."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Lean declaration, may contain sorry."},
                "instructions": {"type": "string", "description": "Extra guidance (optional)."},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "leanstral_solve",
        "description": (
            "Ask Leanstral, compile the answer, feed compiler errors back, retry. "
            "Returns a verified proof or the failed attempts with diagnostics. "
            "This is the one to reach for on a sorry-ed goal."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Lean declaration to prove."},
                "attempts": {"type": "integer", "description": "Max attempts (default 3)."},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "lean_api_summary",
        "description": ("Signatures of every declaration in Proof/Shevelev.lean with proof "
                        "bodies stripped — a compact map of the available API."),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _dispatch(name: str, args: dict) -> str:
    if name == "lean_check":
        r = lean_check_fast(args["snippet"])
        head = "OK — compiles clean" if r["ok"] else "FAILED"
        parts = [f"{head}  ({r['elapsed']}s"
                 + (f", {r['cold_start']}s cold start" if r.get("cold_start") else "") + ")"]
        if r["uses_sorry"]:
            parts.append("contains sorry")
        for e in r["errors"]:
            parts.append(f"error (snippet line {e['line']}): {e['msg']}")
        for w in r["warnings"]:
            parts.append(f"warning (snippet line {w['line']}): {w['msg']}")
        return "\n".join(parts)

    if name == "leanstral_ask":
        return ask(args["goal"], extra=args.get("instructions", ""))

    if name == "leanstral_solve":
        r = solve(args["goal"], attempts=int(args.get("attempts", 3)), verbose=False)
        lines = [("VERIFIED — compiles clean" if r["ok"] else "UNVERIFIED — best attempt below"),
                 f"attempts: {r['attempts']}", "", r["proof"]]
        if not r["ok"]:
            last = r["history"][-1]
            lines += ["", "remaining diagnostics:"]
            lines += [f"  line {e['line']}: {e['msg']}" for e in last["errors"][:8]]
            if last["uses_sorry"]:
                lines.append("  still contains sorry")
        return "\n".join(lines)

    if name == "lean_api_summary":
        return api_summary()

    raise ValueError(f"unknown tool: {name}")


def serve_mcp() -> None:
    """Minimal MCP stdio server: newline-delimited JSON-RPC on stdin/stdout."""
    inp = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
    outp = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

    def reply(msg_id, result=None, error=None):
        m = {"jsonrpc": "2.0", "id": msg_id}
        m["error" if error else "result"] = error or result
        outp.write(json.dumps(m) + "\n")
        outp.flush()

    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method, msg_id = msg.get("method"), msg.get("id")
        if msg_id is None:           # notification — never answered
            continue

        try:
            if method == "initialize":
                ver = msg.get("params", {}).get("protocolVersion", "2024-11-05")
                reply(msg_id, {"protocolVersion": ver,
                               "capabilities": {"tools": {}},
                               "serverInfo": {"name": "leanstral", "version": "1.0.0"}})
            elif method == "tools/list":
                reply(msg_id, {"tools": TOOLS})
            elif method == "tools/call":
                p = msg.get("params", {})
                text = _dispatch(p.get("name", ""), p.get("arguments", {}) or {})
                reply(msg_id, {"content": [{"type": "text", "text": text}]})
            elif method == "ping":
                reply(msg_id, {})
            else:
                reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})
        except Exception as e:  # never let one bad call kill the server
            reply(msg_id, {"content": [{"type": "text", "text": f"ERROR: {e}"}],
                           "isError": True})


# ── CLI ───────────────────────────────────────────────────────────

USAGE = """\
leantool — on-demand Leanstral with a fast compiler gate

  python leantool.py ask   "<lean decl>"        candidate proof, unverified (~10s)
  python leantool.py solve "<lean decl>" [-n 3] ask → compile → retry until it builds
  python leantool.py check <file.lean | ->      compile a snippet against the library
  python leantool.py api                        print the API summary sent as context
  python leantool.py mcp                        run as an MCP server (used by .mcp.json)
"""


def main() -> None:
    if len(sys.argv) < 2:
        print(USAGE)
        return
    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "mcp":
        serve_mcp()
    elif cmd == "api":
        print(api_summary())
    elif cmd == "ask":
        print(ask(" ".join(rest)))
    elif cmd == "solve":
        n = 3
        if "-n" in rest:
            i = rest.index("-n")
            n = int(rest[i + 1])
            rest = rest[:i] + rest[i + 2:]
        r = solve(" ".join(rest), attempts=n)
        print()
        print(r["proof"])
        print()
        print("VERIFIED" if r["ok"] else "UNVERIFIED — see diagnostics above")
        sys.exit(0 if r["ok"] else 1)
    elif cmd == "bench":
        # Does keeping the server warm actually pay off? Measure, don't assume.
        probes = [
            "example (n : ℕ) : H n 0 0 = 1 := by simp [H, F, coef]",
            "example (n : ℕ) (hn : 0 < n) (i : ℤ) (m s : ℕ) :\n"
            "    H n i (m + s) = ∑ j ∈ Finset.range n, H n (j : ℤ) s * H n (i - (j : ℤ)) m :=\n"
            "  theorem2_H n hn i m s",
            "example (q : ℤ) : mark 1 q = 1 := mark_one_eq q",
        ]
        srv = LeanServer()
        for k, p in enumerate(probes, 1):
            r = srv.check(p)
            tag = "cold" if r.get("cold_start") else "warm"
            print(f"  probe {k} [{tag}]: {r['elapsed']}s elaborate"
                  + (f" (+{r['cold_start']}s server start)" if r.get("cold_start") else "")
                  + f" -> {'ok' if r['ok'] else str(len(r['errors'])) + ' error(s)'}", flush=True)
            for e in r["errors"][:3]:
                print(f"      line {e['line']}: {e['msg'][:120]}")
        srv.stop()
    elif cmd == "check":
        src = sys.stdin.read() if (not rest or rest[0] == "-") else \
            open(rest[0], "r", encoding="utf-8").read()
        # A pasted file may already carry its own imports; keep them out of the body.
        body = "\n".join(l for l in src.splitlines() if not l.startswith("import "))
        r = lean_check(body)
        print(f"{'OK' if r['ok'] else 'FAILED'}  ({r['elapsed']}s)")
        for e in r["errors"]:
            print(f"  error  line {e['line']}: {e['msg']}")
        for w in r["warnings"]:
            print(f"  warn   line {w['line']}: {w['msg']}")
        sys.exit(0 if r["ok"] else 1)
    else:
        print(USAGE)
        sys.exit(2)


if __name__ == "__main__":
    main()
