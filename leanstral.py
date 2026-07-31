"""
Leanstral 1.5 — Lean 4 Proof Assistant Script
═══════════════════════════════════════════════

A simple CLI tool that sends Lean 4 proof goals to Mistral's
Leanstral 1.5 model and returns completed proofs.

Usage:
    python leanstral.py                          # interactive mode
    python leanstral.py --goal "theorem ..."     # single goal mode
    python leanstral.py --file MyTheorem.lean    # fill sorry's in a file

Requires:
    pip install httpx
    Set MISTRAL_API_KEY environment variable
"""

import os
import sys
import json
import argparse
import re
import subprocess

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install it with:  pip install httpx")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────

API_URL = "https://api.mistral.ai/v1/chat/completions"
DEFAULT_MODEL = "labs-leanstral-1-5"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are an expert formal verification engineer specializing in Lean 4.
Your goal is to construct valid, machine-checkable Lean 4 proofs.

When provided with a theorem statement or a goal state:
1. Analyze the context, imports, and available hypotheses.
2. Produce a complete proof using idiomatic Lean 4 tactics.
3. Ensure the proof avoids 'sorry' (incomplete proofs).
4. Use Mathlib lemmas and tactics where applicable.
5. Return ONLY the Lean 4 code — no markdown fences, no explanations.
"""

# Will be set by main() from CLI args
_model = DEFAULT_MODEL


# ── API Interaction ───────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("MISTRAL_API_KEY", "")
    if not key:
        print("ERROR: MISTRAL_API_KEY environment variable is not set.")
        print("Get your key at: https://console.mistral.ai/api-keys")
        print()
        print("Set it with:")
        print('  $env:MISTRAL_API_KEY = "your-key-here"')
        sys.exit(1)
    return key


def query_leanstral(goal: str, context: str = "", api_key: str = "") -> str:
    """Send a proof goal to Leanstral 1.5 and return the completion."""

    if not api_key:
        api_key = get_api_key()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    if context:
        messages.append({
            "role": "user",
            "content": f"Here is the surrounding Lean 4 file for context:\n\n```lean\n{context}\n```",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I will use this context when completing the proof.",
        })

    messages.append({
        "role": "user",
        "content": f"Complete the following Lean 4 proof. Return ONLY valid Lean 4 code.\n\n{goal}",
    })

    payload = {
        "model": _model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.1,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    with httpx.Client(timeout=120) as client:
        resp = client.post(API_URL, json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"API Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ── File Mode: fill sorry's ──────────────────────────────────────

# Patterns that start a new top-level Lean declaration
_DECL_PATTERN = re.compile(
    r"^(theorem|lemma|def|noncomputable def|instance|example|@\[)",
    re.MULTILINE
)


def _find_theorem_block(lines: list[str], sorry_line_idx: int) -> tuple[int, int]:
    """Find the theorem/lemma block surrounding a sorry on `sorry_line_idx`.

    Returns (start_line_idx, end_line_idx) as 0-indexed inclusive line range.
    """
    # Search backwards for the declaration start
    start = sorry_line_idx
    for i in range(sorry_line_idx, -1, -1):
        stripped = lines[i].lstrip()
        if _DECL_PATTERN.match(stripped):
            start = i
            break
        # Also catch doc comments / attributes right before
        if stripped.startswith("/-") or stripped.startswith("/--"):
            start = i
            break

    # Search forwards for the next declaration (or end of file)
    end = len(lines) - 1
    for i in range(sorry_line_idx + 1, len(lines)):
        stripped = lines[i].lstrip()
        if _DECL_PATTERN.match(stripped):
            end = i - 1
            # Skip trailing blank lines
            while end > sorry_line_idx and lines[end].strip() == "":
                end -= 1
            break

    return start, end


def _extract_context_window(lines: list[str], start: int, end: int,
                            window: int = 30) -> str:
    """Extract surrounding context (imports, definitions) for the LLM."""
    # Always include the first 40 lines (imports + open statements)
    header_end = min(40, start)
    header = "\n".join(lines[:header_end])

    # Include `window` lines before the block
    ctx_start = max(header_end, start - window)
    before = "\n".join(lines[ctx_start:start])

    return f"{header}\n-- [...]\n{before}" if before.strip() else header


def fill_sorries(filepath: str, api_key: str) -> str:
    """Read a .lean file, find `sorry`s, and ask Leanstral to fill them ONE AT A TIME.

    Instead of sending the entire file for rewriting (which causes truncation
    on large files), this extracts each theorem block containing a sorry,
    sends just that block with surrounding context, and splices the fix back in.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()

    # Find all lines containing `sorry` (as a tactic, not in comments/strings)
    sorry_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip comment-only lines and doc comments
        if stripped.startswith("--") or stripped.startswith("/-"):
            continue
        # Match `sorry` as a standalone token (not part of a word)
        if re.search(r'\bsorry\b', stripped):
            sorry_lines.append(i)

    if not sorry_lines:
        print(f"  ⏭  No 'sorry' found in {filepath}. Skipping.")
        return content

    print(f"  📝 Found {len(sorry_lines)} sorry(s) in {os.path.basename(filepath)}")
    print(f"     Using per-sorry replacement mode (safe for large files)")
    print()

    # Process each sorry from LAST to FIRST so line numbers stay valid
    for idx, sorry_line_idx in enumerate(reversed(sorry_lines), 1):
        line_num = sorry_line_idx + 1  # 1-indexed for display

        # Find the theorem block
        block_start, block_end = _find_theorem_block(lines, sorry_line_idx)
        block = "\n".join(lines[block_start:block_end + 1])

        # Get surrounding context
        context = _extract_context_window(lines, block_start, block_end)

        print(f"  [{idx}/{len(sorry_lines)}] Line {line_num}: "
              f"processing block (lines {block_start+1}-{block_end+1})...")

        # Ask Leanstral to fill just this block
        prompt = (
            "The following Lean 4 theorem/lemma contains `sorry`. "
            "Replace the `sorry` with a valid proof.\n"
            "Return ONLY the complete theorem/lemma with the proof filled in. "
            "Do NOT return anything else — no imports, no other theorems, "
            "no markdown fences, no explanations.\n\n"
            f"```lean\n{block}\n```"
        )

        try:
            result = query_leanstral(prompt, context=context, api_key=api_key)

            # Strip markdown fences
            result = re.sub(r"^```(?:lean)?\n?", "", result)
            result = re.sub(r"\n?```$", "", result)
            result = result.strip()

            # Sanity checks
            if "sorry" in result:
                print(f"     ⚠️  Leanstral still returned sorry — keeping original")
                continue
            if len(result) < 10:
                print(f"     ⚠️  Leanstral returned suspiciously short output — skipping")
                continue

            # Replace the block in the file
            new_lines = lines[:block_start] + result.splitlines() + lines[block_end + 1:]
            lines = new_lines

            print(f"     ✅ Filled successfully")

        except Exception as e:
            print(f"     ❌ Error: {e}")

    print()
    return "\n".join(lines) + "\n"


def write_inplace(filepath: str, new_content: str, backup: bool = True):
    """Overwrite a file in place, optionally creating a .bak backup."""

    if backup:
        backup_path = filepath + ".bak"
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"  💾 Backup saved: {os.path.basename(backup_path)}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"  ✅ Updated: {filepath}")


def process_file(filepath: str, api_key: str, inplace: bool = False,
                 output: str = "", no_backup: bool = False):
    """Process a single .lean file: fill sorry's and write result."""

    result = fill_sorries(filepath, api_key)

    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    # If nothing changed (no sorry found), skip writing
    if result == original:
        return False

    if inplace:
        write_inplace(filepath, result, backup=not no_backup)
        return True
    elif output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"  ✅ Written to: {output}")
        return True
    else:
        print(result)
        return True


def process_directory(dirpath: str, api_key: str, no_backup: bool = False):
    """Scan a directory for .lean files with sorry's and fill them in place."""

    lean_files = []
    for root, _dirs, files in os.walk(dirpath):
        for fname in sorted(files):
            if fname.endswith(".lean"):
                lean_files.append(os.path.join(root, fname))

    if not lean_files:
        print(f"No .lean files found in {dirpath}")
        return

    # Filter to only files containing sorry
    sorry_files = []
    for fpath in lean_files:
        with open(fpath, "r", encoding="utf-8") as f:
            if "sorry" in f.read():
                sorry_files.append(fpath)

    if not sorry_files:
        print(f"No sorry's found in any .lean files under {dirpath}")
        return

    print(f"Found {len(sorry_files)} file(s) with sorry's:")
    for fpath in sorry_files:
        print(f"  • {os.path.relpath(fpath, dirpath)}")
    print()

    filled = 0
    for i, fpath in enumerate(sorry_files, 1):
        print(f"[{i}/{len(sorry_files)}] Processing {os.path.relpath(fpath, dirpath)}...")
        try:
            if process_file(fpath, api_key, inplace=True, no_backup=no_backup):
                filled += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()

    print(f"Done! Filled sorry's in {filled}/{len(sorry_files)} file(s).")


def watch_mode(dirpath: str, api_key: str, interval: int = 5, no_backup: bool = False):
    """Watch a directory for .lean files with sorry's and auto-fill them."""

    import time

    print("=" * 60)
    print("  Leanstral 1.5 — Watch Mode")
    print("=" * 60)
    print(f"  Watching: {dirpath}")
    print(f"  Interval: {interval}s")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Track files we've already processed (by content hash)
    processed = {}

    try:
        while True:
            for root, _dirs, files in os.walk(dirpath):
                for fname in sorted(files):
                    if not fname.endswith(".lean"):
                        continue
                    fpath = os.path.join(root, fname)

                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()

                    if "sorry" not in content:
                        continue

                    # Hash content to detect changes
                    import hashlib
                    content_hash = hashlib.md5(content.encode()).hexdigest()

                    if processed.get(fpath) == content_hash:
                        continue  # Already processed this version

                    rel = os.path.relpath(fpath, dirpath)
                    print(f"[{time.strftime('%H:%M:%S')}] Detected sorry in {rel}")

                    try:
                        result = fill_sorries(fpath, api_key)
                        with open(fpath, "r", encoding="utf-8") as f:
                            original = f.read()
                        if result != original:
                            write_inplace(fpath, result, backup=not no_backup)
                        # Mark as processed with NEW hash
                        new_hash = hashlib.md5(result.encode()).hexdigest()
                        processed[fpath] = new_hash
                    except Exception as e:
                        print(f"  ❌ Error: {e}")
                        processed[fpath] = content_hash  # Don't retry same content

                    print()

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nWatch mode stopped.")


# ── Compilation helpers (shared by simplify / bridge modes) ───────

def find_project_root(path: str) -> str:
    """Walk up from `path` to the directory holding a lakefile."""
    d = os.path.dirname(os.path.abspath(path))
    while d and d != os.path.dirname(d):
        if (os.path.exists(os.path.join(d, "lakefile.toml"))
                or os.path.exists(os.path.join(d, "lakefile.lean"))):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(path))


def run_build(root: str) -> tuple[bool, str]:
    """Run `lake build` in `root`. Returns (compiled_ok, combined_output).

    `lake build` (not `lake env lean`) is used deliberately: it applies the
    `leanOptions` from lakefile.toml, so the mathlibStandardSet style linters
    run. That way a "readability" pass can never trade a lint warning for a
    shorter proof without us noticing.
    """
    r = subprocess.run(["lake", "build"], cwd=root,
                       capture_output=True, text=False)
    # Decode with errors='replace' to handle Unicode output on cp949 consoles.
    stdout = r.stdout.decode("utf-8", errors="replace") if r.stdout else ""
    stderr = r.stderr.decode("utf-8", errors="replace") if r.stderr else ""
    return r.returncode == 0, stdout + stderr


def _has_new_warning(output: str, filename: str) -> bool:
    """True if the build output reports a warning mentioning `filename`."""
    base = os.path.basename(filename)
    for line in output.splitlines():
        if "warning:" in line and base in line:
            return True
    return False


# ── Declaration parsing (statement-lock for the simplify pass) ────

# A theorem/lemma/example keyword at the start of a line (MULTILINE so it can be
# found inside a block that opens with a doc comment or attribute).
_STMT_DECL = re.compile(r"(?m)^\s*(?:@\[[^\]]*\]\s*)?(theorem|lemma|example)\b")


def _iter_decls(lines: list[str]):
    """Yield (kind, start_idx, end_idx) for each theorem/lemma/example block.

    `def`/`instance`/etc. are intentionally skipped: rewriting the body of a
    *definition* changes its meaning, whereas a theorem's proof body does not.
    """
    # First pass: the keyword line of every eligible declaration.
    kw_lines = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(?:@\[[^\]]*\]\s*)?(theorem|lemma|example)\b", line)
        if m:
            kw_lines.append((i, m.group(1)))

    # Second pass: extend each start up over its doc-comment / attribute preamble.
    pre_starts = []
    for kw, kind in kw_lines:
        pre = kw
        while pre > 0:
            s = lines[pre - 1].strip()
            if s.endswith("-/") or s.startswith("/-") or s.startswith("@[") or s.startswith("--"):
                pre -= 1
            else:
                break
        pre_starts.append((pre, kw, kind))

    # A block ends just before the NEXT block's (pre-extended) start, so the
    # ranges never overlap on a shared doc-comment line.
    for idx, (pre, kw, kind) in enumerate(pre_starts):
        end = pre_starts[idx + 1][0] - 1 if idx + 1 < len(pre_starts) else len(lines) - 1
        while end > kw and lines[end].strip() == "":
            end -= 1
        yield kind, pre, end


def _statement_of(block: str) -> str | None:
    """The declaration's *type*, normalized: from the decl keyword up to the
    top-level `:=` that begins the proof. Returns None if no such `:=` found.

    Doc comments and attributes above the keyword are excluded, so rewording a
    docstring is allowed while the mathematical statement stays locked.
    """
    m = _STMT_DECL.search(block)
    if not m:
        return None
    kw_pos = m.start(1)  # position of the keyword itself
    depth = 0
    i = kw_pos
    while i < len(block) - 1:
        c = block[i]
        if c in "([{⟨":
            depth += 1
        elif c in ")]}⟩":
            depth -= 1
        elif depth == 0 and block[i] == ":" and block[i + 1] == "=":
            return " ".join(block[kw_pos:i].split())
        i += 1
    return None


def inspect_file(filepath: str, api_key: str, output_report: str = "",
                 max_decls: int = 0) -> None:
    """Collect Leanstral's proof simplifications for manual review.

    For each theorem/lemma, get Leanstral's candidate proof and save it to a
    report file WITHOUT applying the build gate or statement-lock checks.
    This lets you compare original vs candidate side-by-side and decide which
    to keep, then iterate on the prompt based on what you observe.

    Output report format:
      [N/total] name
        original (Nchars): ... first 100 chars ...
        candidate (Nchars): ... first 100 chars ...
        status: (ok|statement_changed|no_proof|error)
        reason: ...

    To apply selected candidates manually, edit the report, then use --simplify
    with a refined prompt based on what you learned.
    """
    root = find_project_root(filepath)
    print(f"  Project root: {root}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    decls = list(_iter_decls(lines))
    if max_decls > 0:
        decls = decls[:max_decls]

    if output_report:
        report = open(output_report, "w", encoding="utf-8")
        print(f"  Saving inspection report to: {output_report}\n")
    else:
        report = None

    def log(msg: str):
        print(msg, flush=True)
        if report:
            report.write(msg + "\n")

    log(f"Leanstral Proof Simplification Inspection")
    log(f"File: {filepath}")
    log(f"Declarations: {len(decls)}\n")

    inspected = 0
    for n, (kind, start, end) in enumerate(decls, 1):
        block = "\n".join(lines[start:end + 1])
        sig = _statement_of(block)
        name = lines[start].strip()[:80]

        log(f"\n[{n}/{len(decls)}] {kind.upper()} {name}")
        log(f"  lines {start+1}-{end+1}")

        if sig is None:
            log(f"  status: NO_STATEMENT")
            continue

        prompt = (
            "Below is a COMPLETE, already-compiling Lean 4 theorem/lemma. "
            "Return a version with a shorter or more idiomatic proof.\n"
            "HARD CONSTRAINTS:\n"
            "  - Keep the statement (everything up to `:=`) EXACTLY as given.\n"
            "  - Keep the same declaration name.\n"
            "  - Return ONLY the theorem/lemma, no fences, no commentary, no imports.\n\n"
            f"```lean\n{block}\n```"
        )

        try:
            result = query_leanstral(prompt, context="", api_key=api_key)
        except Exception as e:
            log(f"  status: QUERY_ERROR")
            log(f"  error: {e}")
            continue

        result = re.sub(r"^```(?:lean)?\n?", "", result)
        result = re.sub(r"\n?```$", "", result).strip()

        orig_chars = len("".join(block.split()))
        cand_chars = len("".join(result.split()))

        log(f"  original chars: {orig_chars}")
        log(f"  candidate chars: {cand_chars}")
        log(f"  delta: {cand_chars - orig_chars:+d} chars")

        if "sorry" in result:
            log(f"  status: INCOMPLETE (still has sorry)")
            log(f"  reason: Leanstral could not complete the proof")
        elif _statement_of(result) != sig:
            log(f"  status: STATEMENT_CHANGED")
            log(f"  reason: Proof statement does not match original")
        elif cand_chars >= orig_chars:
            log(f"  status: NOT_SHORTER")
            log(f"  reason: Candidate is not shorter than original")
        else:
            log(f"  status: OK_SHORTER")
            log(f"  saved for review")

        log(f"  original (first 120 chars):")
        log(f"    {block[:120].replace(chr(10), ' ')}")
        log(f"  candidate (first 120 chars):")
        log(f"    {result[:120].replace(chr(10), ' ')}")

        inspected += 1

    log(f"\n\nInspection complete: {inspected} declarations reviewed")
    if report:
        report.close()


def simplify_file(filepath: str, api_key: str, max_decls: int = 0,
                  min_shrink: int = 8) -> int:
    """Ask Leanstral for a cleaner proof of each theorem/lemma, keeping a
    candidate ONLY if the whole project still builds warning-free AND the
    statement is byte-identical. Returns the number of proofs replaced.

    Everything is compiler-gated: an imperfect suggestion is discarded, never
    committed, so this pass can only shrink proofs or leave them untouched.
    """
    root = find_project_root(filepath)
    print(f"  Project root: {root}")
    print("  Baseline build (must be green before we start)...", flush=True)
    ok, _ = run_build(root)
    if not ok:
        print("  ❌ Baseline build failed — fix the file before simplifying.")
        return 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    # A candidate is written to disk *before* it is built, so if this process is
    # interrupted between the write and the revert the source is left broken.
    # Keep a pristine copy and restore it on any abnormal exit.
    import atexit
    import shutil
    pristine = "\n".join(lines) + "\n"
    shutil.copy2(filepath, filepath + ".bak")
    print(f"  💾 Backup: {os.path.basename(filepath)}.bak")

    _state = {"done": False}

    def _restore_if_interrupted():
        if not _state["done"]:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pristine)
            print(f"\n  ↩  Interrupted — {os.path.basename(filepath)} restored.")

    atexit.register(_restore_if_interrupted)

    decls = list(_iter_decls(lines))
    if max_decls > 0:
        decls = decls[:max_decls]
    print(f"  {len(decls)} theorem/lemma block(s) to try.\n")

    replaced = 0
    # Process last-to-first so earlier line indices stay valid after splicing.
    for n, (kind, start, end) in enumerate(reversed(decls), 1):
        block = "\n".join(lines[start:end + 1])
        sig = _statement_of(block)
        name = lines[start].strip()[:60]
        print(f"  [{n}/{len(decls)}] lines {start+1}-{end+1}: {name}", flush=True)
        if sig is None:
            print("     ⏭  could not isolate statement — skipping")
            continue

        prompt = (
            "Below is a COMPLETE, already-compiling Lean 4 theorem/lemma. "
            "Return a version with a shorter or more idiomatic proof.\n"
            "HARD CONSTRAINTS:\n"
            "  - Keep the statement (everything up to `:=`) EXACTLY as given.\n"
            "  - Keep the same declaration name.\n"
            "  - Return ONLY the theorem/lemma, no fences, no commentary, no imports.\n\n"
            f"```lean\n{block}\n```"
        )
        try:
            result = query_leanstral(prompt, context="", api_key=api_key)
        except Exception as e:
            print(f"     ❌ query error: {e}")
            continue
        result = re.sub(r"^```(?:lean)?\n?", "", result)
        result = re.sub(r"\n?```$", "", result).strip()

        if "sorry" in result:
            print("     ⏭  candidate contains sorry — rejected")
            continue
        if _statement_of(result) != sig:
            print("     ⏭  statement changed — rejected (statement-lock)")
            continue
        # Non-whitespace length is our readability proxy.
        if len("".join(result.split())) > len("".join(block.split())) - min_shrink:
            print("     ⏭  not meaningfully shorter — skipping")
            continue

        # Try it: splice, build, keep only if green and warning-free.
        candidate = lines[:start] + result.splitlines() + lines[end + 1:]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(candidate) + "\n")
        ok, out = run_build(root)
        if ok and not _has_new_warning(out, filepath):
            lines = candidate
            replaced += 1
            print(f"     ✅ accepted (−{len(''.join(block.split())) - len(''.join(result.split()))} chars)")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            reason = "build failed" if not ok else "introduced a warning"
            print(f"     ↩  reverted ({reason})")

    # Ensure the on-disk file matches our accepted state.
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _state["done"] = True
    print(f"\n  Done: {replaced} proof(s) shortened, statements unchanged.")
    return replaced


def bridge_file(filepath: str, api_key: str) -> int:
    """Fidelity mode: fill `sorry`-ed *bridge* goals — statements you have
    transcribed verbatim from the paper — using the surrounding development,
    then certify each by a full build.

    This is `fill_sorries` with a build gate bolted on: a bridge proof is only
    kept if the project compiles, so a green run is a machine-checked proof
    that your formalization implies the paper's printed claim.
    """
    root = find_project_root(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        original = f.read()

    filled = fill_sorries(filepath, api_key)
    if filled == original:
        print("  No bridge sorries to fill.")
        return 0

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(filled)
    print("  Building to certify the bridge proofs...", flush=True)
    ok, out = run_build(root)
    if ok and not _has_new_warning(out, filepath):
        print("  ✅ All bridge proofs compile — the paper's claims are certified "
              "as consequences of the development.")
        return 1
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(original)
        print("  ↩  Build not clean — reverted. Leanstral could not derive at "
              "least one bridge from the current API (or added a warning).")
        print("     Inspect the goal by hand; the statement may need adjusting, "
              "or the bridge may genuinely not follow.")
        return 0


# ── Interactive Mode ──────────────────────────────────────────────

def interactive_mode(api_key: str):
    """Run an interactive loop where you paste theorem statements."""

    print("=" * 60)
    print("  Leanstral 1.5 — Interactive Proof Assistant")
    print("=" * 60)
    print()
    print("Paste a Lean 4 theorem statement (with or without sorry).")
    print("Press Enter twice to submit. Type 'quit' to exit.")
    print()

    while True:
        print("─" * 60)
        print("Enter your Lean 4 goal:")
        print()

        lines = []
        empty_count = 0
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "quit":
                print("Goodbye!")
                return
            if line == "":
                empty_count += 1
                if empty_count >= 2:
                    break
                lines.append(line)
            else:
                empty_count = 0
                lines.append(line)

        goal = "\n".join(lines).strip()
        if not goal:
            continue

        print()
        print("⏳ Querying Leanstral 1.5...")
        print()

        try:
            result = query_leanstral(goal, api_key=api_key)
            print("✅ Completed proof:")
            print()
            print(result)
            print()
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


# ── CLI Entry Point ───────────────────────────────────────────────

def main():
    global _model

    parser = argparse.ArgumentParser(
        description="Leanstral 1.5 — Lean 4 Proof Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Interactive mode
  python leanstral.py

  # Single goal
  python leanstral.py --goal "theorem foo : 1 + 1 = 2 := by sorry"

  # Fill sorry's and print to stdout
  python leanstral.py --file Proof/MyTheorem.lean

  # Fill sorry's and overwrite the file directly (saves .bak backup)
  python leanstral.py --file Proof/MyTheorem.lean --inplace

  # Fill sorry's in ALL .lean files under a directory
  python leanstral.py --dir Proof/

  # Watch mode: auto-fill sorry's as you write them
  python leanstral.py --watch Proof/
        """,
    )
    parser.add_argument(
        "--goal", "-g",
        type=str,
        help="A single Lean 4 theorem statement to complete",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="A .lean file with sorry's to fill",
    )
    parser.add_argument(
        "--inplace", "-i",
        action="store_true",
        help="Overwrite the file in place (creates .bak backup)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating .bak backup when using --inplace",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (for --file mode). Defaults to stdout.",
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Scan a directory for .lean files with sorry's and fill them all",
    )
    parser.add_argument(
        "--watch", "-w",
        type=str,
        help="Watch a directory and auto-fill sorry's as they appear",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Polling interval in seconds for --watch mode (default: 5)",
    )
    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspection mode: collect Leanstral's proof simplifications for each "
             "theorem/lemma WITHOUT applying build gates. Save candidates to a report "
             "so you can compare originals vs proposals and iterate on the prompt. "
             "Use --report to save the inspection report.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Save inspection report to this file (used with --inspect).",
    )
    parser.add_argument(
        "--simplify", "-s",
        type=str,
        help="Readability pass: replace each theorem/lemma proof in FILE with a "
             "shorter one, keeping it only if the project still builds warning-free "
             "and the statement is unchanged (statement-locked, compiler-gated).",
    )
    parser.add_argument(
        "--max-decls",
        type=int,
        default=0,
        help="For --simplify: only try the first N declarations (0 = all). "
             "Useful for a quick smoke test, since each candidate triggers a build.",
    )
    parser.add_argument(
        "--bridge", "-b",
        type=str,
        help="Fidelity pass: fill the `sorry`-ed bridge goals in FILE (statements "
             "transcribed from the paper) from the surrounding development, and keep "
             "them only if the project compiles — certifying the paper's claims.",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()
    _model = args.model

    api_key = get_api_key()

    # ── Single goal mode ──
    if args.goal:
        result = query_leanstral(args.goal, api_key=api_key)
        print(result)
        return

    # ── Inspect mode ──
    if args.inspect:
        if not os.path.exists(args.inspect):
            print(f"ERROR: File not found: {args.inspect}")
            sys.exit(1)
        inspect_file(args.inspect, api_key, output_report=args.report,
                     max_decls=args.max_decls)
        return

    # ── Simplify (readability) mode ──
    if args.simplify:
        if not os.path.exists(args.simplify):
            print(f"ERROR: File not found: {args.simplify}")
            sys.exit(1)
        simplify_file(args.simplify, api_key, max_decls=args.max_decls)
        return

    # ── Bridge (fidelity) mode ──
    if args.bridge:
        if not os.path.exists(args.bridge):
            print(f"ERROR: File not found: {args.bridge}")
            sys.exit(1)
        bridge_file(args.bridge, api_key)
        return

    # ── File mode ──
    if args.file:
        if not os.path.exists(args.file):
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)

        process_file(args.file, api_key,
                     inplace=args.inplace,
                     output=args.output or "",
                     no_backup=args.no_backup)
        return

    # ── Directory mode ──
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"ERROR: Not a directory: {args.dir}")
            sys.exit(1)

        process_directory(args.dir, api_key, no_backup=args.no_backup)
        return

    # ── Watch mode ──
    if args.watch:
        if not os.path.isdir(args.watch):
            print(f"ERROR: Not a directory: {args.watch}")
            sys.exit(1)

        watch_mode(args.watch, api_key,
                   interval=args.interval,
                   no_backup=args.no_backup)
        return

    # ── Interactive mode (default) ──
    interactive_mode(api_key)


if __name__ == "__main__":
    main()
