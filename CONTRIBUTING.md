# Contributing & Development Guide

## For Readers: Navigating This Repository

### Understand the Project

1. **Start here:** [README.md](README.md) — overview and quick-start instructions
2. **Mathematics:** Sections 1–4 of README show the definitions and theorems with full equational proofs
3. **Lean code:** Same sections also explain how each math step maps to Lean tactics
4. **Main file:** [`Shevelev.lean`](Shevelev.lean) — the formal proof (121 lines)

### Read the Proof in Lean

Open [`Shevelev.lean`](Shevelev.lean) and read top-to-bottom:

- **Lines 1–22:** Module docstring explaining the index convention and overall structure
- **Lines 26–68:** `sum_geom_pow` — the core lemma: root-of-unity filter over geometric families
- **Lines 70–100:** `theorem1_H` and `theorem1_K` — instantiate the filter for Formulas (8) and (9)
- **Lines 102–130:** `theorem1_H_exp` and `theorem1_K_exp` — concrete exponential forms

### Trace a Single Theorem

Example: to understand **Formula (8)** (`theorem1_H`):

1. Read §2 of README: the **mathematical proof** with step-by-step equation chain
2. Read the **Lean Correspondence** subsection: shows the exact `calc` lines
3. Open [`Shevelev.lean`](Shevelev.lean) line 62: see `theorem1_H` in full context
4. Note the last line: `simpa only [one_mul, one_zpow] using sum_geom_pow …`
   - This says: apply the general `sum_geom_pow` lemma with $b = 1$, which cancels the weight

### Build and Verify Locally

**Option 1: Docker (fastest, no installation)**

```bash
# Build locally
docker build -t shevelev:latest .

# Run with cached Mathlib
docker run --rm -it shevelev:latest
root@proof:/proof# lake env lean Shevelev.lean  # instant
```

**Option 2: Local with cache**

```bash
python get_cache.py           # ~5–10 min (one-time)
lake env lean Shevelev.lean   # ~5–10 min first time, then cached
```

**Option 3: Full local build**

```bash
lake build  # rebuilds Mathlib (~20–30 min first time)
```

**Output:** Exit code 0, empty diagnostics = success.

---

## For Developers: Contributing Changes

### Code Organization

The proof is organized around **one core lemma** and **four instantiations**:

| Declaration | Lines | Purpose |
|---|---|---|
| `sum_geom_pow` | 26–68 | General filter: $\sum_j (bq^j+1)^m (bq^j)^{-r}$ → sums over divisible terms |
| `theorem1_H` | 70–75 | Instantiate with $(b,q) = (1,\omega)$ |
| `theorem1_K` | 77–104 | Instantiate with $(b,q) = (\mu, \mu^2)$ and compute weights |
| `theorem1_H_exp` | 106–112 | Concrete: $\omega = e^{2\pi i/n}$ |
| `theorem1_K_exp` | 114–130 | Concrete: $\mu = e^{\pi i/n}$ |

**Design principle:** No auxiliary definitions. The sums $H_s(m,n)$ and $K_s(m,n)$ appear directly in theorem statements.

### Making Changes

**To refactor or extend:**

1. **Understand the dependency chain:**
   - All four theorems depend on `sum_geom_pow`
   - Never split `sum_geom_pow` into smaller lemmas without updating all four proofs

2. **Edit [`Shevelev.lean`](Shevelev.lean):**
   ```bash
   # Make your change, then verify immediately
   lake env lean Shevelev.lean
   # If successful, exit code 0 and empty output
   ```

3. **Update the README** if:
   - You change the proof strategy (update §2–3 "Lean Correspondence")
   - You add/remove theorems (update the "Contents" section)
   - You change line numbers (update references)

4. **Commit with a clear message:**
   ```bash
   git add Shevelev.lean README.md
   git commit -m "Brief summary of change

   Longer explanation if needed. Reference the line numbers or
   sections that changed.
   
   e.g., 'Inlined hterm into sum_geom_pow (lines 37–41 → 40–41).'"
   ```

### Common Tasks

#### Simplify the proof further

Look for opportunities to:
- Merge `calc` steps (fewer `_ =` lines)
- Inline single-use `have` statements
- Use automation (`simp`, `ring`, `omega`) more aggressively

**Example:** If you find a `have key : ...` that's used only once, try:
```lean
rw [show (complex expr) = (simpler expr) by (proof)]  -- instead of a lemma
```

#### Add a new instantiation

Want to add Formula (8) with a *different* root, like $\zeta = e^{\pi i/n}$ (3rd root)?

1. Add a new theorem:
```lean
theorem theorem1_H_3rd_root (n : ℕ) (hn : 0 < n) (r : ℤ) (m : ℕ) :
    ... := by
  rw [theorem1_H n hn (Complex.isPrimitiveRoot_exp_of_3 n hn.ne') r m, ...]
```

2. Update README §4 (Concrete Instantiations) with the new formula

3. Verify: `lake env lean Shevelev.lean`

#### Fix a compilation error

1. **Read the error carefully** — Lean's error messages pinpoint the exact tactic that failed
2. **Check line numbers:** Match them against [`Shevelev.lean`](Shevelev.lean)
3. **Test incrementally:** Make one small fix, recompile, repeat
4. **Ask:** If stuck, file an issue with:
   - The full error output
   - The change you made
   - What you expected to happen

### Testing the Build

Before pushing, always:

```bash
# 1. Typecheck the main file
lake env lean Shevelev.lean

# 2. (Optional) Run a full build
lake build

# 3. Verify no untracked changes are left
git status

# 4. Review your commits
git log --oneline -5
```

---

## Project Structure

### Directory Layout

```
Shevelev/
├── Shevelev.lean           # Main formalization
├── Proof.lean              # Library root (imports Shevelev.lean)
├── README.md               # Mathematical proofs + Lean code
├── CONTRIBUTING.md         # This file
├── get_cache.py            # Cache download helper
├── lakefile.toml           # Lake build config
├── lean-toolchain          # Lean 4.30.0
├── lake-manifest.json      # Mathlib v4.30.0
└── .github/workflows/
    └── lean_action_ci.yml  # CI: typecheck on every push
```

### Build Files (generated, safe to delete)

```
.lake/                      # Build cache (can be large ~500MB)
.leanscratch/               # Scratch files from local editing
```

If things go wrong: `rm -rf .lake && python get_cache.py` to reset.

---

## CI & Verification

Every push runs GitHub Actions (`.github/workflows/lean_action_ci.yml`):

1. **Typecheck:** `lake env lean Shevelev.lean` (fast)
2. **Report:** Pass or fail, logged publicly

View results: Click the green ✓ or red ✗ on any commit on GitHub.

---

## References

- **Shevelev's paper:** [arXiv:1706.01454v4](https://arxiv.org/abs/1706.01454)
- **Lean 4:** https://lean-lang.org/
- **Mathlib4:** https://github.com/leanprover-community/mathlib4

---

## Questions?

- **How do I understand the math?** Start with README §1–2.
- **How do I run the code?** Follow "Quick Start" in README.
- **How do I add a theorem?** Follow "Common Tasks" → "Add a new instantiation" above.
- **What if the build is slow?** Use `python get_cache.py` to cache Mathlib.
- **What if I break something?** `git diff` shows your changes; `git checkout .` reverts them.
