/-
  Fermat's Theorem on Sums of Two Squares (Christmas Theorem, 1640)
  ══════════════════════════════════════════════════════════════════

  An odd prime p can be expressed as a sum of two squares
  if and only if p ≡ 1 (mod 4).

  More precisely:
    • p = 2  ⟹  p = 1² + 1²                          (trivial)
    • p ≡ 1 (mod 4)  ⟹  ∃ a b, a² + b² = p           (hard direction)
    • p ≡ 3 (mod 4)  ⟹  p is NOT a sum of two squares  (easy direction)

  Proof strategy for the hard direction:
    The key insight is that ℤ[i] (the Gaussian integers) form a
    Euclidean domain, hence a unique factorisation domain. If p ≡ 1 (mod 4),
    then −1 is a square mod p, which means p is not irreducible in ℤ[i].
    Factoring p = (a + bi)(a − bi) in ℤ[i] gives p = a² + b².

  This file uses Mathlib's formalisation of these results.
-/

import Mathlib.NumberTheory.SumTwoSquares
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

-- ════════════════════════════════════════════════════════════════
-- §1  The Brahmagupta–Fibonacci Identity
-- ════════════════════════════════════════════════════════════════

/-
  The set of sums of two squares is closed under multiplication:
    (a² + b²)(c² + d²) = (ac − bd)² + (ad + bc)²

  This identity, known since antiquity, shows that to characterise
  sums of two squares it suffices to understand the prime case.
-/

theorem brahmagupta_fibonacci (a b c d : ℤ) :
    (a ^ 2 + b ^ 2) * (c ^ 2 + d ^ 2) =
    (a * c - b * d) ^ 2 + (a * d + b * c) ^ 2 := by
  ring

-- ════════════════════════════════════════════════════════════════
-- §2  The Easy Direction — Mod 4 Obstruction
-- ════════════════════════════════════════════════════════════════

/-
  Every perfect square is congruent to 0 or 1 mod 4:
    • (2k)²   = 4k²     ≡ 0 (mod 4)
    • (2k+1)² = 4k²+4k+1 ≡ 1 (mod 4)

  Therefore a sum of two squares is ≡ 0, 1, or 2 (mod 4) — never 3.
  So any prime p ≡ 3 (mod 4) cannot be a sum of two squares.
-/

-- Squares mod 4 are 0 or 1
theorem sq_mod_four (n : ZMod 4) : n ^ 2 = 0 ∨ n ^ 2 = 1 := by
  decide +revert

-- A sum of two squares mod 4 is 0, 1, or 2 — never 3
theorem sum_two_sq_mod_four (a b : ZMod 4) :
    a ^ 2 + b ^ 2 ≠ 3 := by
  decide +revert

-- ════════════════════════════════════════════════════════════════
-- §3  Fermat's Theorem — The Hard Direction
-- ════════════════════════════════════════════════════════════════

/-
  If p is prime and p % 4 ≠ 3, then p is a sum of two squares.

  The condition p % 4 ≠ 3 covers both:
    • p = 2        (since 2 % 4 = 2)
    • p ≡ 1 (mod 4)

  The proof proceeds through the Gaussian integers ℤ[i]:
    1. Since p ≡ 1 (mod 4), by quadratic reciprocity (or direct
       argument), −1 is a quadratic residue mod p.
    2. So there exists m with m² ≡ −1 (mod p), meaning p ∣ (m² + 1).
    3. In ℤ[i], m² + 1 = (m + i)(m − i), and p divides this product.
    4. But p divides neither factor (since 1/p ∉ ℤ), so p is not
       irreducible in ℤ[i].
    5. By unique factorisation in ℤ[i] (a Euclidean domain via the
       norm), p = (a + bi)(a − bi) for some a, b ∈ ℤ.
    6. Taking norms: p = a² + b².

  Mathlib formalises this as `Nat.Prime.sq_add_sq`.
-/

-- The main theorem: primes not ≡ 3 (mod 4) are sums of two squares
theorem fermat_sum_two_squares (p : ℕ) [hp : Fact p.Prime] (hmod : p % 4 ≠ 3) :
    ∃ a b : ℕ, a ^ 2 + b ^ 2 = p :=
  Nat.Prime.sq_add_sq hmod

-- When p is given as a hypothesis rather than a Fact instance
theorem fermat_sum_two_squares' (p : ℕ) (hp : p.Prime) (hmod : p % 4 ≠ 3) :
    ∃ a b : ℕ, a ^ 2 + b ^ 2 = p := by
  haveI : Fact p.Prime := ⟨hp⟩
  exact Nat.Prime.sq_add_sq hmod

-- ════════════════════════════════════════════════════════════════
-- §4  Concrete Examples
-- ════════════════════════════════════════════════════════════════

/-
  Let us verify the theorem for several small primes.

    2  = 1² + 1²    (p ≡ 2 mod 4)
    5  = 1² + 2²    (p ≡ 1 mod 4)
    13 = 2² + 3²    (p ≡ 1 mod 4)
    17 = 1² + 4²    (p ≡ 1 mod 4)
    29 = 2² + 5²    (p ≡ 1 mod 4)
    37 = 1² + 6²    (p ≡ 1 mod 4)

  Note: 3, 7, 11, 19, 23 are ≡ 3 (mod 4) and are NOT sums of two squares.
-/

example : 1 ^ 2 + 1 ^ 2 = 2  := by norm_num
example : 1 ^ 2 + 2 ^ 2 = 5  := by norm_num
example : 2 ^ 2 + 3 ^ 2 = 13 := by norm_num
example : 1 ^ 2 + 4 ^ 2 = 17 := by norm_num
example : 2 ^ 2 + 5 ^ 2 = 29 := by norm_num
example : 1 ^ 2 + 6 ^ 2 = 37 := by norm_num

-- ════════════════════════════════════════════════════════════════
-- §5  The −1 Quadratic Residue Characterisation
-- ════════════════════════════════════════════════════════════════

/-
  A key lemma underlying the proof is the characterisation of when
  −1 is a square in ℤ/nℤ.

  For squarefree n:
    IsSquare (−1 : ZMod n) ↔ ∀ q ∣ n, q % 4 ≠ 3

  In particular, for a prime p:
    IsSquare (−1 : ZMod p) ↔ p % 4 ≠ 3

  This is a consequence of quadratic reciprocity and the
  Chinese Remainder Theorem.
-/

-- −1 is a square mod n iff no divisor is ≡ 3 (mod 4)
theorem neg_one_is_square_iff (n : ℕ) (hn : Squarefree n) :
    IsSquare (-1 : ZMod n) ↔ ∀ {q : ℕ}, q ∣ n → q % 4 ≠ 3 :=
  ZMod.isSquare_neg_one_iff' hn

-- ════════════════════════════════════════════════════════════════
-- §6  The Full Characterisation for Any Natural Number
-- ════════════════════════════════════════════════════════════════

/-
  Fermat's theorem generalises beyond primes. A positive natural number n
  is a sum of two squares if and only if every prime factor q ≡ 3 (mod 4)
  appears with an EVEN exponent in the prime factorisation of n.

  Equivalently: n is a sum of two squares iff n = a² · m where
  −1 is a square mod m (i.e., no prime factor of m is ≡ 3 mod 4).

  For example:
    • 50 = 2 · 5² = 1² + 7²           ✓  (no factor ≡ 3 mod 4)
    • 45 = 3² · 5 = 3² + 6²           ✓  (3 appears with even exponent)
    • 12 = 2² · 3                      ✗  (3 appears with odd exponent)
-/

-- n is a sum of two squares iff n = a² · m with IsSquare (-1 : ZMod m)
theorem sum_two_sq_iff_sq_mul (n : ℕ) :
    (∃ x y : ℕ, n = x ^ 2 + y ^ 2) ↔
    ∃ a b : ℕ, n = a ^ 2 * b ∧ IsSquare (-1 : ZMod b) :=
  Nat.eq_sq_add_sq_iff_eq_sq_mul
