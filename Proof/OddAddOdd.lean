/-
  The Sum of Two Odd Numbers is Even
  ═══════════════════════════════════

  If m and n are odd, then m + n is even.

  Recall:
    • n is even  ↔  ∃ k, n = 2 * k
    • n is odd   ↔  ∃ k, n = 2 * k + 1

  Proof:  m = 2a + 1,  n = 2b + 1
          m + n = 2a + 1 + 2b + 1 = 2(a + b + 1)   ∎
-/

import Mathlib.Tactic.Ring
import Mathlib.Data.Nat.Parity

-- ════════════════════════════════════════════════════════════════
-- §1  Main Theorem (on ℤ)
-- ════════════════════════════════════════════════════════════════

theorem odd_add_odd_is_even (m n : ℤ) (hm : Odd m) (hn : Odd n) :
    Even (m + n) := by
  obtain ⟨a, rfl⟩ := hm
  obtain ⟨b, rfl⟩ := hn
  exact ⟨a + b + 1, by ring⟩

-- ════════════════════════════════════════════════════════════════
-- §2  Main Theorem (on ℕ)
-- ════════════════════════════════════════════════════════════════

theorem odd_add_odd_is_even_nat (m n : ℕ) (hm : Odd m) (hn : Odd n) :
    Even (m + n) := by
  obtain ⟨a, rfl⟩ := hm
  obtain ⟨b, rfl⟩ := hn
  exact ⟨a + b + 1, by ring⟩

-- ════════════════════════════════════════════════════════════════
-- §3  One-Liner via Mathlib
-- ════════════════════════════════════════════════════════════════

theorem odd_add_odd_is_even' (m n : ℤ) (hm : Odd m) (hn : Odd n) :
    Even (m + n) :=
  hm.add_odd hn

-- ════════════════════════════════════════════════════════════════
-- §4  Concrete Examples
-- ════════════════════════════════════════════════════════════════

example : Even (3 + 5 : ℤ) := ⟨4, by norm_num⟩
example : Even (7 + 9 : ℤ) := ⟨8, by norm_num⟩
example : Even (1 + 1 : ℤ) := ⟨1, by norm_num⟩
example : Even (99 + 101 : ℤ) := ⟨100, by norm_num⟩
