/-
  The Triangle Inequality in ℝ
  ════════════════════════════

  For all x, y ∈ ℝ:   |x + y| ≤ |x| + |y|

  This is the simplest, most direct formalisation using Mathlib.
-/

import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith

-- ════════════════════════════════════════════════════════════════
-- §1  The Triangle Inequality  —  |x + y| ≤ |x| + |y|
-- ════════════════════════════════════════════════════════════════

-- One-liner: Mathlib already knows this as `abs_add`
theorem triangle_inequality (x y : ℝ) : |x + y| ≤ |x| + |y| :=
  abs_add x y

-- ════════════════════════════════════════════════════════════════
-- §2  Reverse Triangle Inequality  —  | |x| − |y| | ≤ |x − y|
-- ════════════════════════════════════════════════════════════════

theorem reverse_triangle_inequality (x y : ℝ) :
    | |x| - |y| | ≤ |x - y| :=
  abs_abs_sub_abs_le_abs_sub x y

-- ════════════════════════════════════════════════════════════════
-- §3  A From-Scratch Proof (no abs_add)
-- ════════════════════════════════════════════════════════════════

/-
  We prove |x + y| ≤ |x| + |y| by cases, using only the
  characterisation |a| = max a (-a) and basic arithmetic.
-/

theorem triangle_inequality' (x y : ℝ) : |x + y| ≤ |x| + |y| := by
  -- Suffices to show  x + y ≤ |x| + |y|  and  -(x + y) ≤ |x| + |y|
  rw [abs_le]
  constructor
  · linarith [neg_abs_le x, neg_abs_le y]
  · linarith [le_abs_self x, le_abs_self y]

-- ════════════════════════════════════════════════════════════════
-- §4  Concrete Examples
-- ════════════════════════════════════════════════════════════════

example : |(3 : ℝ) + (-5)| ≤ |(3 : ℝ)| + |(-5 : ℝ)| := by norm_num
example : |(-2 : ℝ) + (-3)| ≤ |(-2 : ℝ)| + |(-3 : ℝ)| := by norm_num
example : |(7 : ℝ) + 2| ≤ |(7 : ℝ)| + |(2 : ℝ)| := by norm_num
