/-
  Taylor's Theorem in ℝ
  ═════════════════════

  Taylor's theorem states that a sufficiently smooth function can be
  approximated near a point x₀ by a polynomial (the Taylor polynomial),
  with an explicit remainder term.

  For f : ℝ → ℝ with n+1 continuous derivatives on [a, b]:

    f(b) = Σ_{k=0}^{n} f⁽ᵏ⁾(a)/k! · (b-a)ᵏ  +  Rₙ(a, b)

  where the remainder Rₙ can be expressed in several forms:
    • Lagrange:  Rₙ = f⁽ⁿ⁺¹⁾(c)/(n+1)! · (b-a)ⁿ⁺¹  for some c ∈ (a, b)
    • Integral:  Rₙ = ∫ₐᵇ f⁽ⁿ⁺¹⁾(t)/n! · (b-t)ⁿ dt

  This file contains theorem STATEMENTS with `sorry` placeholders,
  intended to be filled by Leanstral 1.5.
-/

import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

open Set

-- ════════════════════════════════════════════════════════════════
-- §1  Taylor Polynomial Evaluation at Specific Points
-- ════════════════════════════════════════════════════════════════

/-
  The zeroth-order Taylor polynomial of f at a, evaluated at b,
  is simply f(a).
-/
theorem taylor_order_zero (f : ℝ → ℝ) (a b : ℝ)
    (s : Set ℝ) (hs : s ∈ nhds a) :
    taylorWithinEval f 0 s a a = f a := by
  simp [taylorWithinEval]

-- ════════════════════════════════════════════════════════════════
-- §2  Smoothness of Polynomial Functions
-- ════════════════════════════════════════════════════════════════

/-
  Polynomial functions are infinitely differentiable (C^∞).
  These are basic building blocks for testing Taylor's theorem.
-/

-- x² is smooth
theorem sq_smooth : ContDiff ℝ ⊤ (fun x : ℝ => x ^ 2) := by
  refine ContDiff.pow ?_ 2
  exact contDiff_id

-- x³ is smooth
theorem cube_smooth : ContDiff ℝ ⊤ (fun x : ℝ => x ^ 3) := by
  refine ContDiff.pow ?_ 3
  exact contDiff_id

-- a polynomial a*x² + b*x + c is smooth
theorem quadratic_smooth (a b c : ℝ) :
    ContDiff ℝ ⊤ (fun x : ℝ => a * x ^ 2 + b * x + c) := by
  refine ContDiff.add ?_ ?_
  · refine ContDiff.add ?_ ?_
    · exact ContDiff.const_mul a (ContDiff.pow contDiff_id 2)
    · exact ContDiff.const_mul b contDiff_id
  · exact contDiff_const

-- ════════════════════════════════════════════════════════════════
-- §3  The Remainder is Small (little-o)
-- ════════════════════════════════════════════════════════════════

/-
  A key consequence of Taylor's theorem: the remainder
  is o((x - a)ⁿ) as x → a.

  That is, the Taylor polynomial is the best polynomial
  approximation of degree n near the expansion point.
-/

-- For a C^n function, the Taylor remainder is o(|x - a|^n)
theorem taylor_remainder_is_littleO
    (f : ℝ → ℝ) (a : ℝ) (n : ℕ)
    (hf : ContDiff ℝ n f) :
    (fun x => f x - taylorWithinEval f n univ a x) =o[nhds a]
    (fun x => (x - a) ^ n) := by
  have := taylorWithinEval_zero hf a
  -- Use the existing lemma from Mathlib
  exact hf.taylorWithinEval_isLittleO (x := a) (n := n)

-- ════════════════════════════════════════════════════════════════
-- §4  Derivative Properties
-- ════════════════════════════════════════════════════════════════

/-
  Basic derivative facts used in Taylor's theorem.
-/

-- The derivative of x² is 2x
theorem deriv_sq : deriv (fun x : ℝ => x ^ 2) = fun x => 2 * x := by
  ext x
  simp [deriv_pow, pow_two, mul_comm]

-- The derivative of x³ is 3x²
theorem deriv_cube : deriv (fun x : ℝ => x ^ 3) = fun x => 3 * x ^ 2 := by
  ext x
  simp [deriv_pow, pow_two, pow_three, mul_comm]

-- x^n is differentiable
theorem differentiable_pow (n : ℕ) :
    Differentiable ℝ (fun x : ℝ => x ^ n) := by
  exact differentiable_pow n

-- ════════════════════════════════════════════════════════════════
-- §5  exp is Its Own Taylor Series
-- ════════════════════════════════════════════════════════════════

/-
  The exponential function has a particularly elegant Taylor expansion:
  every derivative of exp is exp itself, so the Taylor polynomial
  of exp at 0 is exactly Σ xⁿ/n!.

  Here we verify that exp is C^∞ (a prerequisite for Taylor's theorem).
-/

-- exp is infinitely differentiable
theorem exp_smooth : ContDiff ℝ ⊤ Real.exp := by
  exact Real.contDiff_exp

-- The derivative of exp is exp
theorem deriv_exp : deriv Real.exp = Real.exp := by
  ext x
  exact deriv_exp x

-- exp is strictly positive
theorem exp_pos (x : ℝ) : 0 < Real.exp x := by
  exact Real.exp_pos x

-- ════════════════════════════════════════════════════════════════
-- §6  Simple Concrete Verifications
-- ════════════════════════════════════════════════════════════════

/-
  Verify basic facts that appear in Taylor's theorem applications.
-/

-- 0! = 1
example : Nat.factorial 0 = 1 := by
  norm_num

-- 1! = 1
example : Nat.factorial 1 = 1 := by
  norm_num

-- 2! = 2
example : Nat.factorial 2 = 2 := by
  norm_num

-- (x - a)⁰ = 1 for the constant term
example (x a : ℝ) : (x - a) ^ 0 = 1 := by
  simp

-- The first-order Taylor approximation of f at a is f(a) + f'(a)(x-a)
-- This is just the tangent line approximation.
theorem linear_approx (f : ℝ → ℝ) (a : ℝ)
    (hf : DifferentiableAt ℝ f a) :
    HasDerivAt f (deriv f a) a := by
  exact hf.hasDerivAt
