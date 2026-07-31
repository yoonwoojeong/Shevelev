/-
  Linearity of the Inner Product in Euclidean Space
  ══════════════════════════════════════════════════

  The real inner product ⟪·, ·⟫ is bilinear:
    • ⟪x + y, z⟫ = ⟪x, z⟫ + ⟪y, z⟫         (additive, first arg)
    • ⟪c • x, y⟫ = c · ⟪x, y⟫               (homogeneous, first arg)
    • same in the second argument
    • ⟪x, y⟫ = ⟪y, x⟫                       (symmetry / commutativity)

  We state these for a general real inner product space (E, ⟪·,·⟫),
  which includes ℝⁿ (Euclidean space) as the primary example.
-/

import Mathlib.Analysis.InnerProductSpace.Basic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- ════════════════════════════════════════════════════════════════
-- §1  Additivity
-- ════════════════════════════════════════════════════════════════

-- ⟪x + y, z⟫ = ⟪x, z⟫ + ⟪y, z⟫
theorem inner_product_add_left (x y z : E) :
    ⟪x + y, z⟫_ℝ = ⟪x, z⟫_ℝ + ⟪y, z⟫_ℝ :=
  inner_add_left x y z

-- ⟪x, y + z⟫ = ⟪x, y⟫ + ⟪x, z⟫
theorem inner_product_add_right (x y z : E) :
    ⟪x, y + z⟫_ℝ = ⟪x, y⟫_ℝ + ⟪x, z⟫_ℝ :=
  inner_add_right x y z

-- ════════════════════════════════════════════════════════════════
-- §2  Scalar Homogeneity
-- ════════════════════════════════════════════════════════════════

-- ⟪c • x, y⟫ = c * ⟪x, y⟫
theorem inner_product_smul_left (c : ℝ) (x y : E) :
    ⟪c • x, y⟫_ℝ = c * ⟪x, y⟫_ℝ :=
  real_inner_smul_left c x y

-- ⟪x, c • y⟫ = c * ⟪x, y⟫
theorem inner_product_smul_right (c : ℝ) (x y : E) :
    ⟪x, c • y⟫_ℝ = c * ⟪x, y⟫_ℝ :=
  real_inner_smul_right c x y

-- ════════════════════════════════════════════════════════════════
-- §3  Symmetry
-- ════════════════════════════════════════════════════════════════

-- ⟪x, y⟫ = ⟪y, x⟫   (real inner product is symmetric)
theorem inner_product_comm (x y : E) :
    ⟪x, y⟫_ℝ = ⟪y, x⟫_ℝ :=
  real_inner_comm x y

-- ════════════════════════════════════════════════════════════════
-- §4  Full Bilinearity  —  ⟪a•x + b•y, z⟫ = a⟪x,z⟫ + b⟪y,z⟫
-- ════════════════════════════════════════════════════════════════

theorem inner_product_linear_left (a b : ℝ) (x y z : E) :
    ⟪a • x + b • y, z⟫_ℝ = a * ⟪x, z⟫_ℝ + b * ⟪y, z⟫_ℝ := by
  rw [inner_add_left, real_inner_smul_left, real_inner_smul_left]

theorem inner_product_linear_right (a b : ℝ) (x y z : E) :
    ⟪z, a • x + b • y⟫_ℝ = a * ⟪z, x⟫_ℝ + b * ⟪z, y⟫_ℝ := by
  rw [inner_add_right, real_inner_smul_right, real_inner_smul_right]

-- ════════════════════════════════════════════════════════════════
-- §5  Inner Product with Zero
-- ════════════════════════════════════════════════════════════════

theorem inner_product_zero_left (x : E) : ⟪0, x⟫_ℝ = 0 :=
  inner_zero_left x

theorem inner_product_zero_right (x : E) : ⟪x, 0⟫_ℝ = 0 :=
  inner_zero_right x

-- ════════════════════════════════════════════════════════════════
-- §6  Positive Definiteness
-- ════════════════════════════════════════════════════════════════

-- ⟪x, x⟫ ≥ 0
theorem inner_product_nonneg (x : E) : 0 ≤ ⟪x, x⟫_ℝ :=
  real_inner_self_nonneg

-- ⟪x, x⟫ = 0 ↔ x = 0
theorem inner_product_eq_zero_iff (x : E) : ⟪x, x⟫_ℝ = 0 ↔ x = 0 :=
  inner_self_eq_zero
