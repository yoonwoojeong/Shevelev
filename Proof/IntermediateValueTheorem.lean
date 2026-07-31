/-
  The Intermediate Value Theorem
  ═══════════════════════════════

  If f : ℝ → ℝ is continuous on the closed interval [a, b] and y lies
  between f(a) and f(b), then there exists c ∈ [a, b] such that f(c) = y.

  Informally: a continuous real-valued function cannot "jump" over a value
  — it must pass through every intermediate value.

  This file presents several formulations of the IVT:
    • The classical statement on [a, b] with f(a) ≤ y ≤ f(b)
    • The symmetric variant covering both orderings of f(a) and f(b)
    • The zero-crossing (Bolzano) corollary
    • The general topological version via preconnected sets
    • The image-of-interval characterisation
    • Concrete numerical examples

  All proofs use Mathlib's formalisation in
  `Mathlib.Topology.Order.IntermediateValue`.
-/

import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.ContinuousOn
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

open Set

-- ════════════════════════════════════════════════════════════════
-- §1  The Classical Intermediate Value Theorem
-- ════════════════════════════════════════════════════════════════

/-
  The most familiar form of the IVT:

  If f is continuous on [a, b], a ≤ b, f(a) ≤ y ≤ f(b),
  then there exists c ∈ [a, b] with f(c) = y.

  Mathlib provides `intermediate_value_Icc` which handles
  both orderings of f(a) and f(b) in a single disjunction.
-/

-- Version 1: f(a) ≤ y ≤ f(b)
theorem ivt_classical {f : ℝ → ℝ} {a b y : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hfa : f a ≤ y) (hfb : y ≤ f b) :
    ∃ c ∈ Icc a b, f c = y :=
  intermediate_value_Icc hab hf (Or.inl ⟨hfa, hfb⟩)

-- Version 2: f(b) ≤ y ≤ f(a)  (decreasing case)
theorem ivt_classical_desc {f : ℝ → ℝ} {a b y : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hfb : f b ≤ y) (hfa : y ≤ f a) :
    ∃ c ∈ Icc a b, f c = y :=
  intermediate_value_Icc hab hf (Or.inr ⟨hfb, hfa⟩)

-- Version 3: symmetric — y lies between f(a) and f(b) in either order
theorem ivt_symmetric {f : ℝ → ℝ} {a b y : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hy : f a ≤ y ∧ y ≤ f b ∨ f b ≤ y ∧ y ≤ f a) :
    ∃ c ∈ Icc a b, f c = y :=
  intermediate_value_Icc hab hf hy

-- ════════════════════════════════════════════════════════════════
-- §2  Bolzano's Theorem (Zero-Crossing Corollary)
-- ════════════════════════════════════════════════════════════════

/-
  A special case of the IVT often attributed to Bolzano (1817):

  If f is continuous on [a, b], f(a) ≤ 0 ≤ f(b) (or vice versa),
  then f has a zero in [a, b].

  This follows by taking y = 0 in the IVT.
-/

theorem bolzano_zero_crossing {f : ℝ → ℝ} {a b : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hfa : f a ≤ 0) (hfb : 0 ≤ f b) :
    ∃ c ∈ Icc a b, f c = 0 :=
  intermediate_value_Icc hab hf (Or.inl ⟨hfa, hfb⟩)

theorem bolzano_zero_crossing' {f : ℝ → ℝ} {a b : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hfa : 0 ≤ f a) (hfb : f b ≤ 0) :
    ∃ c ∈ Icc a b, f c = 0 :=
  intermediate_value_Icc hab hf (Or.inr ⟨hfb, hfa⟩)

-- ════════════════════════════════════════════════════════════════
-- §3  The Topological Generalisation
-- ════════════════════════════════════════════════════════════════

/-
  The IVT generalises far beyond ℝ. Mathlib's deepest formulation
  is `IsPreconnected.intermediate_value₂`:

  If s is a preconnected set, f and g are continuous on s with values
  in a linear order, and f(a) ≤ g(a) while g(b) ≤ f(b) for some
  a, b ∈ s, then there exists c ∈ s with f(c) = g(c).

  Taking g to be the constant function gives the one-function IVT.
-/

-- IVT for two continuous functions on a preconnected set
theorem ivt_two_functions
    {α : Type*} [TopologicalSpace α]
    {β : Type*} [TopologicalSpace β] [LinearOrder β] [OrderTopology β]
    [OrderClosedTopology β]
    {s : Set α} (hs : IsPreconnected s)
    {f g : α → β} {a b : α}
    (ha : a ∈ s) (hb : b ∈ s)
    (hf : ContinuousOn f s) (hg : ContinuousOn g s)
    (ha' : f a ≤ g a) (hb' : g b ≤ f b) :
    ∃ c ∈ s, f c = g c :=
  hs.intermediate_value₂ ha hb hf hg ha' hb'

-- IVT on a preconnected set (one function, one target value)
theorem ivt_preconnected
    {α : Type*} [TopologicalSpace α]
    {β : Type*} [TopologicalSpace β] [LinearOrder β] [OrderTopology β]
    [OrderClosedTopology β]
    {s : Set α} (hs : IsPreconnected s)
    {f : α → β} {a b : α} {y : β}
    (ha : a ∈ s) (hb : b ∈ s)
    (hf : ContinuousOn f s)
    (ha' : f a ≤ y) (hb' : y ≤ f b) :
    ∃ c ∈ s, f c = y :=
  hs.intermediate_value ha hb hf ha' hb'

-- ════════════════════════════════════════════════════════════════
-- §4  Image of an Interval
-- ════════════════════════════════════════════════════════════════

/-
  A key consequence of the IVT: the continuous image of a closed
  interval under a monotone function is again a closed interval.

  f continuous and monotone on [a, b] ⟹ f '' [a, b] = [f(a), f(b)]
-/

theorem image_Icc_of_continuous_monotone {f : ℝ → ℝ} {a b : ℝ}
    (hab : a ≤ b)
    (hf : ContinuousOn f (Icc a b))
    (hmono : MonotoneOn f (Icc a b)) :
    f '' (Icc a b) = Icc (f a) (f b) :=
  hf.image_Icc_of_monotoneOn hab hmono

-- ════════════════════════════════════════════════════════════════
-- §5  The Univ Version (Globally Continuous Functions)
-- ════════════════════════════════════════════════════════════════

/-
  For a function continuous on all of ℝ (or any connected space):
  if f(a) ≤ y ≤ f(b) then ∃ c, f(c) = y.

  This is `intermediate_value_univ` in Mathlib.
-/

theorem ivt_global {f : ℝ → ℝ} (hf : Continuous f)
    {a b y : ℝ} (ha : f a ≤ y) (hb : y ≤ f b) :
    ∃ c, f c = y :=
  intermediate_value_univ a b hf ha hb

-- ════════════════════════════════════════════════════════════════
-- §6  Fixed Point on [0, 1]
-- ════════════════════════════════════════════════════════════════

/-
  A beautiful corollary of the IVT:

  If f : [0, 1] → [0, 1] is continuous (more precisely,
  f maps [0, 1] into [0, 1]), then f has a fixed point.

  Proof idea: apply Bolzano's theorem to g(x) = f(x) − x.
  Since f(0) ≥ 0 we have g(0) ≥ 0, and since f(1) ≤ 1 we have g(1) ≤ 0.
  So g has a zero, i.e. f(c) = c.

  Mathlib provides `exists_mem_Icc_isFixedPt_of_mapsTo`.
-/

theorem fixed_point_unit_interval {f : ℝ → ℝ}
    (hf : ContinuousOn f (Icc 0 1))
    (hmaps : MapsTo f (Icc 0 1) (Icc 0 1)) :
    ∃ c ∈ Icc (0 : ℝ) 1, f c = c :=
  exists_mem_Icc_isFixedPt_of_mapsTo (by norm_num : (0 : ℝ) ≤ 1) hf hmaps

-- ════════════════════════════════════════════════════════════════
-- §7  Concrete Examples
-- ════════════════════════════════════════════════════════════════

/-
  Example 1:  √2 exists.

  The function f(x) = x² is continuous. f(1) = 1 < 2 and f(2) = 4 > 2.
  By the IVT, there exists c ∈ [1, 2] with c² = 2.
-/

example : ∃ c ∈ Icc (1 : ℝ) 2, c ^ 2 = 2 := by
  have h1 : (1 : ℝ) ≤ 2 := by norm_num
  have hf : ContinuousOn (fun x : ℝ => x ^ 2) (Icc 1 2) :=
    continuousOn_pow 2
  have hfa : (fun x : ℝ => x ^ 2) 1 ≤ 2 := by norm_num
  have hfb : 2 ≤ (fun x : ℝ => x ^ 2) 2 := by norm_num
  exact intermediate_value_Icc h1 hf (Or.inl ⟨hfa, hfb⟩)

/-
  Example 2:  The equation x³ + x − 1 = 0 has a solution in [0, 1].

  f(0) = −1 < 0  and  f(1) = 1 > 0.
  By the IVT (Bolzano), there exists c ∈ [0, 1] with f(c) = 0.
-/

example : ∃ c ∈ Icc (0 : ℝ) 1, c ^ 3 + c - 1 = 0 := by
  have h1 : (0 : ℝ) ≤ 1 := by norm_num
  -- g(x) = x³ + x − 1
  let g : ℝ → ℝ := fun x => x ^ 3 + x - 1
  have hg : ContinuousOn g (Icc 0 1) := by
    apply ContinuousOn.sub
    · exact ContinuousOn.add (continuousOn_pow 3) continuousOn_id
    · exact continuousOn_const
  have hg0 : g 0 ≤ 0 := by norm_num [g]
  have hg1 : 0 ≤ g 1 := by norm_num [g]
  obtain ⟨c, hc_mem, hc_eq⟩ := intermediate_value_Icc h1 hg (Or.inl ⟨hg0, hg1⟩)
  exact ⟨c, hc_mem, by linarith⟩

-- ════════════════════════════════════════════════════════════════
-- §8  Intervals Are Connected
-- ════════════════════════════════════════════════════════════════

/-
  The IVT in Mathlib rests on the topological fact that every
  interval in a conditionally complete, densely ordered space
  with the order topology is preconnected.

  In fact, the converse also holds: every preconnected set in
  such a space is an interval. Together these give:

    S is preconnected  ↔  S is order-connected
-/

-- Closed intervals are preconnected
theorem Icc_is_preconnected (a b : ℝ) :
    IsPreconnected (Icc a b) :=
  isPreconnected_Icc

-- Open intervals are preconnected
theorem Ioo_is_preconnected (a b : ℝ) :
    IsPreconnected (Ioo a b) :=
  isPreconnected_Ioo

-- Preconnected ↔ order-connected (the converse of IVT, in a sense)
theorem preconnected_iff_ordConnected (s : Set ℝ) :
    IsPreconnected s ↔ OrdConnected s :=
  isPreconnected_iff_ordConnected
