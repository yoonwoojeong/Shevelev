/-
  De Morgan's Laws — proved using the Iff structure

  Law 1: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q   (constructive)
  Law 2: ¬(P ∧ Q) ↔ ¬P ∨ ¬Q   (requires classical logic)
-/

-- Law 1: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q
theorem not_or_iff_and_not (P Q : Prop) : ¬(P ∨ Q) ↔ ¬P ∧ ¬Q :=
  Iff.intro
    -- mp: ¬(P ∨ Q) → ¬P ∧ ¬Q
    (fun h => And.intro
      (fun hp => h (Or.inl hp))   -- if P, then P ∨ Q, contradicting h
      (fun hq => h (Or.inr hq))) -- if Q, then P ∨ Q, contradicting h
    -- mpr: ¬P ∧ ¬Q → ¬(P ∨ Q)
    (fun h hpq => Or.elim hpq h.left h.right)
      -- case P: h.left (¬P) applied to P gives False
      -- case Q: h.right (¬Q) applied to Q gives False

-- Law 2: ¬(P ∧ Q) ↔ ¬P ∨ ¬Q  (classical)
open Classical in
theorem not_and_iff_or_not (P Q : Prop) : ¬(P ∧ Q) ↔ ¬P ∨ ¬Q :=
  Iff.intro
    -- mp: ¬(P ∧ Q) → ¬P ∨ ¬Q
    (fun h => Or.elim (em P)
      (fun hp => Or.inr (fun hq => h (And.intro hp hq)))
        -- P holds, so ¬Q must hold (otherwise P ∧ Q contradicts h)
      (fun hnp => Or.inl hnp))
        -- ¬P holds directly
    -- mpr: ¬P ∨ ¬Q → ¬(P ∧ Q)
    (fun h hpq => Or.elim h
      (fun hnp => hnp hpq.left)   -- ¬P contradicts P from P ∧ Q
      (fun hnq => hnq hpq.right)) -- ¬Q contradicts Q from P ∧ Q
