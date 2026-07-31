/-
  Syllogism — classical forms of deductive reasoning

  1. Hypothetical Syllogism:  (P → Q) → (Q → R) → (P → R)
  2. Disjunctive Syllogism:   (P ∨ Q) → ¬P → Q
  3. Categorical Syllogism:   (∀ x, P x → Q x) → (∀ x, Q x → R x) → (∀ x, P x → R x)
-/

-- 1. Hypothetical Syllogism: if P implies Q and Q implies R, then P implies R
theorem hypothetical_syllogism (P Q R : Prop)
    : (P → Q) → (Q → R) → (P → R) :=
  fun hpq hqr hp => hqr (hpq hp)

-- 2. Disjunctive Syllogism: if P or Q holds and P is false, then Q holds
theorem disjunctive_syllogism (P Q : Prop)
    : (P ∨ Q) → ¬P → Q :=
  fun hpq hnp => Or.elim hpq
    (fun hp => absurd hp hnp)  -- P contradicts ¬P
    (fun hq => hq)             -- Q holds directly

-- 3. Categorical Syllogism: "All P are Q, all Q are R, therefore all P are R"
--    (the classic Aristotelian form, generalised over a universe α)
theorem categorical_syllogism (α : Type) (P Q R : α → Prop)
    (h1 : ∀ x, P x → Q x) (h2 : ∀ x, Q x → R x)
    : ∀ x, P x → R x :=
  fun x hp => h2 x (h1 x hp)
