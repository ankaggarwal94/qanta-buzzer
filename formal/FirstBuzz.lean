import Std

/-! Exact reference for the first-buzz weights in evaluation/metrics.py.
A row contains a conditional buzz probability and the payoff if buzzing now.
For S_q the payoff is correctness. For Expected Wins it is the expected payoff
after conditioning on the opponent. -/
namespace FirstBuzz

def surviving : Rat → List Rat → Rat
  | s, [] => s
  | s, h :: hs => surviving (s * (1 - h)) hs

def masses : Rat → List Rat → List Rat
  | _, [] => []
  | s, h :: hs => s * h :: masses (s * (1 - h)) hs

theorem conservation (s : Rat) (hs : List Rat) :
    (masses s hs).sum + surviving s hs = s := by
  induction hs generalizing s with
  | nil =>
    simp only [masses, surviving, List.sum_nil]
    grind
  | cons h hs ih =>
    simp only [masses, surviving, List.sum_cons]
    have tail := ih (s * (1 - h))
    grind

def forward : Rat → List (Rat × Rat) → Rat
  | _, [] => 0
  | s, (h, payoff) :: rows => s * h * payoff + forward (s * (1 - h)) rows

def weighted : List Rat → List Rat → Rat
  | w :: ws, v :: vs => w * v + weighted ws vs
  | _, _ => 0

theorem mass_payoffs (rows : List (Rat × Rat)) (s : Rat) :
    weighted (masses s (rows.map Prod.fst)) (rows.map Prod.snd) = forward s rows := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    rcases row with ⟨h, payoff⟩
    simp only [List.map_cons, masses, weighted, forward, ih]

def backward : List (Rat × Rat) → Rat
  | [] => 0
  | (h, payoff) :: rows => h * payoff + (1 - h) * backward rows

theorem score_agreement (rows : List (Rat × Rat)) (s : Rat) :
    forward s rows = s * backward rows := by
  induction rows generalizing s with
  | nil => simp [forward, backward]
  | cons row rows ih =>
    rcases row with ⟨h, payoff⟩
    simp only [forward, backward, ih]
    grind

#print axioms mass_payoffs
#print axioms conservation
#print axioms score_agreement

end FirstBuzz
