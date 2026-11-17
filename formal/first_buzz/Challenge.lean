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

theorem conservation (start : Rat) (hazards : List Rat) :
    List.sum (masses start hazards) + surviving start hazards = start := by sorry

def forward : Rat → List (Rat × Rat) → Rat
  | _, [] => 0
  | s, (h, payoff) :: rows => s * h * payoff + forward (s * (1 - h)) rows

def weighted : List Rat → List Rat → Rat
  | w :: ws, v :: vs => w * v + weighted ws vs
  | _, _ => 0

theorem mass_payoffs (xs : List (Rat × Rat)) (start : Rat) :
    weighted (masses start (List.map Prod.fst xs)) (List.map Prod.snd xs) = forward start xs := by sorry

def backward : List (Rat × Rat) → Rat
  | [] => 0
  | (h, payoff) :: rows => h * payoff + (1 - h) * backward rows

theorem score_agreement (xs : List (Rat × Rat)) (start : Rat) :
    forward start xs = start * backward xs := by sorry

end FirstBuzz
