import Std

namespace ACL

/-- Nat labels stand for the distinct canonical class IDs in the ordered pool. -/
def draw : List Nat → List Nat → Option (List Nat)
  | _, [] => some []
  | pool, i :: rest => do
    let x ← pool[i]?
    let xs ← draw (pool.eraseIdx i) rest
    return x :: xs

def encode : List Nat → List Nat → List Nat
  | _, [] => []
  | pool, x :: xs => pool.idxOf x :: encode (pool.erase x) xs

def legal : Nat → List Nat → Prop
  | _, [] => True
  | n, i :: ds => i < n ∧ legal (n-1) ds

def ways : Nat → Nat → Nat
  | _, 0 => 1
  | n, d+1 => n * ways (n-1) d

/-- Independent uniform digit and independent final position, scaled by position count. -/
def traceMass (k : Nat) : Nat → Nat → Rat
  | _, 0 => 1 / (k : Rat)
  | n, d+1 => traceMass k (n-1) d / (n : Rat)

def prize (lo hi weight : Rat) := lo + (hi-lo)*weight

/-- Check every realized prefix payout, represented by its exact binary rational. -/
def approve (values : List Rat) (first last : Rat) : Bool :=
  decide (values.head? = some first ∧ values.getLast? = some last ∧
    values.Pairwise (· ≥ ·))

def shapeSum (γ prev : Rat) : List Rat → Rat
  | [] => 0
  | next :: tail => (γ*next-prev) + γ * shapeSum γ next tail

def lastPotential (prev : Rat) : List Rat → Rat
  | [] => prev
  | next :: tail => lastPotential next tail

end ACL
