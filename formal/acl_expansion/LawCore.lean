import Core

namespace ACL

/-- Cartesian product of the shrinking digit ranges in draw order. -/
def traces : Nat → Nat → List (List Nat)
  | _, 0 => [[]]
  | n, d+1 => (List.range n).flatMap fun i => (traces (n-1) d).map (i :: ·)

/-- Enumerate actual encoder input preimages of a fixed ordered answer selection. -/
def preimages (pool xs : List Nat) : List (List Nat) :=
  (traces pool.length xs.length).filter fun digits => draw pool digits == some xs

end ACL
