import Core

namespace ACL

theorem approvedSchedule (vs : List Rat) (first last : Rat) (ok : approve vs first last = true) :
    vs.head? = some first ∧ vs.getLast? = some last ∧ vs.Pairwise (· ≥ ·) := by
  sorry

theorem legalSuccess (pool ds : List Nat) : (∃ xs, draw pool ds = some xs) ↔ legal pool.length ds := by
  sorry

theorem drawSound (pool ds xs : List Nat) (hn : pool.Nodup) (h : draw pool ds = some xs) :
    xs.length = ds.length ∧ xs.Nodup ∧ xs ⊆ pool ∧ encode pool xs = ds := by
  sorry

theorem drawComplete (pool xs : List Nat) (hn : pool.Nodup) (hx : xs.Nodup) (hs : xs ⊆ pool) :
    draw pool (encode pool xs) = some xs := by
  sorry

theorem goldInsertion (xs : List Nat) (gold pos : Nat) (hn : xs.Nodup) (hg : gold ∉ xs) (hp : pos ≤ xs.length) :
    let menu := xs.insertIdx pos gold
    menu.Nodup ∧ menu.length = xs.length + 1 ∧ menu[pos]? = some gold ∧ menu.eraseIdx pos = xs := by
  sorry

theorem uniformMass (n d k : Nat) (hd : d ≤ n) (hk : 0 < k) :
    traceMass k n d * (ways n d : Rat) * (k : Rat) = 1 := by
  sorry

theorem prizeBounds (lo hi w : Rat) (hh : lo ≤ hi) (hw : 0 ≤ w) (hu : w ≤ 1) :
    lo ≤ prize lo hi w ∧ prize lo hi w ≤ hi := by
  sorry

theorem prizeEndpoints (lo hi : Rat) : prize lo hi 1 = hi ∧ prize lo hi 0 = lo := by
  sorry

theorem prizeMonotone (lo hi a b : Rat) (hh : lo ≤ hi) (hw : b ≤ a) :
    prize lo hi b ≤ prize lo hi a := by
  sorry

theorem telescoping (γ initial : Rat) (path : List Rat) :
    shapeSum γ initial path = γ ^ path.length * lastPotential initial path - initial := by
  sorry

theorem witness : draw [10,20,30] [1,1] = some [20,30] ∧
    shapeSum (1/2) 1 [2,0] = -1 ∧ prize 1 3 (1/2) = 2 := by
  sorry

end ACL
