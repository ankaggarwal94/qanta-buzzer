import LawCore

namespace ACL

theorem traceCard (n d : Nat) : (traces n d).length = ways n d := by
  sorry

theorem traceLegal (n d : Nat) (ds : List Nat) : ds ∈ traces n d ↔ ds.length = d ∧ legal n ds := by
  sorry

theorem traceDistinct (n d : Nat) : (traces n d).Nodup := by
  sorry

theorem supportedPreimage (pool xs : List Nat) (hp : pool.Nodup) (hx : xs.Nodup) (sub : xs ⊆ pool) :
    preimages pool xs = [encode pool xs] := by
  sorry

theorem unsupportedPreimage (pool xs : List Nat) (hp : pool.Nodup) (bad : ¬ (xs.Nodup ∧ xs ⊆ pool)) :
    preimages pool xs = [] := by
  sorry

theorem goldUnique (xs ys : List Nat) (g p q : Nat) (hx : xs.Nodup) (hy : ys.Nodup)
    (gx : g ∉ xs) (gy : g ∉ ys) (px : p ≤ xs.length) (qy : q ≤ ys.length)
    (eq : xs.insertIdx p g = ys.insertIdx q g) : p = q ∧ xs = ys := by
  sorry

end ACL
