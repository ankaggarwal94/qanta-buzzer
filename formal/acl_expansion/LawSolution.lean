import LawCore
import Solution

namespace ACL

theorem traceCard (n d : Nat) : (traces n d).length = ways n d := by
  induction d generalizing n with
  | zero => rfl
  | succ d ih => simp [traces, ways, List.length_flatMap, ih, List.map_const', List.sum_replicate_nat]

theorem traceLegal (n d : Nat) (ds : List Nat) : ds ∈ traces n d ↔ ds.length = d ∧ legal n ds := by
  induction d generalizing n ds with
  | zero => cases ds <;> simp [traces, legal]
  | succ d ih =>
    cases ds with
    | nil => simp [traces]
    | cons i ds => simp [traces, legal, ih]; grind

theorem traceDistinct (n d : Nat) : (traces n d).Nodup := by
  induction d generalizing n with
  | zero => simp [traces]
  | succ d ih =>
    apply List.pairwise_flatMap.mpr
    constructor
    · intro i _
      apply List.pairwise_map.mpr
      exact (ih (n-1)).imp fun h => fun eq => h (List.cons.inj eq).2
    · apply List.nodup_range.imp
      intro i j h a ha b hb eq
      obtain ⟨x, _, rfl⟩ := List.mem_map.mp ha
      obtain ⟨y, _, rfl⟩ := List.mem_map.mp hb
      exact h (List.cons.inj eq).1

theorem supportedPreimage (pool xs : List Nat) (hp : pool.Nodup) (hx : xs.Nodup) (sub : xs ⊆ pool) :
    preimages pool xs = [encode pool xs] := by
  have decoded := drawComplete pool xs hp hx sub
  obtain ⟨hlen, _, _, _⟩ := drawSound pool _ xs hp decoded
  have valid := (legalSuccess pool (encode pool xs)).mp ⟨xs, decoded⟩
  have mem : encode pool xs ∈ preimages pool xs := by
    exact List.mem_filter.mpr ⟨(traceLegal _ _ _).mpr ⟨hlen.symm, valid⟩, by simp [decoded]⟩
  have only : ∀ ds ∈ preimages pool xs, ds = encode pool xs := by
    intro ds hm
    have ok : draw pool ds = some xs := by simpa using (List.mem_filter.mp hm).2
    exact (drawSound pool ds xs hp ok).2.2.2.symm
  have nd : (preimages pool xs).Nodup := List.filter_sublist.nodup (traceDistinct _ _)
  cases h : preimages pool xs with
  | nil => simp_all
  | cons a rest =>
    rw [h] at only nd
    have ha := only a (by simp)
    have empty : rest = [] := List.eq_nil_iff_forall_not_mem.mpr (fun b hb => by
      have eq := only b (by simp [hb])
      exact (List.nodup_cons.mp nd).1 (by simpa [eq, ha] using hb))
    simp [ha, empty]

theorem unsupportedPreimage (pool xs : List Nat) (hp : pool.Nodup) (bad : ¬ (xs.Nodup ∧ xs ⊆ pool)) :
    preimages pool xs = [] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro ds hm
  have ok : draw pool ds = some xs := by simpa using (List.mem_filter.mp hm).2
  obtain ⟨_, nd, sub, _⟩ := drawSound pool ds xs hp ok
  exact bad ⟨nd, sub⟩

theorem goldUnique (xs ys : List Nat) (g p q : Nat) (hx : xs.Nodup) (hy : ys.Nodup)
    (gx : g ∉ xs) (gy : g ∉ ys) (px : p ≤ xs.length) (qy : q ≤ ys.length)
    (eq : xs.insertIdx p g = ys.insertIdx q g) : p = q ∧ xs = ys := by
  obtain ⟨nd, _, hp, ex⟩ := goldInsertion xs g p hx gx px
  obtain ⟨_, _, hq, ey⟩ := goldInsertion ys g q hy gy qy
  rw [← eq] at hq ey
  obtain ⟨ip, gp⟩ := List.getElem?_eq_some_iff.mp hp
  obtain ⟨iq, gq⟩ := List.getElem?_eq_some_iff.mp hq
  have pIdx := nd.idxOf_getElem p ip
  have qIdx := nd.idxOf_getElem q iq
  rw [gp] at pIdx
  rw [gq] at qIdx
  have pq := pIdx.symm.trans qIdx
  exact ⟨pq, ex.symm.trans (pq ▸ ey)⟩

end ACL
