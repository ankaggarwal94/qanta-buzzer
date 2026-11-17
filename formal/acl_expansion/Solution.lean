import Core

namespace ACL

theorem approvedSchedule (vs : List Rat) (first last : Rat) (ok : approve vs first last = true) :
    vs.head? = some first ∧ vs.getLast? = some last ∧ vs.Pairwise (· ≥ ·) := by
  simpa [approve] using ok

theorem legalSuccess (pool ds : List Nat) : (∃ xs, draw pool ds = some xs) ↔ legal pool.length ds := by
  induction ds generalizing pool with
  | nil => simp [draw, legal]
  | cons i ds ih =>
    by_cases hi : i < pool.length
    · have tail := ih (pool.eraseIdx i)
      rw [List.length_eraseIdx_of_lt hi] at tail
      cases h : draw (pool.eraseIdx i) ds <;> simp_all [draw, legal, getElem?_pos]
    · simp [draw, legal, hi, getElem?_neg]

theorem drawSound (pool ds xs : List Nat) (hn : pool.Nodup) (h : draw pool ds = some xs) :
    xs.length = ds.length ∧ xs.Nodup ∧ xs ⊆ pool ∧ encode pool xs = ds := by
  induction ds generalizing pool xs with
  | nil => simp [draw] at h; subst xs; simp [encode]
  | cons i ds ih =>
    by_cases hi : i < pool.length
    · let x := pool[i]
      have he : pool.erase x = pool.eraseIdx i :=
        List.erase_eq_eraseIdx_of_idxOf (hn.idxOf_getElem i hi)
      cases hh : draw (pool.eraseIdx i) ds with
      | none => simp [draw, getElem?_pos pool i hi, hh] at h
      | some tail =>
        simp [draw, getElem?_pos pool i hi, hh] at h
        subst xs
        obtain ⟨hl, ht, sub, henc⟩ := ih _ _ (hn.eraseIdx i) hh
        have fresh : x ∉ tail := fun hx => hn.not_mem_erase (he ▸ sub hx)
        refine ⟨by simp [hl], List.nodup_cons.mpr ⟨fresh, ht⟩, ?_, ?_⟩
        · exact List.cons_subset.mpr ⟨List.getElem_mem hi, fun _ hmem => List.eraseIdx_subset (sub hmem)⟩
        · simp [encode, ← henc, hn.idxOf_getElem i hi, ← he, x]
    · simp [draw, getElem?_neg pool i hi] at h

theorem drawComplete (pool xs : List Nat) (hn : pool.Nodup) (hx : xs.Nodup) (hs : xs ⊆ pool) :
    draw pool (encode pool xs) = some xs := by
  induction xs generalizing pool with
  | nil => rfl
  | cons x xs ih =>
    obtain ⟨hm, sub⟩ := List.cons_subset.mp hs
    obtain ⟨hx', hxs⟩ := List.nodup_cons.mp hx
    have hi := List.idxOf_lt_length_of_mem hm
    have he : pool.erase x = pool.eraseIdx (pool.idxOf x) :=
      List.erase_eq_eraseIdx_of_idxOf rfl
    have sub' : xs ⊆ pool.erase x := by
      intro y hy
      exact hn.mem_erase_iff.mpr ⟨fun h => hx' (h ▸ hy), sub hy⟩
    have recur := ih _ (List.erase_sublist.nodup hn) hxs sub'
    have hg : pool[pool.idxOf x]? = some x :=
      List.getElem?_eq_some_iff.mpr ⟨hi, List.getElem_idxOf hi⟩
    simp [draw, encode, hg, ← he, recur]

theorem goldInsertion (xs : List Nat) (gold pos : Nat) (hn : xs.Nodup) (hg : gold ∉ xs) (hp : pos ≤ xs.length) :
    let menu := xs.insertIdx pos gold
    menu.Nodup ∧ menu.length = xs.length + 1 ∧ menu[pos]? = some gold ∧ menu.eraseIdx pos = xs := by
  exact ⟨(List.perm_insertIdx gold xs hp).symm.nodup (List.nodup_cons.mpr ⟨hg, hn⟩),
    List.length_insertIdx_of_le_length hp gold, by simp [List.getElem?_insertIdx_self, hp],
    List.eraseIdx_insertIdx_self gold⟩

theorem uniformMass (n d k : Nat) (hd : d ≤ n) (hk : 0 < k) :
    traceMass k n d * (ways n d : Rat) * (k : Rat) = 1 := by
  induction d generalizing n with
  | zero =>
    simp [traceMass, ways]
    have h : (k : Rat) ≠ 0 := by exact_mod_cast Nat.ne_of_gt hk
    grind
  | succ d ih =>
    have hr := ih (n-1) (by omega)
    have hn : (n : Rat) ≠ 0 := by exact_mod_cast (show n ≠ 0 by omega)
    simp [traceMass, ways]
    grind

theorem prizeBounds (lo hi w : Rat) (hh : lo ≤ hi) (hw : 0 ≤ w) (hu : w ≤ 1) :
    lo ≤ prize lo hi w ∧ prize lo hi w ≤ hi := by
  unfold prize
  have hgap : 0 ≤ hi-lo := by grind
  have := Rat.mul_nonneg hgap hw
  have := Rat.mul_le_mul_of_nonneg_left hu hgap
  grind

theorem prizeEndpoints (lo hi : Rat) : prize lo hi 1 = hi ∧ prize lo hi 0 = lo := by
  unfold prize
  grind

theorem prizeMonotone (lo hi a b : Rat) (hh : lo ≤ hi) (hw : b ≤ a) :
    prize lo hi b ≤ prize lo hi a := by
  unfold prize
  have := Rat.mul_le_mul_of_nonneg_left hw (show 0 ≤ hi-lo by grind)
  grind

theorem telescoping (γ initial : Rat) (path : List Rat) :
    shapeSum γ initial path = γ ^ path.length * lastPotential initial path - initial := by
  induction path generalizing initial with
  | nil => simp [shapeSum, lastPotential]; grind
  | cons next path ih =>
    simp [shapeSum, lastPotential, ih, Rat.pow_succ]
    grind

theorem witness : draw [10,20,30] [1,1] = some [20,30] ∧
    shapeSum (1/2) 1 [2,0] = -1 ∧ prize 1 3 (1/2) = 2 := by
  constructor
  · decide
  · simp [shapeSum, prize]
    grind

end ACL
