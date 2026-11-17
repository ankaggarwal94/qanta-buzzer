module
public import Definitions
import all Definitions
import Mathlib.Analysis.Calculus.Deriv.Slope

public noncomputable section

namespace ACL.Curves

private theorem ofEnds {f : ℝ → ℝ} (h0 : f 0 = 1) (h1 : f 1 = 0)
    (dec : AntitoneOn f (Set.Icc 0 1)) : ValidCurve f := by
  refine ⟨h0, h1, dec, ?_⟩
  intro x hx
  exact ⟨h1 ▸ dec hx (Set.right_mem_Icc.mpr zero_le_one) hx.2,
         h0 ▸ dec (Set.left_mem_Icc.mpr zero_le_one) hx hx.1⟩

theorem validLinear : ValidCurve linear := by
  apply ofEnds
  · simp [linear]
  · simp [linear]
  · intro x _ y _ xy
    exact sub_le_sub_left xy 1

theorem validExponential (k : ℝ) (hk : 0 ≤ k) : ValidCurve (exponential k) := by
  rcases hk.eq_or_lt with h | h
  · rw [← h, show exponential 0 = linear by funext x; simp [exponential]]
    exact validLinear
  have hden : 0 < 1 - Real.exp (-k) := by
    have := Real.exp_lt_one_iff.mpr (neg_lt_zero.mpr h)
    linarith
  apply ofEnds
  · simp [exponential, h.ne', hden.ne']
  · simp [exponential, h.ne']
  · intro x hx y hy hxy
    simp only [exponential, if_neg h.ne']
    exact div_le_div_of_nonneg_right (sub_le_sub_right
      (Real.exp_le_exp.mpr (mul_le_mul_of_nonpos_left hxy (neg_nonpos.mpr hk))) _) hden.le

theorem exponentialZeroLimit (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    Filter.Tendsto (fun k => exponential k x) (nhdsWithin 0 (Set.Ioi 0)) (nhds (linear x)) := by
  have ex : HasDerivAt (fun k : ℝ => Real.exp (-k * x)) (-x) 0 := by
    simpa using (((hasDerivAt_id (0 : ℝ)).neg.mul_const x).exp)
  have e1 : HasDerivAt (fun k : ℝ => Real.exp (-k)) (-1) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).neg.exp)
  have hn := (hasDerivAt_iff_tendsto_slope_left_right.mp (ex.sub e1)).2
  have hd := (hasDerivAt_iff_tendsto_slope_left_right.mp
    ((hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub e1)).2
  have hh := hn.div hd (by norm_num)
  simp only [zero_sub, neg_neg, div_one] at hh
  rw [show linear x = -x - -1 by unfold linear; ring]
  apply Filter.Tendsto.congr' _ hh
  filter_upwards [self_mem_nhdsWithin] with k (hk : 0 < k)
  simp only [Pi.div_apply, slope_def_field, Pi.sub_apply, neg_zero, zero_mul,
    Real.exp_zero, sub_self, sub_zero, exponential, if_neg hk.ne']
  field_simp [hk.ne']

theorem validPower (k : ℝ) (hk : 0 < k) : ValidCurve (power k) := by
  apply ofEnds
  · simp [power]
  · simp [power, Real.zero_rpow hk.ne']
  · intro x hx y hy hxy
    exact Real.rpow_le_rpow (sub_nonneg.mpr hy.2) (sub_le_sub_left hxy 1) hk.le

theorem validCosine : ValidCurve cosine := by
  apply ofEnds
  · simp [cosine]
  · simp [cosine, Real.cos_pi_div_two]
  · intro x hx y hy hxy
    have hpi := Real.pi_pos
    have x0 := mul_nonneg hpi.le hx.1
    have y1 := mul_le_mul_of_nonneg_left hy.2 hpi.le
    have xy := mul_le_mul_of_nonneg_left hxy hpi.le
    have y0 := mul_nonneg hpi.le hy.1
    have ca := Real.cos_le_cos_of_nonneg_of_le_pi (by linarith : 0 ≤ Real.pi * x / 2)
      (by linarith : Real.pi * y / 2 ≤ Real.pi) (by linarith)
    have cb := Real.cos_nonneg_of_mem_Icc (x := Real.pi * y / 2) ⟨by linarith, by linarith⟩
    exact pow_le_pow_left₀ cb ca 2

theorem validStep (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) : ValidCurve (step t) := by
  apply ofEnds
  · simp [step, ht]
  · simp [step, not_lt.mpr hu]
  · intro x hx y hy hxy
    unfold step
    split_ifs <;> norm_num <;> linarith

theorem validSegment (x0 x1 y0 y1 : ℝ) (hx : x0 < x1) (hy : y1 ≤ y0) :
    let f := segment x0 x1 y0 y1
    f x0 = y0 ∧ f x1 = y1 ∧ AntitoneOn f (Set.Icc x0 x1) ∧
    ∀ x ∈ Set.Icc x0 x1, f x ∈ Set.Icc y1 y0 := by
  have h0 : segment x0 x1 y0 y1 x0 = y0 := by
    simp [segment, (sub_pos.mpr hx).ne']
  have h1 : segment x0 x1 y0 y1 x1 = y1 := by simp [segment]
  have dec : AntitoneOn (segment x0 x1 y0 y1) (Set.Icc x0 x1) := by
    intro a ha b hb hab
    exact add_le_add_right (mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right
      (sub_le_sub_left hab _) (sub_nonneg.mpr hx.le)) (sub_nonneg.mpr hy)) _
  refine ⟨h0, h1, dec, ?_⟩
  intro x h
  exact ⟨h1.ge.trans (dec h (Set.right_mem_Icc.mpr hx.le) h.2),
         (dec (Set.left_mem_Icc.mpr hx.le) h h.1).trans h0.le⟩

private theorem atHead (a : ℝ × ℝ) (tail : List (ℝ × ℝ)) {x : ℝ} (hx : x ≤ a.1) :
    piecewise (a :: tail) x = a.2 := by
  cases tail <;> simp [piecewise, hx]

private theorem pwDec (a : ℝ × ℝ) (tail : List (ℝ × ℝ))
    (ho : (a :: tail).Pairwise (fun a b => a.1 < b.1))
    (hd : (a :: tail).Pairwise (fun a b => b.2 ≤ a.2)) :
    Antitone (piecewise (a :: tail)) := by
  induction tail generalizing a with
  | nil => simp [piecewise, Antitone]
  | cons b rest ih =>
    rcases List.pairwise_cons.mp ho with ⟨headO, tailO⟩
    rcases List.pairwise_cons.mp hd with ⟨headD, tailD⟩
    have ab := headO b (by simp)
    have ba := headD b (by simp)
    have tailDec := ih b tailO tailD
    obtain ⟨s0, s1, sDec, sBound⟩ := validSegment a.1 b.1 a.2 b.2 ab ba
    intro x y xy
    rw [piecewise, piecewise]
    split_ifs with ya yb xa xb
    all_goals first | trivial | linarith | assumption | skip
    all_goals first
      | exact (sBound _ ⟨by linarith, by linarith⟩).2
      | exact sDec ⟨by linarith, by linarith⟩ ⟨by linarith, by linarith⟩ xy
      | exact (tailDec (show b.1 ≤ y by linarith)).trans
          ((atHead b rest le_rfl).le.trans ba)
      | exact (tailDec (show b.1 ≤ y by linarith)).trans
          ((atHead b rest le_rfl).le.trans (sBound _ ⟨by linarith, by linarith⟩).1)
      | exact tailDec xy

private theorem atKnot (a : ℝ × ℝ) (tail : List (ℝ × ℝ)) (z : ℝ × ℝ)
    (ho : (a :: tail).Pairwise (fun a b => a.1 < b.1)) (hz : z ∈ a :: tail) :
    piecewise (a :: tail) z.1 = z.2 := by
  induction tail generalizing a with
  | nil =>
    simp only [List.mem_singleton] at hz
    subst z
    rfl
  | cons b rest ih =>
    rcases List.mem_cons.mp hz with rfl | h
    · exact atHead _ _ le_rfl
    rcases List.pairwise_cons.mp ho with ⟨h0, ht⟩
    have az := h0 z h
    have bz : b.1 ≤ z.1 := by
      rcases List.mem_cons.mp h with rfl | later
      · rfl
      · exact ((List.pairwise_cons.mp ht).1 _ later).le
    simpa [piecewise, not_le.mpr az, not_lt.mpr bz] using ih b ht h

theorem validPiecewise (knots : List (ℝ × ℝ))
    (first : knots.head? = some (0, 1)) (last : knots.getLast? = some (1, 0))
    (ordered : knots.Pairwise (fun a b => a.1 < b.1))
    (descending : knots.Pairwise (fun a b => b.2 ≤ a.2)) :
    ValidCurve (piecewise knots) := by
  cases knots with
  | nil => simp at first
  | cons a tail =>
    apply ofEnds
    · simpa using atKnot a tail (0,1) ordered (List.mem_of_mem_head? first)
    · simpa using atKnot a tail (1,0) ordered (List.mem_of_mem_getLast? last)
    · exact (pwDec a tail ordered descending).antitoneOn _

theorem validRescale (start end_ : ℝ) (hs : end_ ≤ start) (f : ℝ → ℝ) (hf : ValidCurve f) :
    let g := rescale start end_ f
    g 0 = start ∧ g 1 = end_ ∧ AntitoneOn g (Set.Icc 0 1) ∧
    ∀ x ∈ Set.Icc (0 : ℝ) 1, g x ∈ Set.Icc end_ start := by
  refine ⟨by simp [rescale, hf.start], by simp [rescale, hf.finish], ?_, ?_⟩
  · intro x hx y hy hxy
    exact add_le_add_right (mul_le_mul_of_nonneg_left (hf.dec hx hy hxy) (sub_nonneg.mpr hs)) _
  · intro x hx
    obtain ⟨lo, hi⟩ := hf.bounded x hx
    have := mul_le_mul_of_nonneg_left hi (sub_nonneg.mpr hs)
    have := mul_nonneg (sub_nonneg.mpr hs) lo
    constructor <;> unfold rescale <;> linarith

theorem constantPayout (c : ℝ) :
    let f := fun _ : ℝ => c
    f 0 = c ∧ f 1 = c ∧ AntitoneOn f (Set.Icc 0 1) ∧
    ∀ x ∈ Set.Icc (0 : ℝ) 1, f x ∈ Set.Icc c c := by
  simp [AntitoneOn, Set.mem_Icc]

theorem piecewiseWitness :
    let knots : List (ℝ × ℝ) := [(0, 1), (1/2, 1/4), (1, 0)]
    knots.head? = some (0, 1) ∧ knots.getLast? = some (1, 0) ∧
    knots.Pairwise (fun a b => a.1 < b.1) ∧ knots.Pairwise (fun a b => b.2 ≤ a.2) ∧
    piecewise knots (1/2) = 1/4 := by
  norm_num [List.pairwise_cons, piecewise]

end ACL.Curves
