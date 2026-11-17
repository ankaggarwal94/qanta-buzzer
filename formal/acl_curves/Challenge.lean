module

public import Definitions

public noncomputable section

namespace ACL.Curves

theorem validLinear : ValidCurve linear := by
  sorry

theorem validExponential (k : ℝ) (hk : 0 ≤ k) : ValidCurve (exponential k) := by
  sorry

theorem exponentialZeroLimit (x : ℝ) (hx : x ∈ Set.Icc (0 : ℝ) 1) :
    Filter.Tendsto (fun k => exponential k x) (nhdsWithin 0 (Set.Ioi 0)) (nhds (linear x)) := by
  sorry

theorem validPower (k : ℝ) (hk : 0 < k) : ValidCurve (power k) := by
  sorry

theorem validCosine : ValidCurve cosine := by
  sorry

theorem validStep (t : ℝ) (ht : 0 < t) (hu : t ≤ 1) : ValidCurve (step t) := by
  sorry

theorem validSegment (x0 x1 y0 y1 : ℝ) (hx : x0 < x1) (hy : y1 ≤ y0) :
    let f := segment x0 x1 y0 y1
    f x0 = y0 ∧ f x1 = y1 ∧ AntitoneOn f (Set.Icc x0 x1) ∧
    ∀ x ∈ Set.Icc x0 x1, f x ∈ Set.Icc y1 y0 := by
  sorry

theorem validPiecewise (knots : List (ℝ × ℝ))
    (first : knots.head? = some (0, 1)) (last : knots.getLast? = some (1, 0))
    (ordered : knots.Pairwise (fun a b => a.1 < b.1))
    (descending : knots.Pairwise (fun a b => b.2 ≤ a.2)) :
    ValidCurve (piecewise knots) := by
  sorry

theorem validRescale (start end_ : ℝ) (hs : end_ ≤ start) (f : ℝ → ℝ) (hf : ValidCurve f) :
    let g := rescale start end_ f
    g 0 = start ∧ g 1 = end_ ∧ AntitoneOn g (Set.Icc 0 1) ∧
    ∀ x ∈ Set.Icc (0 : ℝ) 1, g x ∈ Set.Icc end_ start := by
  sorry

theorem constantPayout (c : ℝ) :
    let f := fun _ : ℝ => c
    f 0 = c ∧ f 1 = c ∧ AntitoneOn f (Set.Icc 0 1) ∧
    ∀ x ∈ Set.Icc (0 : ℝ) 1, f x ∈ Set.Icc c c := by
  sorry

theorem piecewiseWitness :
    let knots : List (ℝ × ℝ) := [(0, 1), (1/2, 1/4), (1, 0)]
    knots.head? = some (0, 1) ∧ knots.getLast? = some (1, 0) ∧
    knots.Pairwise (fun a b => a.1 < b.1) ∧ knots.Pairwise (fun a b => b.2 ≤ a.2) ∧
    piecewise knots (1/2) = 1/4 := by
  sorry

end ACL.Curves
