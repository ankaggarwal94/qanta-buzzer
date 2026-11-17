import BridgeDefinitions
import Definitions

noncomputable section
namespace ACL.Curves

theorem piecewiseExprRefines (ks : List (Rat × Rat)) (x : ℝ) :
    realValue x (ACL.pieceExpr ks) = piecewise (realKnots ks) x := by
  induction ks using ACL.pieceExpr.induct with
  | case1 => simp [ACL.pieceExpr, realValue, realKnots, piecewise]
  | case2 a => simp [ACL.pieceExpr, realValue, realKnots, piecewise]
  | case3 a b tail ih =>
    simp only [realKnots, List.map_cons] at ih ⊢
    rw [ACL.pieceExpr, piecewise]
    simp [realValue, ACL.segmentExpr, segment, ih]

private theorem cosHalf (x : ℝ) :
    (1 + Real.cos (Real.pi * x)) / 2 = cosine x := by
  have h := Real.cos_two_mul (Real.pi * x / 2)
  rw [show 2 * (Real.pi * x / 2) = Real.pi * x by ring] at h
  unfold cosine
  linarith

theorem payoutExprRefines (arm : ACL.Arm) (start finish shape threshold baseline : Rat)
    (ks : List (Rat × Rat)) (x : ℝ) :
    realValue x (ACL.payoutExpr arm start finish shape threshold baseline ks) =
    expectedPayout arm start finish shape threshold baseline ks x := by
  cases arm <;> simp only [ACL.payoutExpr, expectedPayout, ACL.decayExpr, realValue,
    linear, power, rescale, step, Rat.cast_one, Rat.cast_zero]
  case exponential =>
    by_cases h : shape = 0
    · simp [h, realValue, exponential, linear]
    · simp [h, realValue, exponential]
  case cosine => rw [show ((2 : Rat) : ℝ) = 2 by norm_num, cosHalf]
  case piecewise => rw [piecewiseExprRefines]

end ACL.Curves
