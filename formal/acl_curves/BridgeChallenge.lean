import BridgeDefinitions

noncomputable section
namespace ACL.Curves

theorem piecewiseExprRefines (ks : List (Rat × Rat)) (x : ℝ) :
    realValue x (ACL.pieceExpr ks) = piecewise (realKnots ks) x := by
  sorry

theorem payoutExprRefines (arm : ACL.Arm) (start finish shape threshold baseline : Rat)
    (ks : List (Rat × Rat)) (x : ℝ) :
    realValue x (ACL.payoutExpr arm start finish shape threshold baseline ks) =
    expectedPayout arm start finish shape threshold baseline ks x := by
  sorry

end ACL.Curves
