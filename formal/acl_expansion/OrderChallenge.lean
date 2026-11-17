import Core

namespace ACL

theorem policyOrder (γ initial leftReturn rightReturn : Rat) (left right : List Rat) :
    (leftReturn + shapeSum γ initial left - γ ^ left.length * lastPotential initial left ≤
      rightReturn + shapeSum γ initial right - γ ^ right.length * lastPotential initial right) ↔
    leftReturn ≤ rightReturn := by
  sorry

end ACL
