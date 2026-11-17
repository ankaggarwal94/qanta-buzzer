import TransitionCore

namespace ACL

theorem checkedTransition (base γ current successor prev rounded next : Rat) (even : Bool)
    (ok : inCell prev rounded next (total base γ current successor) even = true) :
    let v := base + γ * successor - current
    prev < rounded ∧ rounded < next ∧ prev + rounded ≤ 2*v ∧ 2*v ≤ rounded + next ∧
    (even = false → prev + rounded < 2*v ∧ 2*v < rounded + next) := by
  sorry

end ACL
