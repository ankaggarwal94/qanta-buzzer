import Std

namespace ACL

def total (base γ current successor : Rat) := base + γ * successor - current

/-- The caller supplies the exact adjacent representables and the parity of the target. -/
def inCell (prev rounded next value : Rat) (even : Bool) : Bool :=
  decide (prev < rounded ∧ rounded < next ∧
    (prev + rounded < 2 * value ∨ (even = true ∧ prev + rounded = 2 * value)) ∧
    (2 * value < rounded + next ∨ (even = true ∧ 2 * value = rounded + next)))

end ACL
