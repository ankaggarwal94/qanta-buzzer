import Core.Wire

namespace ACL

theorem expressionWire (e : Expr) : Encodes e (expressionJson e) := by
  induction e <;> constructor <;> assumption

end ACL
