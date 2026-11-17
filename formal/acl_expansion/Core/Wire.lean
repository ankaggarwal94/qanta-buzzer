import Core.Expr
import Lean.Data.Json

namespace ACL

open Lean

private def opcode (name : String) (args : List Json) : Json :=
  .arr ((toJson name :: args).toArray)

def expressionJson : ACL.Expr → Json
  | .lit q => opcode "q" [toJson q.num, toJson q.den]
  | .progress => opcode "p" []
  | .add a b => opcode "add" [expressionJson a, expressionJson b]
  | .sub a b => opcode "sub" [expressionJson a, expressionJson b]
  | .mul a b => opcode "mul" [expressionJson a, expressionJson b]
  | .div a b => opcode "div" [expressionJson a, expressionJson b]
  | .pow a b => opcode "pow" [expressionJson a, expressionJson b]
  | .exp a => opcode "exp" [expressionJson a]
  | .cospi a => opcode "cospi" [expressionJson a]
  | .ifLT a b yes no => opcode "if_lt" [expressionJson a, expressionJson b,
      expressionJson yes, expressionJson no]
  | .ifLE a b yes no => opcode "if_le" [expressionJson a, expressionJson b,
      expressionJson yes, expressionJson no]

/-- The wire grammar records opcode, arity, operand order and exact numerator/denominator. -/
inductive Encodes : Expr → Json → Prop
  | lit (q : Rat) : Encodes (.lit q) (.arr #[.str "q", toJson q.num, toJson q.den])
  | progress : Encodes .progress (.arr #[.str "p"])
  | add : Encodes a u → Encodes b v → Encodes (.add a b) (.arr #[.str "add", u, v])
  | sub : Encodes a u → Encodes b v → Encodes (.sub a b) (.arr #[.str "sub", u, v])
  | mul : Encodes a u → Encodes b v → Encodes (.mul a b) (.arr #[.str "mul", u, v])
  | div : Encodes a u → Encodes b v → Encodes (.div a b) (.arr #[.str "div", u, v])
  | pow : Encodes a u → Encodes b v → Encodes (.pow a b) (.arr #[.str "pow", u, v])
  | exp : Encodes a u → Encodes (.exp a) (.arr #[.str "exp", u])
  | cospi : Encodes a u → Encodes (.cospi a) (.arr #[.str "cospi", u])
  | ifLT : Encodes a u → Encodes b v → Encodes yes y → Encodes no n →
      Encodes (.ifLT a b yes no) (.arr #[.str "if_lt", u, v, y, n])
  | ifLE : Encodes a u → Encodes b v → Encodes yes y → Encodes no n →
      Encodes (.ifLE a b yes no) (.arr #[.str "if_le", u, v, y, n])

end ACL
