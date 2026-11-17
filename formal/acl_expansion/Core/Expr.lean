import Std

namespace ACL

inductive Expr where
  | lit (value : Rat)
  | progress
  | add (a b : Expr)
  | sub (a b : Expr)
  | mul (a b : Expr)
  | div (a b : Expr)
  | pow (a b : Expr)
  | exp (a : Expr)
  | cospi (a : Expr)
  | ifLT (lhs rhs yes no : Expr)
  | ifLE (lhs rhs yes no : Expr)
deriving Repr

inductive Arm where
  | baseline | constant | linear | exponential | power | cosine | step | piecewise
deriving Repr

def segmentExpr (x0 x1 y0 y1 : Rat) : Expr :=
  .add (.lit y1) (.mul (.sub (.lit y0) (.lit y1))
    (.div (.sub (.lit x1) .progress) (.sub (.lit x1) (.lit x0))))

def pieceExpr : List (Rat × Rat) → Expr
  | [] => .lit 0
  | [a] => .lit a.2
  | a :: b :: tail => .ifLE .progress (.lit a.1) (.lit a.2)
      (.ifLT .progress (.lit b.1) (segmentExpr a.1 b.1 a.2 b.2) (pieceExpr (b :: tail)))

def decayExpr (arm : Arm) (shape threshold : Rat) (knots : List (Rat × Rat)) : Expr :=
  let linear : Expr := .sub (.lit 1) .progress
  match arm with
  | .baseline | .constant => .lit 1
  | .linear => linear
  | .exponential =>
    if shape = 0 then linear else
      let e : Expr := .exp (.lit (-shape))
      .div (.sub (.exp (.mul (.lit (-shape)) .progress)) e) (.sub (.lit 1) e)
  | .power => .pow linear (.lit shape)
  | .cosine => .div (.add (.lit 1) (.cospi .progress)) (.lit 2)
  | .step => .ifLT .progress (.lit threshold) (.lit 1) (.lit 0)
  | .piecewise => pieceExpr knots

def payoutExpr (arm : Arm) (start finish shape threshold baseline : Rat)
    (knots : List (Rat × Rat)) : Expr :=
  match arm with
  | .baseline => .lit baseline
  | .constant => .lit start
  | arm => .add (.lit finish) (.mul (.sub (.lit start) (.lit finish))
      (decayExpr arm shape threshold knots))

end ACL
