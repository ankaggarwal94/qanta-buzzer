import Definitions
import Core.Expr

noncomputable section
namespace ACL.Curves

/-- Mathematical interpretation of the pure opcode tree sent to the enclosure evaluator. -/
def realValue (x : ℝ) : ACL.Expr → ℝ
  | .lit r => (r : ℝ)
  | .progress => x
  | .add a b => realValue x a + realValue x b
  | .sub a b => realValue x a - realValue x b
  | .mul a b => realValue x a * realValue x b
  | .div a b => realValue x a / realValue x b
  | .pow a b => (realValue x a) ^ (realValue x b)
  | .exp a => Real.exp (realValue x a)
  | .cospi a => Real.cos (Real.pi * realValue x a)
  | .ifLT a b yes no => if realValue x a < realValue x b then realValue x yes else realValue x no
  | .ifLE a b yes no => if realValue x a ≤ realValue x b then realValue x yes else realValue x no

def realKnots (ks : List (Rat × Rat)) : List (ℝ × ℝ) :=
  ks.map (fun a => ((a.1 : ℝ), (a.2 : ℝ)))

/-- Reference payout specified with the independently reviewed analytic functions. -/
def expectedPayout (arm : ACL.Arm) (start finish shape threshold baseline : Rat)
    (ks : List (Rat × Rat)) (x : ℝ) : ℝ :=
  match arm with
  | .baseline => (baseline : ℝ)
  | .constant => (start : ℝ)
  | .linear => rescale start finish linear x
  | .exponential => rescale start finish (exponential shape) x
  | .power => rescale start finish (power shape) x
  | .cosine => rescale start finish cosine x
  | .step => rescale start finish (step threshold) x
  | .piecewise => rescale start finish (piecewise (realKnots ks)) x

end ACL.Curves
