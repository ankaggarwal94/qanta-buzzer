import Lean.Data.Json
import Core
import TransitionCore
import Core.Wire

open Lean

private def rational (pair : Int × Nat) : Except String Rat := do
  if pair.2 == 0 then throw "zero denominator"
  return (pair.1 : Rat) / (pair.2 : Rat)

private def parseArm : String → Except String ACL.Arm
  | "baseline" => .ok .baseline
  | "constant" => .ok .constant
  | "linear" => .ok .linear
  | "exponential" => .ok .exponential
  | "power" => .ok .power
  | "cosine" => .ok .cosine
  | "step" => .ok .step
  | "piecewise" => .ok .piecewise
  | _ => .error "unknown reward arm"

private def dispatch (req : Json) : Except String Json := do
  let op : String ← req.getObjValAs? String "op"
  match op with
  | "formula" =>
    let readRat := fun key => do rational (← req.getObjValAs? (Int × Nat) key)
    let arm ← parseArm (← req.getObjValAs? String "arm")
    let raw ← req.getObjValAs? (List ((Int × Nat) × (Int × Nat))) "knots"
    let knots ← raw.mapM fun (p, v) => do return (← rational p, ← rational v)
    return ACL.expressionJson (ACL.payoutExpr arm (← readRat "start") (← readRat "end")
      (← readRat "shape") (← readRat "threshold") (← readRat "baseline") knots)
  | "draw" =>
    let n ← req.getObjValAs? Nat "n"
    let digits ← req.getObjValAs? (List Nat) "digits"
    let some xs := ACL.draw (List.range n) digits | throw "invalid draw"
    return toJson xs
  | "schedule" =>
    let values ← req.getObjValAs? (List (Int × Nat)) "values"
    let first ← req.getObjValAs? (Int × Nat) "first"
    let last ← req.getObjValAs? (Int × Nat) "last"
    if ACL.approve (← values.mapM rational) (← rational first) (← rational last) then
      return toJson true
    throw "invalid realized schedule"
  | "insert" =>
    let xs ← req.getObjValAs? (List Nat) "xs"
    let gold ← req.getObjValAs? Nat "gold"
    let pos ← req.getObjValAs? Nat "pos"
    if xs.Nodup ∧ gold ∉ xs ∧ pos ≤ xs.length then
      return toJson (xs.insertIdx pos gold)
    throw "invalid gold insertion"
  | "transition" =>
    let readRat := fun key => do rational (← req.getObjValAs? (Int × Nat) key)
    let base ← readRat "base"
    let gamma ← readRat "gamma"
    let current ← readRat "current"
    let successor ← readRat "successor"
    let prev ← readRat "prev"
    let rounded ← readRat "rounded"
    let next ← readRat "next"
    let even ← req.getObjValAs? Bool "even"
    if ACL.inCell prev rounded next (ACL.total base gamma current successor) even then
      return toJson true
    throw "reward total outside rounding cell"
  | _ => throw "unknown operation"

def main : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  repeat
    let line ← input.getLine
    if line.isEmpty then break
    let result := (Json.parse line).bind dispatch
    let obj := match result with
      | .ok data => Json.mkObj [("value", data)]
      | .error msg => Json.mkObj [("error", toJson msg)]
    output.putStrLn obj.compress
    output.flush
