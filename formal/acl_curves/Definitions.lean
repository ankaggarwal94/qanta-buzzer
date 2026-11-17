module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Positivity
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.FieldSimp

public noncomputable section

namespace ACL.Curves

structure ValidCurve (f : ℝ → ℝ) : Prop where
  start : f 0 = 1
  finish : f 1 = 0
  dec : AntitoneOn f (Set.Icc 0 1)
  bounded : ∀ x ∈ Set.Icc (0 : ℝ) 1, f x ∈ Set.Icc (0 : ℝ) 1

def linear (x : ℝ) : ℝ := 1 - x

def exponential (k x : ℝ) : ℝ :=
  if k = 0 then linear x
  else (Real.exp (-k * x) - Real.exp (-k)) / (1 - Real.exp (-k))

def power (k x : ℝ) : ℝ := (1 - x) ^ k

def cosine (x : ℝ) : ℝ := Real.cos (Real.pi * x / 2) ^ 2

def step (t x : ℝ) : ℝ := if x < t then 1 else 0

def rescale (start end_ : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  end_ + (start - end_) * f x

def segment (x0 x1 y0 y1 x : ℝ) : ℝ :=
  y1 + (y0 - y1) * ((x1 - x) / (x1 - x0))

def piecewise (knots : List (ℝ × ℝ)) (x : ℝ) : ℝ :=
  match knots with
  | [] => 0
  | [a] => a.2
  | a :: b :: rest =>
      if x ≤ a.1 then a.2 else if x < b.1 then
        segment a.1 b.1 a.2 b.2 x
      else piecewise (b :: rest) x

end ACL.Curves
