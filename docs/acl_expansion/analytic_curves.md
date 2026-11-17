# Analytic reward curves

The 11 statements in `formal/acl_curves/Challenge.lean` describe the real-valued formulas underlying `RewardSpec.correct_reward`. A separate agent reviewed the Python definitions, wrote the challenge and recorded its SHA-256 before proof construction. The frozen definitions digest is `c4d6cc74376b7be1156f796e05eedf08070e4c0be2e54c499fd960167aa3656a` and the challenge digest is `1f7df1a569a7242ee3e9f46821ea02f586f524b3e87eb554974612d818ffe563`. The proof body digest is `0b2c01b7510ab9c779b5ca74be2dcc9899606adb79ddd0aac51629e2b403754b`. [Definitions](../../formal/acl_curves/Definitions.lean) · [Challenge](../../formal/acl_curves/Challenge.lean) · [Solution](../../formal/acl_curves/Solution.lean)

For each normalized decay on 0 ≤ x ≤ 1, `ValidCurve` asserts a value of 1 at 0, a value of 0 at 1, antitonicity throughout the interval and range contained in [0,1]. The affine rescaling to arbitrary end ≤ start preserves monotonicity and gives the corresponding endpoints and range. The baseline and constant cases have a separate constant-payout theorem.

| Family | Real formula and assumptions |
| --- | --- |
| Linear | 1 − x |
| Exponential | (exp(−kx) − exp(−k))/(1 − exp(−k)) for k > 0. The k = 0 branch is linear. The one-sided limit at 0 equals 1 − x, proved via numerator and denominator derivative slopes. |
| Power | (1 − x)ᵏ for real k > 0 |
| Cosine | cos²(πx/2) |
| Step | 1 for x < t and 0 otherwise, with 0 < t ≤ 1 |
| Piecewise linear | An arbitrary finite ordered list with first (0,1), last (1,0), strictly increasing abscissas and weakly decreasing ordinates. The recursive interpolation agrees at every knot. The witness has an interior knot and a changed slope. |

The Python exponential evaluates a factored expression using `expm1` and switches evaluation order near zero. Over the reals, exp(−kx) × (1 − exp(−k(1−x))) equals exp(−kx) − exp(−k). The cosine implementation evaluates sin²(π(1−x)/2) beyond the midpoint. Both identities explain the link between the normalized real definition and the runtime expression. Binary64 operations, libm approximations, endpoint dispatch and the final clamp require separate runtime evidence. The verified per-horizon checker validates the actual realized binary64 schedule of audited episodes. [Runtime implementation](../../qb_env/acl_rewards.py) · [Runtime evidence](proofs.md)

From `formal/acl_curves`, native Windows and POSIX environments with Lean 4.35.0 on PATH can build the proof with the same commands. The Lake manifest pins all transitive package revisions, including mathlib `56a005855ea17e0b0958f73ec801ae140b0a1679`.

```sh
lake exe cache get Mathlib.Analysis.SpecialFunctions.Pow.Real Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic Mathlib.Analysis.SpecialFunctions.ExpDeriv Mathlib.Tactic.Linarith Mathlib.Tactic.Positivity Mathlib.Tactic.Ring Mathlib.Tactic.FieldSimp
lake build Solution
```

The independently run Linux comparator accepted all 11 statements with the default Lean checker, Lean paranoid kernel, lean4lean, nanoda, con-leche and con-ron. Lean4lean verified 18,763 declarations. Con-leche and con-ron each verified 18,766. A clean Linux workspace rebuilt the local artifacts against the pinned mathlib dependencies. Use `lake comparator --config comparator.json --paranoid` from this project in Linux, including a Linux VM or WSL distribution with user namespaces and bubblewrap enabled. The separately retained reviewed-input manifest SHA-256 is `dc1277deb55950615066a66915fbf4b60a7c826aeea25a28e3ecc9ae2b9bea49`. [Freeze manifest](../../formal/acl_curves/freeze.json) · [Comparator contract](../../formal/acl_curves/comparator.json) · [Proof receipt](evidence/analytic_curve_proof_receipt.json)

## Runtime formula selection

The audited runtime's compiled `ACL.payoutExpr` takes the arm and exact rational parameters and emits an expression tree. The bridge interprets its constants, progress variable, arithmetic, real power, exponential, cosine of π times the argument, and strict/weak conditional tests over the reals. Two theorem statements identify the tree's value with the independently reviewed curve. `piecewiseExprRefines` covers every finite rational knot list, including an empty list. `payoutExprRefines` covers every arm, every rational parameter tuple and every real progress value. The cosine proof uses the double-angle identity. Parameter and knot validation in actual runs impose the stronger assumptions needed for monotonicity and endpoints. [Compiled expression source](../../formal/acl_expansion/Core/Expr.lean) · [Reference semantics](../../formal/acl_curves/BridgeDefinitions.lean) · [Bridge challenge](../../formal/acl_curves/BridgeChallenge.lean) · [Proof](../../formal/acl_curves/BridgeSolution.lean)

The bridge input freeze has SHA-256 `d11a8e03ca4a7ee843d25761759ac90339a60f831f6c0649d9cc026d8fb3a560`. The two-project verifier stages the canonical runtime opcode source directly from `formal/acl_expansion/Core/Expr.lean` and checks both frozen manifests before launching the sandbox. From the repository root in Linux, or after entering a Linux VM/WSL distribution on macOS/Windows, run the following command. [Verifier](../../formal/acl_curves/verify.py)

```sh
python3 formal/acl_curves/verify.py --freeze-sha256 dc1277deb55950615066a66915fbf4b60a7c826aeea25a28e3ecc9ae2b9bea49 --bridge-freeze-sha256 d11a8e03ca4a7ee843d25761759ac90339a60f831f6c0649d9cc026d8fb3a560
```

The clean Linux gate accepted both bridge theorems with the default checker and all five additional kernels. Their transitive axioms were exactly `propext`, `Classical.choice` and `Quot.sound`. The source-to-source equality establishes the denotation of the selected tree. Serializing its opcodes, executing a generic interval evaluator, enclosing transcendental results, and accepting a unique rounded binary64 value are separate runtime obligations. [Combined gate receipt](evidence/curve_bridge_proof_receipt.json)
