# ACL menu and reward proofs

The five frozen challenges cover 20 theorems about the menu decoder, realized payout schedules and potential shaping. On 2026-11-16 (Pacific), all five passed Lean 4.35.0's sandboxed comparator with every bundled checker. The proof author drafted the specifications. A separate coordinating agent inspected the definitions and approved each challenge before its solution was written. This was agent review. A human reviewer can inspect the five challenge modules without reading the tactics. [Frozen inputs](../../formal/acl_expansion/freeze.json) · [Verification receipt](proof_receipt.json)

The review procedure follows Robert Joseph George's distinction between the validity of a proof term and the meaning of its statement. Challenge files use placeholder proof bodies as permitted by Comparator. The definitions and solutions have no placeholders, declared axioms, native decision procedures or unsafe overrides. Lean's release supplies all dependencies of these five challenges. [Getting the (In)formal Statement Right](https://www.robertj1.com/ai4science/getting-the-statement-right/) · [Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/) · [Lake comparator](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Lake/)

## Statements and scientific interpretation

| Theorem | What the checker established | Conditions needed to use the result |
| --- | --- | --- |
| `legalSuccess` | Shrinking-pool decoding succeeds exactly when every digit is in range at its draw. | Digits are natural numbers. |
| `drawSound`, `drawComplete` | Successful outputs have the requested length, have no repeats, belong to the pool and encode back to the input digits. Every ordered distinct selection from a distinct pool decodes from its encoding. | The order and identity of the input class pool are fixed. |
| `traceCard`, `traceLegal`, `traceDistinct` | Explicit enumeration has the falling-factorial cardinality and contains each legal digit vector of the requested length exactly once. | The enumeration is a proof model. Production draws one vector. |
| `supportedPreimage`, `unsupportedPreimage`, `goldUnique` | Filtering enumerated digits by the actual decoder yields exactly one preimage for a supported ordered selection and none for an unsupported one. Equal valid final menus have the same distractors and gold position. | The class pool has no duplicates and excludes gold. |
| `goldInsertion` | Insertion at any allowed position preserves uniqueness, increases size by one, places gold at the recorded index and dropping that index recovers the selection. | Gold was absent, distractors are distinct and the index is at most the distractor count. |
| `uniformMass` | Trace mass times the falling factorial times the number of legal gold positions equals one. | The recurrence assumes conditionally uniform indices in each shrinking pool and an independent uniform gold position. |
| `approvedSchedule` | When the executable checker accepts a finite list, its actual first and last entries equal the declared endpoints, and every earlier entry is at least every later entry. | Values are exact rationals recovered from the Python binary64 payouts. Endpoints are supplied by the runtime contract. |
| `prizeBounds`, `prizeEndpoints`, `prizeMonotone` | The exact affine reward lies between its endpoints for weights in [0,1], reaches its endpoints, and preserves the ordering of weights when start ≥ end. | An analytic identification of each named decay with its weight requires a separate argument. |
| `telescoping` | The exact discounted sum of one-step differences equals discounted last potential minus the initial potential on every finite path, including empty paths. | Each difference uses the same discount and successor potential as the next step. |
| `policyOrder` | Two paths with a common initial potential have the same order of returns after adding shaping and subtracting their respective discounted endpoint potentials. | Endpoint potential is zero on termination. Bootstrap at truncation must apply the corresponding residual correction. |
| `checkedTransition` | An accepted rounded total is in the exact rounding cell of the base return plus discounted successor potential minus current potential. Midpoint equality requires an even target. | Python supplies the correct neighboring representables, parity, and exact input operands. |
| `expressionWire` | The emitted Lean JSON value realizes the inductive grammar for every typed expression, including opcode, arity, operand position and rational literal. | JSON number rendering and the host parser and interpreter are trusted. |
| `witness` | A nonempty two-draw trace, a two-transition terminal path with discount one-half, and an interior interpolated payout all evaluate to the announced values. | The closed example exercises nontrivial cases. |

For `d=k-1` distractors selected from a pool of `n≥d` distinct classes, there are `n(n−1)…(n−d+1)` legal digit traces and `k` gold positions. The inverse theorems identify each supported ordered menu with exactly one such trace and position. Its conditional probability is therefore `1/[k·n(n−1)…(n−d+1)]`. The Python baseline has a point mass on the supplied menu. For prefix and mixed strategies which deterministically select the members and permute them, the pool size equals `d`, yielding probability `1/k!`. For ranked bands and category pools, condition on the actual pool membership, exclusions, scores and tie rules. A seeded PRNG realizes a deterministic sequence. No theorem establishes entropy, independence of a particular seeded stream, category adequacy or semantic correctness of a reviewer’s alias decision. The logged log probability uses ordinary floating-point arithmetic. [Decoder and encoder](../../formal/acl_expansion/Core.lean)

With a common initial state, the exact shaped and bootstrapped return differs from its original return by the action-independent constant minus the initial potential. For random initial states with the same distribution under both policies, taking expectations subtracts the same expected initial potential when the expectations exist. This extension uses linearity of expectation. The formal corollary is pathwise. Comparing policies at different initial-state distributions can change the ordering. Bootstrapping from a truncated state with its potential reset to zero can also change the ordering. Rounded step rewards and downstream discounted sums introduce rounding errors and can tie or reorder near-equal estimates. Optimizer trajectories, learned policy quality and empirical calibration need separate evidence. [Order challenge](../../formal/acl_expansion/OrderChallenge.lean)

## Executable boundary

Build the standalone executable and call `LeanRuntime` to decode a draw, check an insertion, request a payout formula, or validate a supplied schedule or transition. The six installation probes and the Python client tests exercise this interface. Episode settlement, an environment caller and the learner flag belong to the downstream experiment integration. A run must retain its own caller-level evidence before attributing a payout to a learner. [Lean entry point](../../formal/acl_expansion/Runtime.lean) · [Python client](../../formal/acl_expansion/runtime.py)

From the repository root after building the executable:

```sh
python -c "from formal.acl_expansion.runtime import LeanRuntime; r = LeanRuntime(); print(r.draw(4, [1, 2])); r.close()"
```

JSON uses arbitrary-precision signed numerators and positive integer denominators. Python supplies the exact ratio of each finite binary64 value. The base can be a Fraction so adding wait and outcome costs requires no intermediate rounding. The rounding witness uses `nextafter` in both directions, the significand’s low bit and virtual magnitude `2**1024` at finite overflow boundaries. The rational representation treats both signed zeros identically. The tests exercise positive and negative subnormal boundaries, midpoint ties, zero, the smallest normal, the largest finite value and rejection of inconsistent totals. The modeled potential starts from the realized, rounded state potential. [Runtime tests](../../tests/test_acl_lean_runtime.py) · [Transition challenge](../../formal/acl_expansion/TransitionChallenge.lean)

The trusted computing base includes the statement review, the independently retained freeze digest, checker implementations, the sandbox and OS. Executing these definitions additionally trusts Lean’s compiler, C toolchain, runtime, processor, the bridge’s encoding, Python’s float and Fraction implementations, and the caller’s argument selection. The trusted numerical components include the JSON parser, Python opcode interpreter, Fraction and python-flint/FLINT interval operations. The math kernels do not verify those components, text normalization, reservoir filtering, model embeddings, shape-parameter parsing, the real-world label classes, game-state adaptation, or the entire Python experiment. The finite schedule checker alone accepts a wrong curve with correct endpoints and monotonicity. A downstream interpreter must bind its numerical enclosure to the emitted formula. [Analytic proofs](analytic_curves.md) A wrong endpoint supplied as both the expected and observed endpoint can pass. Correctly rounded individual totals do not guarantee a bit-exact telescoping identity for the accumulated trajectory.

## Reproduction and negative control

Use a Linux environment with real bubblewrap, working user namespaces, Python 3.11+ and the official Lean 4.35.0 toolchain. In PowerShell, enter a Linux distribution with `wsl -d <distribution>` from the checkout and run the same Linux verification command. On macOS use a Linux VM for Comparator. Native Windows and POSIX builds of the executable use the same Lake commands.

From the repository root, run the following command in Linux. The verified freeze digest was recorded by the coordinating agent independently of any proof-body changes. Review changed definitions before adopting a new digest.

```sh
python3 formal/acl_expansion/verify.py --freeze-sha256 873dc65663c7bc59359bb77ea7b111e6b385a92cf3053a0795d46915c6022f5c --law-freeze-sha256 9d7a420ee917caf13a314fcc625930365a2c76904b93ffdaecf4eeb59523d75c --wire-freeze-sha256 352464ab213fc34363d04122ae37071fb7d45c704d6ee3abd7c058b3f184ff9a --negative-control
```

The verifier copies explicit files to a fresh staging directory, verifies the trusted inputs and invokes five separate sandboxed comparators. It excludes pre-existing build products. Each configuration restricts transitive axioms to `propext`, `Quot.sound` and `Classical.choice`. The default kernel, Lean paranoid kernel, lean4lean, nanoda, con-leche and con-ron all accepted. The negative control substitutes a separately compiled solution-side dependency that commutes the multiplication in `ACL.prize`. The proof still builds, but Comparator rejects the changed definition. The receipt records checker counts and the exact transitive axioms by exported theorem. [Verifier](../../formal/acl_expansion/verify.py) · [Receipt](proof_receipt.json)

The historical production receipt identifies its original source and binary hashes. The build command below uses the current runtime freeze.

For the audited Linux production executable, run this command from the repository root. It checks the frozen production sources, builds with the checkout read-only and networking disabled, runs six requests in a second read-only sandbox and installs the probed ELF at the standard Lake target. Two independent staging builds produced the same SHA-256, and the Python production client successfully decoded a draw with the installed executable. [Production build receipt](evidence/production_runtime_receipt.json) · [Source manifest](../../formal/acl_expansion/runtime-freeze.json)

```sh
python3 formal/acl_expansion/build_runtime.py --freeze-sha256 b15dec870259717ec14b84112935fc2918f05b6173d8d5f84025e7827f2acd24 --install
```

For a native local executable and an axiom listing on an already reviewed checkout, run the following commands in either PowerShell or a POSIX shell.

```sh
cd formal/acl_expansion
lake build acl_runtime OrderSolution TransitionSolution
lake env lean Audit.lean
lake -f law-lakefile.toml build LawSolution
lake -f law-lakefile.toml env lean LawAudit.lean
lake -f wire-lakefile.toml build WireSolution
lake -f wire-lakefile.toml env lean WireAudit.lean
cd ../..
python -m pytest -q tests/test_acl_lean_runtime.py
```

The runtime tests skip when the executable has not been built. The original receipt records the five native Windows tests at its recorded source hashes. The proof project used the Lean 4.35.0 release, with Linux archive SHA-256 `7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d`. [Lean release](https://github.com/leanprover/lean4/releases/tag/v4.35.0)
