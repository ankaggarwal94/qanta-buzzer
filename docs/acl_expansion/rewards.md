# ACL prefix transitions and reward schedules

`TossupMCEnv(..., acl_reward=RewardSpec(...))` enables this protocol. Legacy callers omit the argument. ACL experiments use `end_mode="no_buzz"`. The opt-in environment also supports forced commitment for compatibility experiments.

## Prefix grid and policy inputs

A question has a fixed ordered menu of K ≥ 2 options and H ≥ 1 prefixes. Reset reveals and scores the first prefix. At every decision the visible count v is in [1,H]. Action 1 through K commits to that option at the visible prefix. Action 0 reveals exactly one more prefix when v < H. At v = H the same action abstains or commits to the current belief argmax according to the configured end mode. Argmax breaks ties at the first slot. Every policy can act on the final prefix, including the only prefix of a singleton. A path takes at most H actions. Padded option actions and actions after termination raise before scoring.

Settlement uses p = (v − 1)/(H − 1), with p = 1 on a singleton. This axis indexes sampled prefix decisions, so two equal token increments need not have equal time increments on an irregular grid. It is private scoring metadata. Text policy inputs contain the visible prefix and menu labels. Numeric policy inputs use the current menu belief, confidence, margin, entropy, stability and the visible count in both legacy time-feature positions. That preserves the input width and removes both dependencies on total question length. Encoders and cached beliefs must use public prefix/menu information only. The ACL runner fits the likelihood on training labels. Private trajectory records, IDs, correctness and gold slots are outside the policy observation.

## Payoffs

The baseline correct payoff is the supplied finite `buzz_correct`. The constant arm always pays `start`. For the remaining arms, a correct answer earns end + (start − end) D(p). Endpoints satisfy 0 ≤ end ≤ start and are finite. D(0) = 1 and D(1) = 0. Each expression defines D on the interior.

| Arm | Decay |
| --- | --- |
| Linear | 1 − p |
| Exponential | (exp(−kp) − exp(−k)) / (1 − exp(−k)) for finite k > 0, and 1 − p at k = 0 |
| Power | (1 − p)^k for finite k > 0 |
| Cosine | (1 + cos(πp)) / 2 |
| Piecewise | Linear interpolation between successive knots (position, decay) |
| Step | 1 before the threshold and 0 at or after it |

Piecewise positions strictly increase, decays weakly decrease and the endpoints must be (0,1) and (1,0). Step thresholds lie in (0,1]. The exponential evaluator factors the numerator using expm1 and divides out the rate separately for small rates. This avoids overflow at large rates and preserves the linear limit for subnormal rates. On each realized horizon, reset validates the finite table against its expected endpoints and exact binary64 nonincrease. It retains that same table for payouts. An optional `acl_schedule_validator` callable independently checks each distinct realized horizon before the environment admits a policy action.

Any wrong selection pays `buzz_incorrect`. Abstention pays `no_buzz_reward`. Each WAIT, including the last, costs `wait_penalty` ≥ 0. A terminal WAIT therefore receives one wait charge and one outcome payoff. An explicit BUZZ has only its outcome payoff. The schedule never scales a wrong answer or an abstention. Each additive reward must be finite. Changing the correct payoff or timing schedule can change policy preferences.

## Potential and episode boundaries

For finite scale c, Φ(s) = c maxᵢ beliefᵢ(s) on live states. Φ is zero at a true terminal. A base transition reward r becomes r + γΦ(s′) − Φ(s), with 0 ≤ γ ≤ 1 and the same discount used by the learner. With identical actions, dynamics, initial distribution and base objective, the discounted shaping terms telescope to −Φ(s₀) on a completed finite episode. This leaves policy ordering unchanged under those assumptions. On a segment of length n ending in a live state, the additional term γⁿΦ(sₙ) persists. Consistent truncated learning bootstraps the shaped continuation V′(sₙ) = V(sₙ) − Φ(sₙ). Dropping that continuation can change the ordering.

A final-prefix WAIT settles the task and returns terminated=true. An external Gym TimeLimit can truncate a live WAIT. The inner `info["acl_transition"]` records the task transition, including its nonzero successor potential. Collectors must use the wrapper's returned termination and truncation flags and the actual final observation for bootstrapping. A simultaneous external time limit and true terminal still has zero continuation.

## Runtime evidence and bounded exploration

Each successful step emits the question index, action, visible counts, progress, wait/outcome/base rewards, current and successor potential, shaping, total and environment termination flags. Replay resets the same immutable environment inputs with `options={"question_idx": index}` and reapplies the recorded actions. The environment checks posterior shape, normalization and range, successor index, terminal disposition count, zero terminal potential and finite reward. A transition violation disarms the environment until a successful reset. The transition error includes its input and output row. Belief errors include the index and offending vector.

The exhaustive test executes the actual Gym reset, mask and step for every legal sequence with H ∈ {1,2,3}, K ∈ {2,3,4}, both end modes and maximum menu size 4. Each live state has K terminal selections and one WAIT successor. Including all-WAIT paths gives HK + 1 trajectories per grid, or 126 paths across the 18 combinations. Tests check each returned flag and reward against an independent oracle, reject padded/out-of-range actions, check telescoping, and attempt a second settlement. Further cases cover correct forced answers, unequal consecutive beliefs, external truncation, singleton endpoint certificates, shuffled private gold/changed suffix length, rejected certificates, finite overflow and deliberately corrupted state. These bounds describe the enumerated cases. Runtime monitors apply to larger realized grids.

Run the following commands from the repository root with the project dependencies installed. They use the same syntax in PowerShell and POSIX shells.

```sh
python -m pytest tests/test_acl_rewards.py -q
python -m scripts.acl_reward_mutations
```

The initial test collection failed on the missing ACL reward module before implementation. The mutation command edits individual method bodies in isolated interpreter memory and runs their pytest witnesses. Its eight mutations target the power exponent, threshold equality, zero-rate limit, duplicated payout, missing discount, missing cutoff potential, skipped final prefix and leaked question length. A successful command requires every witness suite to exit with failed tests for its mutant. The production source files stay untouched.

## Finite arithmetic refinement

`qb_env/acl_settlement.py` treats the stored schedule values, wait/outcome components, discount and state potentials as exact dyadic rationals. The potential of a live state is the binary64 product of its scale and present confidence. Reusing this state function on adjacent edges preserves cancellation. Each transition computes B = wait + outcome and Y = B + γΦ′ − Φ with Python Fraction, then converts Y once to the binary64 scalar returned by Gym. The optional `acl_transition_validator` checks B, γ, Φ, Φ′ and the rounded scalar before it can reach the learner. JSON integer pairs in `exact` encode numerator and positive denominator for B, the shaping term, Y and the rounding errors. JSON readers must preserve arbitrary-precision integers. The ordinary float fields for base and shaping are rounded diagnostics.

The native verifier accepts the scalar when exact Y lies in its supplied binary64 nearest-even rounding cell. Equality at a midpoint is valid only for an even significand. The Python-to-native boundary supplies the rounded scalar's neighboring encodings and parity, including subnormal and overflow edges. This certifies the represented arithmetic relative to the supplied inputs and neighborhood. The compiler, float conversion routines, neighborhood construction, belief estimate, class registry and transcendental schedule evaluations remain trusted. The independent checked schedule contract concerns the realized finite table's endpoints and order.

`learner_reward_float32` is the projection of the emitted Gym scalar onto binary32, as used by SB3's simple reward buffer. T5 accumulates Python reward scalars when forming returns and later casts its targets. The projection recorded here measures double rounding and its signed error at the SB3 reward boundary. It does not model arithmetic in either learner's optimizer. A format overflow in the displayed diagnostics or binary32 projection fails the transition before delivery. Tests include binary64 halfway ties to even in both directions, subnormal underflow and a binary32 midpoint crossed by double rounding.

For an exact trajectory, let eᵢ be a recorded signed rounding error and Yᵢ its exact reference payment. The return difference for the recorded rounded payoffs under exact discounting is precisely Σᵢ γⁱ eᵢ, with absolute bound Σᵢ γⁱ |eᵢ|. If two exact expected policy returns differ by more than the sum of their expected absolute error bounds, rounding these payoffs preserves that strict comparison. Exact ties and smaller margins have no such guarantee. A regression witness uses a terminal wait charge of 2⁻⁵³, abstention payoff 16 and present potential 16. Rounding the base before subtracting the potential returned zero. Single rounding from the exact total returns −2⁻⁵³.

## Policy classes under public payoff instructions

The ordering statement compares policies from the same admissible class, acting with the same information in the same task. Each reward-aware text prompt adds configuration-wide rules, with no question-specific progress value, gold decision or hidden suffix. For a fixed known configuration, these constants are available equally to every policy. An existing frozen model can nevertheless respond differently when the wording changes. A policy that observes private evaluation information or newly informative hidden-state rewards has a different information set. The invariance argument applies to a shared observation-history policy class, or to a fully observed Markov state including any history necessary for the potential. A bounded potential and finite horizon ensure integrability for finite configured payoffs. The formal order result is pathwise. Moving from a common initial state to a common random initial distribution uses linearity of expectation.

## Certified interior payouts

When the verified runner supplies `acl_formula`, the native Lean service emits one JSON expression for the configured reward arm and parameters. The Gym environment evaluates this expression at the exact rational grid index i/(H−1), using 1 for a singleton. This stage constructs the production payout table. The ordinary Python fallback compares step thresholds on the same exact grid and evaluates continuous formulas using binary64 progress. Its intermediate rounding can produce different payouts. The explicit curve certificate records the rational progress even when the binary64 progress diagnostic differs.

The expression language has exact rational literals and progress, arithmetic, powers, exp, cos(πx), and branches with exact rational comparisons. The generic evaluator never receives the arm name. It performs rational-only operations with Fraction and uses exact perfect-root recognition for modest rational powers. For nonrational operations it evaluates the expression with pinned python-flint 0.8.0 and FLINT/Arb ball arithmetic. Each finite lower and upper ball endpoint becomes an outward dyadic bound. Extremely small bounds use directed coarsening well below the binary64 subnormal range. For each point the interpreter increases precision from 96 bits up to 12,288 bits until both bounds round to the same binary64. Ambiguity at that limit fails the reset. When an exact rational midpoint is available, ties-to-even gives a unique candidate directly. [Arb arithmetic guarantees](https://python-flint.readthedocs.io/en/latest/general.html) · [Directed lower/upper endpoints](https://python-flint.readthedocs.io/en/stable/arb.html)

The Lean runtime independently validates each endpoint against that payout's nearest-even cell, using its checked transition operator with zero discount and potentials. Convexity of the rounding cell puts every number between those bounds in the same cell. The table then passes the native endpoint and order check. First transitions carry the emitted expression, and each transition carries its digest, the bound pair, working precision and exact progress. Recomputing with the pinned software verifies the saved certificate. This record is private evaluator data.

This assurance depends on the emitted expression denoting the intended curve, faithful opcode dispatch and serialization, FLINT's enclosure semantics, the rational arithmetic and the float-neighborhood adapter. A proof of monotonicity alone cannot establish that a particular interior payoff is correct. In the regression test a monotone linear payout at p=1/5 has the same endpoints as the required quadratic. It is rejected against the native quadratic expression, whose rounded interior payoff is 16/25. An additional test replaces the ordinary schedule method with the wrong monotone function and confirms that the verified Gym environment still pays the certified amount. The proof and verification do not model the optimization dynamics of PPO.

```sh
python -m pytest tests/test_acl_certified_curve.py -q
```
