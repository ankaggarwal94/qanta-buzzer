# Opt-in ACL expansion experiments

Start in the active repository at revision `aef3ca43e36c682ff33ed32217ab1bbab4e2408e` plus the files in this change. The manifest records Git HEAD and hashes the implementation on disk, including uncommitted code. The original training scripts keep their historical CLI and horizon behavior. The `acl` subcommand selects the new protocol. [Evidence ledger](ledger.md)

## Bounded run

Use Python 3.12 for the recorded CPU environment. Activate a native virtual environment, then run these commands from the repository root in PowerShell or a POSIX shell. Lean 4.35.0 must be on PATH for the executable build.

```sh
python -m pip install -c configs/acl_expansion/runtime-constraints.txt -e ".[acl-verified]" pytest
cd formal/acl_expansion
lake build acl_runtime
cd ../..
python -m scripts.acl_expansion smoke --policy ppo --config configs/acl_expansion/smoke.yaml --out artifacts/acl_expansion --lean-core
python -m scripts.acl_expansion smoke --policy t5 --config configs/acl_expansion/smoke.yaml --out artifacts/acl_expansion --lean-core
```

Each learner runs 18 conditions on the same eight training groups and two held-out groups. Each of the ten menu choices is tested with the baseline reward. Each of the eight reward choices is tested with the frozen fixture menu, plus a separate potential-shaping run. The PPO budget is four environment steps per condition. T5 updates its policy head using four sampled episodes per condition; its encoder is frozen for these smokes. Both learners reload their checkpoints for inference, then replay training traces and held-out predictions. The fixture's baseline and constant payoffs coincide by design. [Smoke inputs](../../scripts/acl_smoke_data.py) · [Realized run IDs and hashes](evidence/learner_smokes.json)

The same 36 conditions also completed on seven reviewed Geography questions from the real ACL freeze. Three questions supplied training episodes, two supplied validation episodes and two supplied test episodes. Each realized menu carried source-specific judgments for all four options. The baseline preserved the original independent-pool options. These runs exercise training, inference and replay on real inputs. Their purposive selection and size preclude performance estimates. To reproduce them with the private archive, run the following from the repository root. [Review and selection](reviewed_geography_subset.md) · [Real learner receipts](evidence/reviewed_geography_smokes.json)

```sh
python -m pip install -e ".[acl-private-review]"
python -m scripts.acl_reviewed_subset --archive "<path-to-archive>/ACL_Dataset_5000_Frozen_Inputs_2026-10-02.zip"
python -m scripts.acl_expansion smoke --config configs/acl_expansion/reviewed_geography_smoke.yaml --policy ppo --lean-core --out artifacts/acl_expansion/real_smokes
python -m scripts.acl_expansion smoke --config configs/acl_expansion/reviewed_geography_smoke.yaml --policy t5 --lean-core --out artifacts/acl_expansion/real_smokes
```

The same run can be invoked from the existing training and evaluation entry points. For example, these two commands train and replay one T5 arm.

```sh
python scripts/train_t5_policy.py acl --config configs/acl_expansion/smoke.yaml --menu mixed --reward power --out artifacts/acl_expansion --lean-core
python scripts/evaluate_all.py acl --policy t5 --config configs/acl_expansion/smoke.yaml --menu mixed --reward power --out artifacts/acl_expansion --lean-core
```

Use `scripts/train_ppo.py acl` for the belief policy. Omitting `--lean-core` selects the Python implementation and yields a separate run identity. The checks on arm names, K, oracle coverage, donor membership and prefix visibility still apply.

## Scientific inputs and matched ablations

The full example pins the original private evaluator file SHA-256. The archive materializer checks the frozen inputs and writes the evaluator, reserved donor pool and unresolved review draft to the paths used by this checkout. The original archive had 5,000 pending answerline reviews. Its 5,000 targets and 1,100 donors need global class judgments. Displayed classes also need judgments for their specific source question. An answerline can accept two globally distinct concepts. Preparation stops when required review evidence is missing or contradictory. [Full config](../../configs/acl_expansion/full.yaml) · [Materialization and review handoff](archive_compatibility.md) · [Source-bound judgments](source_review.md)

```sh
python -m scripts.acl_full_inputs materialize --archive "<path-to-archive>/ACL_Dataset_5000_Frozen_Inputs_2026-10-02.zip"
```

The full example explicitly uses calibration for training, selection for validation, and test for final comparison. The source freeze originally assigned those names for inference. Use this crosswalk only for the new expansion experiment. Source and duplicate groups are checked before building a menu. Donors come from the disjoint reservoir, or from training groups when no reservoir is supplied. A missing class, conflicting answer rule or short pool stops preparation.

All menus are fixed at the first visible prefix. A prefix-retrieval arm ranks candidates using that prefix and retains the menu for the rest of the episode. This keeps slot identity stable through WAIT actions. Slot 0 waits. Slots 1 through K choose the displayed option. A final WAIT incurs one wait cost and the pass payoff. Every prefix, including the last and a singleton prefix, permits an answer. [Menu distributions and tie rules](menu_oracles.md) · [Reward and horizon definitions](rewards.md)

To run a single end-to-end condition, including held-out inference:

```sh
python -m scripts.acl_expansion run --config configs/acl_expansion/full.yaml --policy t5 --menu sbert_hard --reward exponential --potential-scale 0.5 --seed 1 --lean-core --out artifacts/acl_expansion
```

The crossed experiment contains 10 menu arms × 8 reward arms × 2 potential scales per learner and seed. The following commands specify all 960 runs for two learners and seeds 1, 2 and 3. The full config trains the T5 encoder and head. It sets 20 PPO updates per condition. These are reproduction commands; the evidence receipts cover bounded runs.

After compiling the global registry, collect every candidate question/class pair needed by the matrix with this command. It prepares menus without training a policy. Save the adjudicated copy at `data/acl_expansion/review/question_class_decisions.json`. The runner checks its judgments again on each realized menu. [Completing and checking the private decisions](source_review.md)

```sh
python -m scripts.acl_scientific_review template --config configs/acl_expansion/full.yaml --menu baseline --menu uniform_global --menu uniform_category --menu tfidf_easy --menu tfidf_hard --menu sbert_easy --menu sbert_hard --menu mixed --menu prefix_tfidf --menu prefix_sbert --seed 1 --seed 2 --seed 3 --out data/acl_expansion/review/full_matrix_template.json
```

**PowerShell**

```powershell
foreach ($trialSeed in 1,2,3) {
    foreach ($learner in 'ppo','t5') {
        python -m scripts.acl_expansion matrix --config configs/acl_expansion/full.yaml --policy $learner --seed $trialSeed --out artifacts/acl_expansion --lean-core
        if ($LASTEXITCODE -ne 0) { throw "ACL matrix failed" }
    }
}
```

**POSIX shell**

```sh
for trial_seed in 1 2 3; do
  for learner in ppo t5; do
    python -m scripts.acl_expansion matrix --config configs/acl_expansion/full.yaml --policy "$learner" --seed "$trial_seed" --out artifacts/acl_expansion --lean-core || exit 1
  done
done
```

The runner processes one condition at a time. Each menu is prepared once per sweep and reused across its reward variants. The content key includes realized menus, oracle, dataset, split roles, seed, reward parameters, model and tokenizer assets, training settings, software versions, source hashes and the optional verified core. A file lock serializes duplicate workers. Each checkpoint has a content manifest and a parent digest. Repeating a command verifies completed blocks and starts at the next block. Interrupted partial directories have no completion manifest. Each block starts fresh episodes with a seed derived from the original seed and update offset. Block size is part of the experiment. Archived model checkpoints and optimizer states can require substantial space in a full matrix.

## Public inference and analysis

Export sends only the per-prefix decision prompt to the model. Evaluate each prompt in a fresh context. The public file includes identifiers for joining predictions. The model input is the `prompt` field alone. Supply a JSON object mapping every job ID to an integer action. Scoring validates complete coverage and replays reached actions against the same private spec. Run both commands with identical arm overrides.

```sh
python -m scripts.acl_expansion export --config configs/acl_expansion/smoke.yaml --split test --lean-core --out artifacts/acl_expansion/public.json
python -m scripts.acl_expansion score --config configs/acl_expansion/smoke.yaml --split test --lean-core --actions predictions.json --out artifacts/acl_expansion/score.json
python -m scripts.compare_acl_arms --reference artifacts/acl_expansion/REFERENCE_RUN_ID --arm artifacts/acl_expansion/ARM_RUN_ID --split test --out artifacts/acl_expansion/comparison.json
```

The comparison requires identical held-out groups, question IDs, learner settings, source and review identities, seed, K, fixed payoff settings and discount. It reports paired differences in accuracy and the discordant counts. Discounted base and shaped returns are reported within each objective. The paired OE/MC and historical five-round IMCQA interfaces have separate byte-compatible templates, grading rules and a pinned CPU inference smoke. [Matched inference](paired_inference.md)
