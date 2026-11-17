"""Train, infer and replay opt-in ACL arms. Run with python -m scripts.acl_expansion."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from importlib.metadata import version
import math
import os
import shutil
from pathlib import Path
import platform
import random
import subprocess
import tempfile
from time import monotonic

import gymnasium as gym
import numpy as np
import torch
import yaml
from filelock import FileLock

from models.likelihoods import SBERTLikelihood, TfIdfLikelihood
from qb_data.config import merge_overrides
from scripts._common import parse_overrides
from qb_data.acl_menus import STRATEGIES
from qb_data.acl_protocol import ArmSpec, Prepared, digest, encode, prepare, publish_json
from qb_env.text_wrapper import TextObservationWrapper
from qb_env.tossup_env import TossupMCEnv
from scripts.acl_checkpoints import hash_tree as _hash_tree, verify_block
from scripts.stopdff_v5.identity import load_json_strict, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SBERT = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
T5 = ("t5-small", "df1b051c49625cf57a3d0d8d3863ed4d13564fe4")
REWARDS = ("baseline", "constant", "linear", "exponential", "power", "cosine", "piecewise", "step")


@lru_cache(maxsize=2)
def snapshot(model):
    from huggingface_hub import snapshot_download
    name, revision = model
    try:
        cached = Path(snapshot_download(name, revision=revision, local_files_only=True))
        if _complete_snapshot(cached, model):
            return str(cached)
    except FileNotFoundError:
        pass
    fresh = Path(snapshot_download(name, revision=revision,
                                  allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"]))
    if not _complete_snapshot(fresh, model):
        raise ValueError(f"incomplete pinned model snapshot for {name}")
    return str(fresh)


def _complete_snapshot(path, model):
    files = ["config.json", "model.safetensors", "tokenizer_config.json"]
    if model == T5:
        files.extend(["spiece.model", "tokenizer.json"])
    elif model == SBERT:
        files.extend(["tokenizer.json", "modules.json", "1_Pooling/config.json", "sentence_bert_config.json",
                      "config_sentence_transformers.json", "special_tokens_map.json", "vocab.txt"])
    return all((path / name).is_file() and (path / name).stat().st_size for name in files)


@lru_cache(maxsize=1)
def sbert():
    return SBERTLikelihood(snapshot(SBERT))


@dataclass(frozen=True)
class Training:
    updates: int = 20
    rollout_size: int = 32
    epochs: int = 4
    batch_size: int = 8
    freeze_encoder: bool = False
    max_input_length: int = 512
    threads: int = 2
    checkpoint_updates: int = 10

    def manifest(self, policy):
        if policy not in {"ppo", "t5"}:
            raise ValueError("unknown learner")
        return {**asdict(self), "rollout_unit": "transitions" if policy == "ppo" else "complete_episodes"}

    def __post_init__(self):
        for name in ("updates", "epochs", "batch_size", "rollout_size", "max_input_length", "threads",
                     "checkpoint_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.freeze_encoder) is not bool or min(self.batch_size, self.rollout_size) < 2:
            raise ValueError("PPO needs a boolean freeze_encoder and batch/rollout size >= 2")


class Runtime:
    def __init__(self, prepared: Prepared):
        self.prepared = prepared
        # The learner's vocabulary fit uses training option labels only. Each
        # position at run time scores the same kind of public label string.
        labels = sorted({label for q in prepared.questions["train"] for label in q.options})
        self.likelihood = TfIdfLikelihood(corpus_texts=labels)

    def environment(self, questions, *, text=False, sink=None):
        spec = self.prepared.spec
        env = TossupMCEnv(questions=questions, likelihood_model=self.likelihood, K=spec.k,
                          acl_schedule_validator=(self.prepared.core.validate_schedule
                                                  if self.prepared.core is not None else None),
                          acl_transition_validator=(self.prepared.core.validate_transition
                                                    if self.prepared.core is not None else None),
                          acl_formula=(self.prepared.core.formula if self.prepared.core is not None else None),
                          acl_reward=spec.reward, seed=spec.seed, wait_penalty=-spec.wait_cost,
                          no_buzz_reward=spec.abstain_reward, buzz_correct=spec.correct_reward,
                          buzz_incorrect=spec.wrong_reward, end_mode="no_buzz")
        wrapped = TextObservationWrapper(env, acl_spec=spec) if text else env
        return _RecordSteps(wrapped, sink) if sink is not None else wrapped

    def episodes(self, split, decide, *, text=False):
        questions = self.prepared.questions[split]
        env = self.environment(questions, text=text)
        results = []
        for i, q in enumerate(questions):
            obs, _ = env.reset(options={"question_idx": i})
            steps = []
            for index in range(len(q.cumulative_prefixes)):
                action = decide(q.qid, index, obs)
                if type(action) is not int or not 0 <= action <= self.prepared.spec.k:
                    raise ValueError("action must be an integer slot ID or zero")
                obs, reward, terminated, truncated, info = env.step(action)
                steps.append({"index": index, "action": action, "reward": reward,
                              "transition": info["acl_transition"]})
                if terminated or truncated:
                    results.append({"qid": q.qid, "group_id": self.prepared.groups[q.qid],
                                    "menu_id": self.prepared.menus[q.qid]["digest"],
                                    "correct": action == q.gold_index + 1,
                                    "abstained": action == 0 and terminated,
                                    "terminated": terminated, "truncated": truncated, "steps": steps})
                    break
            else:
                raise ValueError("episode exceeded source horizon")
        return results

    def score_actions(self, split: str, choices: dict):
        jobs = {(j["qid"], j["prefix_index"]): j["job_id"]
                for j in self.prepared.public_jobs(split)}
        if set(choices) != set(jobs.values()):
            raise ValueError("action coverage differs from exported public jobs")
        if any(type(a) is not int or not 0 <= a <= self.prepared.spec.k for a in choices.values()):
            raise ValueError("invalid action in submitted predictions")
        return self.episodes(split, lambda q, i, obs: choices[jobs[q, i]])

    def replay_training(self, recorded, policy, expected_units):
        """Validate a PPO step budget or a T5 complete-episode budget for one block."""
        if not isinstance(recorded, list) or not recorded:
            raise ValueError("empty training trace")
        if policy == "ppo" and len(recorded) != expected_units:
            raise ValueError("training trace differs from the PPO step budget")
        completed = 0
        questions = self.prepared.questions["train"]
        by_id = {q.qid: i for i, q in enumerate(questions)}
        env = None
        for row in recorded:
            i = by_id[row["qid"]]
            if env is None or env.terminated or env.truncated:
                env = self.environment([questions[i]] if policy == "t5" else questions)
                env.reset(options={"question_idx": 0 if policy == "t5" else i})
            if row["qid"] != env.question.qid:
                raise ValueError("unclosed training episode before question changed")
            _, reward, term, trunc, info = env.step(row["action"])
            completed += int(term or trunc)
            if row["transition"] != info["acl_transition"] or row["reward"] != reward:
                raise ValueError("training transition replay mismatch")
        if policy == "t5" and (completed != expected_units or not (env.terminated or env.truncated)):
            raise ValueError("training trace differs from the T5 complete-episode budget")


class _RecordSteps(gym.Wrapper):
    def __init__(self, env, sink):
        super().__init__(env)
        self.sink = sink

    def step(self, action):
        result = self.env.step(action)
        self.sink.append({"qid": self.unwrapped.question.qid, "action": int(action),
                          "reward": result[1], "transition": result[4]["acl_transition"]})
        return result


def load_inputs(config, menu, reward, seed, potential, core=None, *, review_draft=False):
    arm = dict(config["arm"])
    arm["reward"] = dict(arm["reward"])
    if menu is not None:
        arm["menu"] = menu
    if reward is not None:
        arm["reward"]["arm"] = reward
    if seed is not None:
        arm["seed"] = seed
    if potential is not None:
        arm["reward"]["potential_scale"] = potential
    spec = ArmSpec.from_dict(arm)
    inputs = config["inputs"]
    if "fixture" in inputs:
        if inputs["fixture"] != "solar-system-v1" or "dataset" in inputs:
            raise ValueError("ambiguous or unknown fixture")
        from scripts.acl_smoke_data import inputs as fixture
        dataset = fixture()
    else:
        if "dataset_file_sha256" in inputs and sha256_file(Path(inputs["dataset"])) != inputs["dataset_file_sha256"]:
            raise ValueError("source evaluator file differs from the requested ACL freeze")
        dataset = load_json_strict(inputs["dataset"])
    if "fixture" not in inputs and dataset.get("evidence_scope") not in (
            "scientific", "synthetic_fixture", "engineering_smoke", "legacy_engineering_control"):
        raise ValueError("external dataset must declare a recognized evidence_scope")
    review_keys = set(inputs).intersection({"review", "reviewed_subset", "class_review"})
    if len(review_keys) > 1:
        raise ValueError("supply one source review mechanism")
    if not review_draft and dataset.get("evidence_scope") == "scientific" and not review_keys:
        raise ValueError("scientific inputs require source-specific answer review")
    review = None
    if review_keys and not review_draft:
        review = load_json_strict(inputs[next(iter(review_keys))])
    aliases = load_json_strict(inputs["oracle"])
    reservoir = load_json_strict(inputs["reservoir"]) if inputs.get("reservoir") else None
    result = prepare(spec, dataset, aliases, reservoir=reservoir, split_roles=inputs.get("split_roles"), core=core,
                     encoder=sbert() if "sbert" in spec.menu else None, contextual_review="class_review" in inputs, review_draft=review_draft)
    if review_draft:
        return result
    if "review" in inputs:
        from qb_data.acl_oracle_review import verify_review
        result.identity["source_review"] = verify_review(result, dataset, review)
    elif "reviewed_subset" in inputs:
        from scripts.acl_reviewed_subset import bind_review
        result.identity["source_review"] = bind_review(result, dataset, review, aliases)
    elif "class_review" in inputs:
        from qb_data.acl_scientific_review import compile_review
        _, result.identity["source_review"] = compile_review(result, dataset, aliases, review)
    return result


@lru_cache(maxsize=1)
def runtime_identity():
    files = ["agents/ppo_buzzer.py", "agents/_math.py", "models/features.py", "models/likelihoods.py", "models/t5_policy.py",
             "qb_data/acl_menus.py", "qb_data/acl_protocol.py", "qb_data/mc_builder.py",
             "qb_data/acl_oracle_review.py", "qb_data/data_loader.py", "qb_data/text_utils.py",
             "qb_env/acl_rewards.py", "qb_env/text_wrapper.py", "qb_env/tossup_env.py",
             "training/train_ppo_t5.py", "training/ppo_math.py",
             "scripts/acl_expansion.py", "scripts/acl_checkpoints.py", "scripts/stopdff_v5/identity.py"]
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/"qb_env").glob("acl_*.py")
              if p.relative_to(ROOT).as_posix() not in files]
    return {"revision": os.environ.get("MODAL_HOST_GIT_COMMIT") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "code": {name: sha256_file(ROOT/name) for name in files},
            "python": platform.python_version(), "system": [platform.system(), platform.machine()],
            "packages": {name: version(name) for name in ("torch", "numpy", "scikit-learn", "gymnasium",
                         "stable-baselines3", "transformers", "sentence-transformers", "tokenizers",
                         "huggingface-hub", "filelock", "scipy", "safetensors")}}


@lru_cache(maxsize=2)
def model_assets(model):
    return {"model": model, "local_files": _hash_tree(Path(snapshot(model)))}


def _train(runtime, policy, settings, stage, previous, first_update, updates, final):
    from agents.ppo_buzzer import PPOBuzzer
    from models.t5_policy import T5PolicyModel
    from training.train_ppo_t5 import PPOTrainer

    p = runtime.prepared
    seed, gamma = (p.spec.seed + first_update) % 2**32, p.spec.reward.gamma
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    trace = []
    train = p.questions["train"]
    if policy == "ppo":
        env = runtime.environment(train, sink=trace)
        if previous is None:
            agent = PPOBuzzer(env, n_steps=settings.rollout_size, n_epochs=settings.epochs, device="cpu",
                              batch_size=settings.batch_size, seed=seed, gamma=gamma)
        else:
            agent = PPOBuzzer.load(previous/"policy.zip", env=env, device="cpu")
            agent.model.set_random_seed(seed)
        agent.train(updates * settings.rollout_size)
        agent.save(stage / "policy.zip")
    else:
        encoder = str(previous/"policy") if previous and not settings.freeze_encoder else snapshot(T5)
        model = T5PolicyModel({"model_name": encoder, "device": "cpu", "num_choices": p.spec.k,
                               "max_input_length": settings.max_input_length})
        _check_token_budget(p, model.tokenizer, settings.max_input_length)
        if settings.freeze_encoder:
            model.encoder.requires_grad_(False)
        trainer = PPOTrainer(model, train, p.questions["val"], {
            "ppo_gamma": gamma, "ppo_epochs_per_iter": settings.epochs, "ppo_batch_size": settings.batch_size,
            "max_input_length": settings.max_input_length, "checkpoint_dir": str(stage)},
            text_env_factory=lambda q: runtime.environment([q], text=True, sink=trace),
            bootstrap_time_limits=True, disable_update_dropout=True)
        if previous is not None:
            head = previous/"policy_head.pt" if settings.freeze_encoder else previous/"policy"/"policy_head.pt"
            model.policy_head.load_state_dict(torch.load(head, weights_only=True, map_location="cpu"))
            trainer.optimizer.load_state_dict(torch.load(previous/"optimizer.pt", weights_only=True, map_location="cpu"))
        for _ in range(updates):
            metrics = trainer.update_policy(trainer.collect_rollouts(settings.rollout_size))
            if not metrics["num_updates"] or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("invalid T5 update")
        if settings.freeze_encoder:
            torch.save(model.policy_head.state_dict(), stage/"policy_head.pt")
        else:
            model.save(str(stage/"policy"))
        if not final:
            torch.save(trainer.optimizer.state_dict(), stage/"optimizer.pt")
    publish_json(stage/"train_steps.json", trace)


def _check_token_budget(prepared, tokenizer, limit):
    """A completed run guarantees every displayed option and prefix fits in T5's input."""
    from qb_data.acl_protocol import policy_prompt
    for questions in prepared.questions.values():
        for q in questions:
            prompts = [policy_prompt(prefix, q.options, prepared.spec) for prefix in q.cumulative_prefixes]
            lengths = tokenizer(prompts, truncation=False, padding=False, return_length=True)["length"]
            if max(lengths) > limit:
                raise ValueError(f"{q.qid}: complete T5 prompt needs {max(lengths)} tokens, configured {limit}")


def _load_policy(runtime, policy, settings, checkpoint):
    if policy == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(str(checkpoint/"policy.zip"), device="cpu")
        return lambda q, i, obs: int(model.predict(obs, deterministic=True)[0].item())
    from models.t5_policy import T5PolicyModel
    model_name = snapshot(T5) if settings.freeze_encoder else str(checkpoint/"policy")
    model = T5PolicyModel({"model_name": model_name, "device": "cpu",
                           "num_choices": runtime.prepared.spec.k,
                           "max_input_length": settings.max_input_length})
    head = checkpoint/"policy_head.pt" if settings.freeze_encoder else checkpoint/"policy"/"policy_head.pt"
    model.policy_head.load_state_dict(torch.load(head, weights_only=True, map_location="cpu"))
    model.eval()

    @torch.inference_mode()
    def decide(qid, index, obs):
        action, _ = model.select_action(**model.encode_input([obs]), deterministic=True)
        return int(action.item())
    return decide


def run(prepared, out, policy, settings, *, replay=False):
    runtime = Runtime(prepared)
    torch.set_num_threads(settings.threads)
    assets = {}
    if policy == "t5":
        assets["encoder"] = model_assets(T5)
    if "sbert" in prepared.spec.menu:
        assets["retriever"] = model_assets(SBERT)
    if prepared.core is not None:
        import flint
        assets["interval_arithmetic"] = {"python_flint": flint.__version__, "flint": flint.__FLINT_VERSION__}
    source_review = prepared.identity.get("source_review", {})
    if "selection_sha256" in source_review:
        assets["subset_review_code_sha256"] = sha256_file(ROOT/"scripts/acl_reviewed_subset.py")
    if "class_review_sha256" in source_review:
        assets["class_review_code_sha256"] = sha256_file(ROOT/"qb_data/acl_scientific_review.py")
    identity = {"schema": "acl-run-v1", "input": prepared.identity, "runtime": runtime_identity(),
                "policy": policy, "training": settings.manifest(policy), "assets": assets, "device": "cpu"}
    run_id = digest(identity)
    folder = out / run_id
    if replay and not (folder/"completed.json").is_file():
        raise ValueError("replay requires an existing completed run")
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder/"run.lock")):
        publish_json(folder/"manifest.json", identity)
        publish_json(folder/"menus.json", prepared.menus)
        parent_digest, checkpoint = None, None
        for start in range(0, settings.updates, settings.checkpoint_updates):
            end = min(start+settings.checkpoint_updates, settings.updates)
            prefix = f"train-{end:06d}-"
            generations = sorted(folder.glob(f"{prefix}*/files.json"))
            if len(generations) > 1:
                raise ValueError("multiple completed checkpoints for one training block")
            header = {"run_id": run_id, "first_update": start, "end": end, "parent": parent_digest}
            if not generations:
                if replay or (folder/"completed.json").exists():
                    raise ValueError("missing training checkpoint for completed run")
                stage = Path(tempfile.mkdtemp(dir=folder, prefix=prefix))
                try:
                    _train(runtime, policy, settings, stage, checkpoint, start, end-start, end == settings.updates)
                    publish_json(stage/"files.json", {**header, "files": _hash_tree(stage)})
                except BaseException:
                    if not (stage/"files.json").exists():
                        shutil.rmtree(stage)
                    raise
                checkpoint = stage
            else:
                checkpoint = generations[0].parent
            manifest = verify_block(checkpoint, header)
            parent_digest = digest(manifest)
            if replay:
                runtime.replay_training(load_json_strict(checkpoint/"train_steps.json"), policy,
                                        (end - start) * settings.rollout_size)
        if (folder/"completed.json").is_file():
            receipt = load_json_strict(folder/"completed.json")
            expected = {"run_id": run_id, "checkpoint_sha256": digest(manifest),
                        "episodes": {s: sha256_file(folder/f"{s}_episodes.json") for s in ("val","test")}}
            if expected != receipt:
                raise ValueError("completion receipt mismatch")
            if not replay:
                return folder
        decide = _load_policy(runtime, policy, settings, checkpoint)
        for split in ("val", "test"):
            episodes = runtime.episodes(split, decide, text=policy == "t5")
            publish_json(folder/f"{split}_episodes.json", episodes)
        publish_json(folder/"completed.json", {"run_id": run_id, "checkpoint_sha256": digest(manifest),
                     "episodes": {s: sha256_file(folder/f"{s}_episodes.json") for s in ("val","test")}})
    return folder


def sweep(config, args, settings, core):
    """Each invocation runs one learner and seed, over one-factor smokes or a crossed ablation."""
    smoke = args.command == "smoke"
    potential = args.potential_scale if args.potential_scale is not None else .5
    receipts = []
    for menu in sorted(STRATEGIES):
        base = load_inputs(config, menu, "baseline", args.seed, 0., core)
        rewards = REWARDS if menu == "baseline" or not smoke else ("baseline",)
        for reward in rewards:
            scales = (0., potential) if not smoke or (menu, reward) == ("baseline", "baseline") else (0.,)
            for scale in dict.fromkeys(scales):
                spec = replace(base.spec, reward=replace(base.spec.reward, arm=reward, potential_scale=scale))
                p = replace(base, spec=spec, identity={**base.identity, "spec": asdict(spec)})
                start = monotonic()
                folder = run(p, args.out, args.policy, settings)
                if smoke:
                    run(p, args.out, args.policy, settings, replay=True)
                receipts.append({"menu": menu, "reward": reward, "potential_scale": scale,
                                 "run_id": folder.name, "elapsed_s": round(monotonic()-start, 3),
                                 "completion_sha256": sha256_file(folder/"completed.json")})
                print(menu, reward, scale, folder.name, flush=True)
    # Timings describe this invocation. Content-address the report to retain each observation.
    receipt = {"command": args.command, "policy": args.policy, "runs": receipts}
    publish_json(args.out/f"{args.command}-{args.policy}-{digest(receipt)}.json", receipt)


def _export_jobs(path, jobs):
    """Publish the canonical JSON array atomically while holding one prompt at a time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stage = Path(stream.name)
    try:
        with stage.open("wb") as output:
            output.write(b"[")
            for i, job in enumerate(jobs):
                if i:
                    output.write(b",")
                output.write(encode(job))
            output.write(b"]\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(stage, path)
        except FileExistsError:
            if sha256_file(stage) != sha256_file(path):
                raise ValueError(f"immutable artifact differs: {path.name}") from None
    finally:
        stage.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Matrix and smoke run one learner/seed slice per invocation.")
    parser.add_argument("command", choices=("run", "replay", "export", "score", "matrix", "smoke"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--menu", choices=sorted(STRATEGIES))
    parser.add_argument("--reward", choices=REWARDS)
    parser.add_argument("--potential-scale", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--policy", choices=("ppo", "t5"), default="ppo")
    parser.add_argument("--split", choices=("train","val","test"), default="test")
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--lean-core", action="store_true",
                        help="Decode draws and approve realized schedules using the built Lean executable")
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    args = parser.parse_intermixed_args(argv)
    if args.command in {"matrix", "smoke"} and (args.menu is not None or args.reward is not None):
        parser.error("matrix and smoke enumerate menus and rewards; use run for a selected arm")
    if args.actions is not None and args.command != "score":
        parser.error("--actions applies only to score")
    if any("=" not in token for token in args.overrides):
        parser.error("configuration overrides must be key=value")
    config = merge_overrides(yaml.safe_load(args.config.read_text(encoding="utf-8")), parse_overrides(args))
    settings = Training(**config.get("training", {}))
    torch.set_num_threads(settings.threads)
    context = nullcontext()
    if args.lean_core:
        from formal.acl_expansion.runtime import LeanRuntime
        context = LeanRuntime()
    with context as core:
        if args.command in {"matrix", "smoke"}:
            sweep(config, args, settings, core)
            return
        p = load_inputs(config, args.menu, args.reward, args.seed, args.potential_scale, core)
        if args.command in {"run", "replay"}:
            print(run(p, args.out, args.policy, settings, replay=args.command == "replay"))
        elif args.command == "export":
            _export_jobs(args.out, p.public_jobs(args.split))
        else:
            if args.actions is None:
                parser.error("score requires --actions (JSON job_id -> integer action)")
            choices = load_json_strict(args.actions)
            publish_json(args.out, {"input": p.identity, "split": args.split,
                         "action_sha256": digest(choices), "source_file_sha256": sha256_file(args.actions),
                         "runtime": runtime_identity(), "episodes": Runtime(p).score_actions(args.split, choices)})


if __name__ == "__main__":
    main()
