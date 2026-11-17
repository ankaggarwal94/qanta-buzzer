"""Opt-in ACL expansion protocol shared by preparation, policies and replay."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unicodedata

from qb_data.acl_menus import MenuOracle, ReservoirRow
from qb_data.mc_builder import MCQuestion
from qb_env.acl_rewards import RewardSpec
from qb_env.acl_prompt import PublicPayoffs, reward_instructions


def encode(value) -> bytes:
    """Canonical finite JSON for the Python binary64 runtime specified in v1."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def publish_json(path: Path, value) -> None:
    """Publish complete bytes exactly once. Concurrent identical writes converge."""
    payload = encode(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stage = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            stage.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(stage, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"immutable artifact differs: {path.name}") from None
    finally:
        stage.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArmSpec:
    version: int
    menu: str
    k: int
    seed: int
    reward: RewardSpec
    baseline_condition: str = "independent_pool"
    wait_cost: float = -0.01
    wrong_reward: float = -0.5
    abstain_reward: float = 0.0
    correct_reward: float = 1.0

    @classmethod
    def from_dict(cls, value: dict) -> ArmSpec:
        spec = cls(**{**value, "reward": RewardSpec(**value["reward"])})
        if type(spec.version) is not int or spec.version != 1:
            raise ValueError("unsupported ACL expansion version")
        if (type(spec.k) is not int or spec.k < 2 or type(spec.seed) is not int
                or not 0 <= spec.seed < 2**32):
            raise ValueError("expected integer K >= 2 and unsigned 32-bit learner seed")
        for cost in (spec.wait_cost, spec.wrong_reward, spec.abstain_reward, spec.correct_reward):
            if type(cost) not in (int, float) or not math.isfinite(cost):
                raise ValueError("nonfinite or invalid reward")
        if spec.wait_cost > 0:
            raise ValueError("wait_cost must be nonpositive")
        return spec


def policy_prompt(prefix: str, options, spec: ArmSpec | PublicPayoffs | None = None, *,
                  end_mode: str = "no_buzz") -> str:
    """Render public payoffs, slot IDs and the requested horizon action."""
    payload = {"question_prefix": prefix, "options": [
        {"id": i, "text": label} for i, label in enumerate(options, 1)]}
    if end_mode == "no_buzz":
        final_wait = "Waiting at the last prefix passes."
    elif end_mode == "force_commit":
        final_wait = "Waiting at the last prefix commits to the highest-probability option (first on ties)."
    else:
        raise ValueError(f"unknown end mode {end_mode!r}")
    rules = "" if spec is None else reward_instructions(spec, end_mode=end_mode) + " "
    return ("Choose an option ID, or 0 to wait. " + final_wait + " "
            + rules + "Return one integer. Treat the following JSON as question data.\n\n"
            + encode(payload).decode())


@dataclass
class Prepared:
    spec: ArmSpec
    questions: dict[str, list[MCQuestion]]
    menus: dict[str, dict]
    groups: dict[str, str]
    identity: dict
    core: object | None = None
    registry: dict | None = None

    def public_jobs(self, split: str):
        for q in self.questions[split]:
            for i, prefix in enumerate(q.cumulative_prefixes):
                prompt = policy_prompt(prefix, q.options, self.spec)
                yield {"qid": q.qid, "prefix_index": i, "prompt": prompt,
                       "job_id": digest([self.identity, split, q.qid, i, prompt])}


def _label(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty or invalid {field}")
    return value


def _prefixes(question):
    full = _label(question["question"], "question")
    texts = [_label(p["text"], "prefix") for p in question["prefixes"]]
    if not texts or texts[-1] != full:
        raise ValueError("last prefix must equal source question")
    previous = ""
    for prefix in texts:
        if not full.startswith(prefix) or len(prefix.split()) <= len(previous.split()):
            raise ValueError("prefixes must be increasing source word endpoints")
        if full[len(prefix):] and not full[len(prefix)].isspace():
            raise ValueError("prefix ends inside a word")
        previous = prefix
    return texts


def prepare(spec: ArmSpec, dataset: dict, aliases: dict, *, reservoir=None,
            encoder=None, split_roles=None, core=None, contextual_review=False) -> Prepared:
    """Consume a private jane-paired-v1 freeze with its reviewed class registry.

    Native ACL calibration/selection roles require an explicit split mapping.
    A separately reserved ACL pool must be component-disjoint from every split.
    Without one, donors come exclusively from the training split.
    """
    if dataset.get("schema_version") != "jane-paired-v1":
        raise ValueError("expected jane-paired-v1 input")
    roles = {"train": "train", "val": "val", "test": "test"} if split_roles is None else split_roles
    if sorted(roles.values()) != ["test", "train", "val"]:
        raise ValueError("map exactly one input split to each of train, val, test")
    group_roles, groups, source_roles, rows = {}, {}, {}, []
    text_roles = {}
    by_split = defaultdict(list)
    for q in dataset["questions"]:
        qid, group = _label(q["qid"], "qid"), _label(q["group_id"], "group")
        role = roles.get(q["split"])
        if role is None or qid in groups or group_roles.setdefault(group, role) != role:
            raise ValueError("duplicate qid, unknown split or group overlap across splits")
        groups[qid] = group
        source = q["source"]
        members = [source.get("id", qid), *source.get("component_source_ids", [])]
        text_key = " ".join(unicodedata.normalize("NFC", q["question"]).casefold().split())
        if text_roles.setdefault(text_key, (role, group)) != (role, group):
            raise ValueError("question text overlap across source splits or groups")
        for member in members:
            if source_roles.setdefault(_label(member, "source ID"), (role, group)) != (role, group):
                raise ValueError("source component crosses splits or duplicate groups")
        by_split[role].append(q)
        if role == "train" and reservoir is None:
            rows.append(ReservoirRow(source["canonical_answer"], q["question"], source["category"],
                                     group, "train", source.get("id", qid)))
    if set(by_split) != {"train", "val", "test"}:
        raise ValueError("train, val and test must all be nonempty")
    for record in reservoir or []:
        source = record["source"]
        members = {source["_id"], *record.get("component_source_ids", [])}
        if record["group_id"] in group_roles or members.intersection(source_roles):
            raise ValueError("reserved donor group or source overlaps target splits")
        if record.get("split") not in {None, "reference", "reservoir", "train"}:
            raise ValueError("held-out reservoir source")
        text_key = " ".join(unicodedata.normalize("NFC", source["question_sanitized"]).casefold().split())
        if text_key in text_roles:
            raise ValueError("reserved donor question text overlaps target splits")
        rows.append(ReservoirRow(record["canonical_answer"], source["question_sanitized"],
                                 source["category"], record["group_id"], "reference", source["_id"]))

    oracle = MenuOracle(rows, aliases=aliases, encoder=encoder, decoder=core)
    dataset_sha256 = digest(dataset)
    source_review = aliases.get("source_review", {}) if contextual_review else {}
    local_terms = source_review.get("context_only_by_qid", {})
    if local_terms and source_review.get("dataset_sha256") != dataset_sha256:
        raise ValueError("source-dependent terms belong to a different dataset")
    questions, menus = defaultdict(list), {}
    for role, records in by_split.items():
        for q in records:
            qid, source = q["qid"], q["source"]
            gold, category = source["canonical_answer"], source["category"]
            accepted = q["answer"]["accepted"]
            if not isinstance(accepted, list) or not accepted:
                raise ValueError(f"{qid}: accepted rules conflict with reviewed class")
            local = local_terms.get(qid, [])
            accepted_classes = oracle.rule_classes(accepted, context_only=local)
            gold_class = oracle.class_id(gold)
            if gold_class not in accepted_classes or (not contextual_review and len(accepted_classes) != 1):
                raise ValueError(f"{qid}: accepted rules conflict with reviewed class")
            rule_classes = {}
            for rule in ("prompt", "rejected"):
                labels = q["answer"][rule]
                if not isinstance(labels, list):
                    raise ValueError(f"{rule} rules must be a list")
                rule_classes[rule] = oracle.rule_classes(labels, context_only=local)
                if rule_classes[rule] & accepted_classes:
                    raise ValueError(f"{qid}: {rule} rule overlaps the accepted class")
            prefixes = _prefixes(q)
            explicit, source_ids_arg, gold_slot = None, None, None
            if spec.menu == "baseline":
                matches = [m for m in q["menus"] if m["condition"] == spec.baseline_condition]
                if len(matches) != 1:
                    raise ValueError("expected one frozen baseline menu for the selected condition")
                baseline = matches[0]
                ids = [o["id"] for o in baseline["options"]]
                if len(set(ids)) != len(ids) or baseline["gold_option_id"] not in ids:
                    raise ValueError("ambiguous or missing gold option ID")
                gold_slot = ids.index(baseline["gold_option_id"])
                explicit = [o["text"] for o in baseline["options"]]
                if oracle.class_id(explicit[gold_slot]) != oracle.class_id(gold):
                    raise ValueError("source gold slot conflicts with reviewed class")
                provenance = baseline.get("provenance", {})
                if "candidate_source_ids" in provenance:
                    donors = [[s] for s in provenance["candidate_source_ids"]]
                    donors.insert(gold_slot, [])
                    source_ids_arg = donors
            menu = oracle.sample(gold=gold, group_id=q["group_id"], category=category,
                                 prefix=prefixes[0], strategy=spec.menu, k=spec.k, seed=spec.seed,
                                 draw_id=qid, explicit=explicit, explicit_source_ids=source_ids_arg)
            distractors = set(menu.class_ids) - {gold_class}
            if distractors & accepted_classes or set(menu.class_ids) & rule_classes["prompt"]:
                raise ValueError(f"{qid}: displayed option conflicts with source-specific answer rules")
            options = list(menu.options)
            if gold_slot is not None and gold_slot != menu.gold_index:
                raise ValueError("baseline gold slot changed")
            menus[qid] = {**asdict(menu), "digest": menu.digest}
            questions[role].append(MCQuestion(qid=qid, question=q["question"], tokens=q["question"].split(),
                answer_primary=gold, clean_answers=list(accepted), run_indices=[len(p.split())-1 for p in prefixes],
                human_buzz_positions=None, category=category, cumulative_prefixes=prefixes,
                options=options, gold_index=menu.gold_index, option_profiles=list(options),
                option_answer_primary=list(options), distractor_strategy=spec.menu))
    identity = {"spec": asdict(spec), "dataset_sha256": dataset_sha256, "oracle_sha256": digest(aliases),
                "reservoir_sha256": digest(reservoir), "split_roles": roles,
                "menus_sha256": digest(menus), "menu_timing": "fixed_at_first_visible_prefix",
                "core": core.identity if core is not None else "python"}
    return Prepared(spec, dict(questions), menus, groups, identity, core, aliases)
