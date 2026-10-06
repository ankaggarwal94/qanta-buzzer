"""Public-only, plain-MCQA jobs for a locked policy's fresh-question test.

Gold labels stay in the separate evaluator dataset. The model sees precisely
the existing plain prompt with four rotations, two menus and five prefixes.
"""
from __future__ import annotations

import math
import re
import unicodedata

from scripts import acl_option_scoring as base
from scripts import imcqa_protocol_design as old

PROTOCOL = "imcqa-tuned-fresh-v1"
SCHEMA = "imcqa-tuned-public-v1"
SALT = "imcqa-tuned-fresh-20261006"
PREFIX_IDS, REWARDS, CONDITIONS = old.PREFIX_IDS, old.REWARDS, old.CONDITIONS
PUBLIC_JOB_KEYS = old.PUBLIC_JOB_KEYS
MODEL = {"tag": "qwen7b", "name": "Qwen/Qwen2.5-7B-Instruct",
         "revision": "a09a35458c702b33eeacc393d103063234e8bc28"}
PACKAGE_KEYS = {"schema_version", "protocol", "n_questions", "model", "source_input_sha256",
                "main_dataset_sha256", "policy_lock_sha256", "sample_size_planning_sha256",
                "selection", "selection_id", "rewards", "wrong_reward", "pass_reward", "prefix_ids", "jobs"}
SELECTION_KEYS = {"salt", "manifest_sha256", "selected_qids", "selected_group_ids",
                  "selected_normalized_text_sha256", "qid_category", "outcomes_used_for_selection"}


def normalized_tokens(text):
    """Return Unicode-normalized word tokens for outcome-blind text matching."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty question text required")
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(text):
    tokens = normalized_tokens(text)
    return frozenset(tokens[i:i + 5] for i in range(max(1, len(tokens) - 4)))


def near_duplicate(left, right):
    """Apply inclusive 0.8 Jaccard with an exact integer comparison."""
    union = len(left | right)
    return not union or 5 * len(left & right) >= 4 * union


def rank(qid):
    return base.sha(f"{SALT}|{qid}".encode())


def _job(source, payload, rotation):
    row = old._real_job(source, payload, "plain", rotation)
    row.update(score_id=source["source_job_id"] + f":tuned:plain:r{rotation}",
               execution="new", source_score_id=None)
    return row


def build_public_package(source_jobs, selection, *, source_input_sha256,
                         main_dataset_sha256, policy_lock_sha256, sample_size_planning_sha256):
    """Render a complete public grid from public source jobs and bound metadata.

    Parameters
    ----------
    source_jobs : list of dict
        Canonical public jobs emitted by ``qb_data.jane_paired.build_jobs``.
    selection : dict
        Only the allowlisted public selection fields, with no answer labels.

    Returns
    -------
    dict
        Validated public-only inference package.
    """
    chosen = selection["selected_qids"]
    lookup = {}
    for source in source_jobs:
        if (source.get("format") != "mc" or source.get("qid") not in chosen
                or source.get("prefix_id") not in PREFIX_IDS or source.get("condition") not in CONDITIONS):
            continue
        key = source["qid"], source["condition"], source["prefix_id"]
        if key in lookup:
            raise ValueError("duplicate source cell")
        if base.sha(source["prompt"].encode()) != source["prompt_sha256"]:
            raise ValueError("source prompt checksum differs")
        payload = base.load_json(source["prompt"].rsplit("\n\n", 1)[1].encode())
        if set(payload) != {"question_prefix", "options"}:
            raise ValueError("source prompt contains nonpublic fields")
        lookup[key] = source, payload
    jobs = []
    for qid in chosen:
        for condition in CONDITIONS:
            for rnd, prefix in enumerate(PREFIX_IDS, 1):
                if (qid, condition, prefix) not in lookup:
                    raise ValueError("missing source cell")
                source, payload = lookup[qid, condition, prefix]
                source = {**source, "source_job_id": source["job_id"],
                          "source_prompt_sha256": source["prompt_sha256"], "round": rnd, "reward": REWARDS[rnd - 1]}
                for rotation in range(4):
                    row = _job(source, payload, rotation)
                    row["score_index"] = len(jobs)
                    jobs.append(row)
    package = {"schema_version": SCHEMA, "protocol": PROTOCOL, "n_questions": len(chosen),
               "model": dict(MODEL), "source_input_sha256": source_input_sha256,
               "main_dataset_sha256": main_dataset_sha256, "policy_lock_sha256": policy_lock_sha256,
               "sample_size_planning_sha256": sample_size_planning_sha256,
               "selection": selection, "selection_id": base.sha(base.canonical(selection)),
               "rewards": list(REWARDS), "wrong_reward": -1.0, "pass_reward": 0.0,
               "prefix_ids": list(PREFIX_IDS), "jobs": jobs}
    validate_public_package(package)
    return package


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_public_package(package):
    """Reject leakage fields, altered prompts and incomplete or unbound grids."""
    if not isinstance(package, dict) or set(package) != PACKAGE_KEYS:
        raise ValueError("unexpected public package fields")
    n = package["n_questions"]
    if (package["schema_version"] != SCHEMA or package["protocol"] != PROTOCOL
            or type(n) is not int or not 4 <= n <= 5000 or package["model"] != MODEL
            or package["rewards"] != list(REWARDS) or package["wrong_reward"] != -1.
            or package["pass_reward"] != 0. or package["prefix_ids"] != list(PREFIX_IDS)
            or any(not _hash(package[k]) for k in ("source_input_sha256", "main_dataset_sha256",
                                                   "policy_lock_sha256", "sample_size_planning_sha256", "selection_id"))):
        raise ValueError("public scientific identity differs")
    selection = package["selection"]
    if (not isinstance(selection, dict) or set(selection) != SELECTION_KEYS
            or selection["salt"] != SALT or not _hash(selection["manifest_sha256"])
            or selection["outcomes_used_for_selection"] is not False
            or base.sha(base.canonical(selection)) != package["selection_id"]):
        raise ValueError("selection identity differs")
    chosen = selection["selected_qids"]
    if (not isinstance(chosen, list) or len(chosen) != n or len(set(chosen)) != n
            or any(not isinstance(q, str) or not q for q in chosen)):
        raise ValueError("question selection differs")
    for key in ("selected_group_ids", "selected_normalized_text_sha256", "qid_category"):
        if not isinstance(selection[key], dict) or set(selection[key]) != set(chosen):
            raise ValueError("selected identity mapping differs")
    if (len(set(selection["selected_group_ids"].values())) != n
            or any(not _hash(v) for v in selection["selected_normalized_text_sha256"].values())
            or any(not isinstance(v, str) or not v for k in ("selected_group_ids", "qid_category") for v in selection[k].values())):
        raise ValueError("selected group or text identity differs")
    rows = package["jobs"]
    if not isinstance(rows, list) or len(rows) != n * 40:
        raise ValueError("public context count differs")
    expected = [(q, c, p, r) for q in chosen for c in CONDITIONS for p in PREFIX_IDS for r in range(4)]
    menus, prefixes, sources, ids = {}, {}, {}, set()
    for index, (job, key) in enumerate(zip(rows, expected)):
        if not isinstance(job, dict) or set(job) != PUBLIC_JOB_KEYS or job["score_index"] != index:
            raise ValueError("public job keys or index differ")
        if (job["qid"], job["condition"], job["prefix_id"], job["rotation"]) != key or job["score_id"] in ids:
            raise ValueError("job coverage or order differs")
        ids.add(job["score_id"])
        if (job["split"] != "test" or job["group_id"] != selection["selected_group_ids"][job["qid"]]
                or job["arm"] != "plain" or job["block"] != "factorial" or job["wait_label"] != "E"
                or job["execution"] != "new" or job["source_score_id"] is not None or job["synthetic_case"] is not None
                or type(job["round"]) is not int or job["round"] != PREFIX_IDS.index(job["prefix_id"]) + 1
                or job["reward"] != REWARDS[job["round"] - 1] or type(job["rotation"]) is not int
                or isinstance(job["fraction"], bool) or not isinstance(job["fraction"], (int, float))
                or not math.isfinite(job["fraction"]) or not 0 < job["fraction"] <= 1
                or job["option_source_ids"] != old.mapping(job["rotation"]) or job["allowed_actions"] != "ABCD"
                or not _hash(job["source_prompt_sha256"]) or not _hash(job["source_job_id"])
                or base.sha(job["prompt"].encode()) != job["prompt_sha256"]):
            raise ValueError("job source or scoring semantics differ")
        payload = old._payload(job["prompt"])
        old._validate_displayed_options(payload["options"], "E")
        options = {job["option_source_ids"][o["id"]]: o["text"] for o in payload["options"]}
        mk, pk = (job["qid"], job["condition"]), (job["qid"], job["prefix_id"])
        if menus.setdefault(mk, options) != options or prefixes.setdefault(pk, payload["question_prefix"]) != payload["question_prefix"]:
            raise ValueError("matched options or prefix changed")
        sk = job["qid"], job["condition"], job["prefix_id"]
        identity = tuple(job[k] for k in ("source_job_id", "source_prompt_sha256", "group_id", "menu_id", "fraction"))
        if sources.setdefault(sk, identity) != identity:
            raise ValueError("source identity changed across rotations")
        original_payload = {"question_prefix": payload["question_prefix"], "options": [{"id": k, "text": options[k]} for k in "ABCD"]}
        rebuilt = _job(job, original_payload, job["rotation"])
        rebuilt["score_index"] = index
        if rebuilt != job:
            raise ValueError("job differs from canonical rerender")
    for qid in chosen:
        previous = ""
        for prefix in PREFIX_IDS:
            current = prefixes[qid, prefix]
            if not current.startswith(previous):
                raise ValueError("non-cumulative prefixes")
            previous = current
        if base.sha(base.canonical(normalized_tokens(previous))) != selection["selected_normalized_text_sha256"][qid]:
            raise ValueError("full-question checksum differs")
    return rows
