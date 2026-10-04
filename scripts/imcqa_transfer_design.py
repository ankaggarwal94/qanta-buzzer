"""Freeze the existing 7B policy and transfer it to unused protocol-development items.

Only public question/menu fields enter inference jobs. Answer labels remain in
the separately hash-bound evaluator file. Selection never inspects outcomes.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata
from typing import Any

from scripts import acl_option_scoring as base
from scripts import imcqa_protocol_design as old
from scripts import imcqa_wait_scoring as prior

PROTOCOL = "imcqa_frozen_transfer_fp32_v1"
SCHEMA = "imcqa-transfer-public-v1"
N_QUESTIONS = 100
SALT = "imcqa-frozen-transfer-20261004"
PREFIX_IDS, REWARDS, CONDITIONS = old.PREFIX_IDS, old.REWARDS, old.CONDITIONS
ARMS = ("plain", "wait")
PUBLIC_JOB_KEYS = old.PUBLIC_JOB_KEYS
SOURCE_SHA256 = {
    "main_jobs": prior.SOURCE_INPUT_SHA256,
    "main_dataset": "d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62",
    "prior_public": old.PRIOR_PUBLIC_SHA256,
    "fitted_parameters": "32b808b9ea8b39fe7251856234c7dfc044f55bbfd9e52c8626aa3581c11659a8",
}
CANONICAL_SOURCE_SHA256 = {
    "main_jobs": "4210a64f70c92945efb01f0a4871432c004d4b082b2067c43d908ea1b2cd8808",
    "main_dataset": "4cb30b6c17eff0967aadb09fadfa1bafec412e060c865d21177c9b01e85b9ae6",
    "prior_public": SOURCE_SHA256["prior_public"],
    "fitted_parameters": "f013fe1eb79b6dbcd074f0efe207d75317e798ae1fd65d6c75ae125604dccb1f",
}
POLICY_VALUES = {
    "independent_pool": (-0.703106162024301, 0.45313818462033534, 0.6, "fixed_1"),
    "same_category_pool": (-0.6355621259799796, 0.42493943060428674, 0.85, "fixed_2"),
}
PACKAGE_KEYS = {"schema_version", "protocol", "source_input_sha256", "main_dataset_sha256",
                "prior_public_sha256", "selection", "frozen_policy", "rewards", "wrong_reward",
                "pass_reward", "prefix_ids", "jobs"}
RULE = ("Rank original selection-split qids by SHA256(salt + '|' + qid), breaking ties by qid; "
        "exclude all prior WAIT qids and groups; greedily reject normalized full-question token "
        "5-gram Jaccard >= 0.8 versus every prior WAIT question or previously accepted question; take first 100.")
NORMALIZATION = "Unicode NFKC then casefold; Python Unicode regex [^\\W_]+ tokens; distinct contiguous token 5-grams; fewer than five tokens form one tuple; exact empty equality only."


def normalized_tokens(text: str) -> tuple[str, ...]:
    """Normalize the full question without using its answer or model outcomes."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("nonempty full question required")
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    tokens = normalized_tokens(text)
    return frozenset(tuple(tokens[i:i+5]) for i in range(max(1, len(tokens)-4)))


def near_duplicate(left: frozenset, right: frozenset) -> bool:
    """Use an integer comparison so the inclusive 0.8 boundary is exact."""
    union = len(left | right)
    return (not union) or 5 * len(left & right) >= 4 * union


def rank(qid: str) -> str:
    return base.sha(f"{SALT}|{qid}".encode())


def select_questions(dataset: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """Produce an auditable, outcome-blind deterministic selection manifest."""
    questions = dataset["questions"]
    lookup = {row["qid"]: row for row in questions}
    if len(lookup) != len(questions):
        raise ValueError("duplicate evaluator qids")
    selected_old = previous["selection"]["selected_qids"]
    if set(selected_old) != {"calibration", "selection"} or any(len(q) != 100 for q in selected_old.values()):
        raise ValueError("prior WAIT question set differs")
    excluded = sorted(qid for values in selected_old.values() for qid in values)
    if len(set(excluded)) != 200 or not set(excluded) <= set(lookup):
        raise ValueError("missing or duplicated prior WAIT question")
    excluded_groups = sorted({lookup[qid]["group_id"] for qid in excluded})
    references = [(qid, shingles(lookup[qid]["question"])) for qid in excluded]
    eligible = sorted((q["qid"] for q in questions if q["split"] == "selection"
                       and q["qid"] not in excluded and q["group_id"] not in excluded_groups),
                      key=lambda qid: (rank(qid), qid))
    accepted, skipped, accepted_groups = [], [], set()
    for qid in eligible:
        q = lookup[qid]
        if q["group_id"] in accepted_groups:
            skipped.append({"qid": qid, "reason": "accepted_group", "reference_qid": next(x for x in accepted if lookup[x]["group_id"] == q["group_id"])})
            continue
        grams = shingles(q["question"])
        collision = next((other for other, oldgrams in references if near_duplicate(grams, oldgrams)), None)
        if collision is not None:
            skipped.append({"qid": qid, "reason": "full_question_5gram_jaccard_ge_0.8", "reference_qid": collision})
            continue
        accepted.append(qid); accepted_groups.add(q["group_id"]); references.append((qid, grams))
        if len(accepted) == N_QUESTIONS:
            break
    if len(accepted) != N_QUESTIONS:
        raise ValueError("insufficient untouched and deduplicated selection questions")
    return {
        "schema_version": "imcqa-transfer-selection-v1", "salt": SALT, "rule": RULE,
        "normalization": NORMALIZATION, "jaccard_threshold": 0.8, "ngram_size": 5,
        "excluded_prior_qids": excluded, "excluded_prior_groups": excluded_groups,
        "eligible_ranked_qids": eligible, "selected_qids": accepted, "skipped": skipped,
        "selected_group_ids": {q: lookup[q]["group_id"] for q in accepted},
        "selected_normalized_text_sha256": {q: base.sha(base.canonical(normalized_tokens(lookup[q]["question"]))) for q in accepted},
        "selected_rank_sha256": {q: rank(q) for q in accepted},
        "outcomes_used_for_selection": False,
        "scope": "Unused by all previous protocol pilots; original corpus outcomes already analyzed, so not a pristine confirmatory test.",
    }


def frozen_policy(fitted: dict[str, Any]) -> dict[str, Any]:
    """Copy the exact previously fitted plain-MCQA policies without any refit."""
    parameters = []
    for condition in CONDITIONS:
        rows = [r for r in fitted["parameters"] if r["answer_source"] == "plain" and r["condition"] == condition]
        if len(rows) != 1:
            raise ValueError("missing unique frozen plain fit")
        row = rows[0]
        if tuple(row[k] for k in ("intercept", "slope", "selected_threshold", "selected_fixed_policy")) != POLICY_VALUES[condition]:
            raise ValueError("frozen coefficients or policies changed")
        keys = ("answer_source", "condition", "intercept", "slope", "feature_clip", "selected_threshold",
                "selected_fixed_policy", "fit_qids", "n_fit_questions", "fit_split", "method")
        parameters.append({k: row[k] for k in keys})
    return {"source_fitted_parameters_sha256": SOURCE_SHA256["fitted_parameters"],
            "parameters": parameters, "refitting_permitted": False,
            "threshold_rule": "First round with sigmoid(intercept + slope * logit(clipped maximum canonical candidate probability)) >= selected_threshold; otherwise PASS.",
            "candidate_tie_break": "Displayed A-D order, then restore canonical identity.",
            "analysis_scope": "Frozen-policy transfer; no risk guarantee or optimal-stopping claim."}


def _job(source: dict[str, Any], payload: dict[str, Any], arm: str, rotation: int) -> dict[str, Any]:
    job = old._real_job(source, payload, arm, rotation)
    job.update(score_id=source["source_job_id"] + f":transfer:factorial:{arm}:r{rotation}:wE",
               execution="new", source_score_id=None)
    return job


def build_public_package(main_package: dict[str, Any], dataset: dict[str, Any],
                         previous: dict[str, Any], fitted: dict[str, Any]) -> dict[str, Any]:
    """Bind all source objects, freeze selection/policy, and render public jobs."""
    for name, value in (("main_jobs", main_package), ("main_dataset", dataset), ("prior_public", previous), ("fitted_parameters", fitted)):
        if base.sha(base.canonical(value)) != CANONICAL_SOURCE_SHA256[name]:
            raise ValueError(f"hash-pinned {name} content differs")
    manifest = select_questions(dataset, previous)
    chosen = manifest["selected_qids"]
    qlookup = {q["qid"]: q for q in dataset["questions"]}
    lookup = {}
    for source in main_package["jobs"]:
        if source.get("format") != "mc" or source.get("qid") not in chosen or source.get("prefix_id") not in PREFIX_IDS or source.get("condition") not in CONDITIONS:
            continue
        key = (source["qid"], source["condition"], source["prefix_id"])
        if key in lookup or source["split"] != "selection" or source["group_id"] != qlookup[source["qid"]]["group_id"]:
            raise ValueError("source identity, grouping, or split mismatch")
        if base.sha(source["prompt"].encode()) != source["prompt_sha256"]:
            raise ValueError("original prompt checksum differs")
        payload = base.load_json(source["prompt"].rsplit("\n\n", 1)[1].encode())
        if set(payload) != {"question_prefix", "options"}:
            raise ValueError("original prompt payload differs")
        lookup[key] = (source, payload)
    jobs = []
    for qid in chosen:
        for condition in CONDITIONS:
            for round_number, prefix_id in enumerate(PREFIX_IDS, 1):
                if (qid, condition, prefix_id) not in lookup:
                    raise ValueError("missing original source cell")
                source, payload = lookup[qid, condition, prefix_id]
                adapted = {**source, "source_job_id": source["job_id"], "source_prompt_sha256": source["prompt_sha256"],
                           "round": round_number, "reward": REWARDS[round_number-1]}
                for rotation in range(4):
                    for arm in ARMS:
                        row = _job(adapted, payload, arm, rotation)
                        row["score_index"] = len(jobs)
                        jobs.append(row)
    package = {
        "schema_version": SCHEMA, "protocol": PROTOCOL,
        "source_input_sha256": SOURCE_SHA256["main_jobs"], "main_dataset_sha256": SOURCE_SHA256["main_dataset"],
        "prior_public_sha256": SOURCE_SHA256["prior_public"],
        "selection": {"salt": SALT, "rule": RULE, "selected_qids": {"selection": chosen},
                      "qid_category": {q: qlookup[q]["source"]["category"] for q in chosen},
                      "manifest": manifest, "manifest_sha256": base.sha(base.canonical(manifest))},
        "frozen_policy": frozen_policy(fitted), "rewards": list(REWARDS), "wrong_reward": -1.0,
        "pass_reward": 0.0, "prefix_ids": list(PREFIX_IDS), "jobs": jobs,
    }
    validate_public_package(package)
    return package


def validate_public_package(package: dict[str, Any]) -> list[dict[str, Any]]:
    """Reject leakage fields, altered policy, missing cells, and prompt mutations."""
    if not isinstance(package, dict) or set(package) != PACKAGE_KEYS:
        raise ValueError("unexpected public package fields")
    if (package["schema_version"] != SCHEMA or package["protocol"] != PROTOCOL
            or package["source_input_sha256"] != SOURCE_SHA256["main_jobs"]
            or package["main_dataset_sha256"] != SOURCE_SHA256["main_dataset"]
            or package["prior_public_sha256"] != SOURCE_SHA256["prior_public"]
            or package["rewards"] != list(REWARDS) or package["wrong_reward"] != -1.0
            or package["pass_reward"] != 0.0 or package["prefix_ids"] != list(PREFIX_IDS)):
        raise ValueError("public source or scientific design identity differs")
    selection = package["selection"]
    if set(selection) != {"salt", "rule", "selected_qids", "qid_category", "manifest", "manifest_sha256"} or selection["salt"] != SALT or selection["rule"] != RULE:
        raise ValueError("selection metadata differs")
    selected = selection["selected_qids"]
    if set(selected) != {"selection"}:
        raise ValueError("only original selection split allowed")
    chosen = selected["selection"]
    if len(chosen) != N_QUESTIONS or len(set(chosen)) != N_QUESTIONS or set(selection["qid_category"]) != set(chosen):
        raise ValueError("selection size or category mapping differs")
    manifest = selection["manifest"]
    manifest_keys = {"schema_version", "salt", "rule", "normalization", "jaccard_threshold", "ngram_size", "excluded_prior_qids", "excluded_prior_groups", "eligible_ranked_qids", "selected_qids", "skipped", "selected_group_ids", "selected_normalized_text_sha256", "selected_rank_sha256", "outcomes_used_for_selection", "scope"}
    if (set(manifest) != manifest_keys or base.sha(base.canonical(manifest)) != selection["manifest_sha256"]
            or manifest["selected_qids"] != chosen or manifest["salt"] != SALT or manifest["rule"] != RULE
            or manifest["normalization"] != NORMALIZATION or manifest["ngram_size"] != 5
            or manifest["jaccard_threshold"] != 0.8 or manifest["outcomes_used_for_selection"] is not False
            or len(set(manifest["excluded_prior_qids"])) != 200 or set(chosen) & set(manifest["excluded_prior_qids"])):
        raise ValueError("selection manifest, deduplication, or exclusion differs")
    eligible = manifest["eligible_ranked_qids"]
    skipped = manifest["skipped"]
    skipped_ids = {r["qid"] for r in skipped}
    if (len(eligible) != len(set(eligible)) or eligible != sorted(eligible, key=lambda q: (rank(q), q))
            or set(eligible) & set(manifest["excluded_prior_qids"])
            or chosen != [q for q in eligible if q not in skipped_ids][:N_QUESTIONS]
            or len(skipped_ids) != len(skipped) or not skipped_ids <= set(eligible)
            or any(set(r) != {"qid", "reason", "reference_qid"} or r["reason"] not in {"accepted_group", "full_question_5gram_jaccard_ge_0.8"} for r in skipped)):
        raise ValueError("deterministic hash ranking differs")
    for key in ("selected_group_ids", "selected_normalized_text_sha256", "selected_rank_sha256"):
        if set(manifest[key]) != set(chosen):
            raise ValueError("selected identity metadata differs")
    if (len(set(manifest["selected_group_ids"].values())) != N_QUESTIONS
            or set(manifest["selected_group_ids"].values()) & set(manifest["excluded_prior_groups"])
            or manifest["selected_rank_sha256"] != {q: rank(q) for q in chosen}):
        raise ValueError("selected groups or rank checksums differ")
    policy = package["frozen_policy"]
    if set(policy) != {"source_fitted_parameters_sha256", "parameters", "refitting_permitted", "threshold_rule", "candidate_tie_break", "analysis_scope"}:
        raise ValueError("unexpected frozen policy fields")
    if policy["source_fitted_parameters_sha256"] != SOURCE_SHA256["fitted_parameters"] or policy["refitting_permitted"] is not False or len(policy["parameters"]) != 2:
        raise ValueError("frozen policy provenance differs")
    for condition, fit in zip(CONDITIONS, policy["parameters"]):
        if (set(fit) != {"answer_source", "condition", "intercept", "slope", "feature_clip", "selected_threshold", "selected_fixed_policy", "fit_qids", "n_fit_questions", "fit_split", "method"}
                or fit["condition"] != condition or fit["answer_source"] != "plain"
                or tuple(fit[k] for k in ("intercept", "slope", "selected_threshold", "selected_fixed_policy")) != POLICY_VALUES[condition]
                or fit["feature_clip"] != [1e-6, .999999] or fit["fit_split"] != "calibration"
                or fit["method"] != "monotone_regularized_logistic_correctness"
                or fit["n_fit_questions"] != 20 or len(set(fit["fit_qids"])) != 20
                or not set(fit["fit_qids"]) <= set(manifest["excluded_prior_qids"]) or set(fit["fit_qids"]) & set(chosen)):
            raise ValueError("frozen fit or question separation differs")
    rows = package["jobs"]
    if not isinstance(rows, list) or len(rows) != N_QUESTIONS * 80:
        raise ValueError("public context count differs")
    expected = [(q, c, p, r, a) for q in chosen for c in CONDITIONS for p in PREFIX_IDS for r in range(4) for a in ARMS]
    canonical_menus, prefixes, sources = {}, {}, {}
    identities = set()
    for index, (job, key) in enumerate(zip(rows, expected)):
        if not isinstance(job, dict) or set(job) != PUBLIC_JOB_KEYS or job["score_index"] != index:
            raise ValueError("public job keys or index differ")
        actual = (job["qid"], job["condition"], job["prefix_id"], job["rotation"], job["arm"])
        if actual != key or job["score_id"] in identities:
            raise ValueError("missing, duplicate, reordered, or unexpected job identity")
        identities.add(job["score_id"])
        if (job["split"] != "selection" or job["group_id"] != manifest["selected_group_ids"][job["qid"]]
                or job["block"] != "factorial" or job["wait_label"] != "E" or job["execution"] != "new"
                or job["source_score_id"] is not None or job["synthetic_case"] is not None
                or type(job["round"]) is not int or job["round"] != PREFIX_IDS.index(job["prefix_id"])+1
                or job["reward"] != REWARDS[job["round"]-1] or type(job["rotation"]) is not int
                or isinstance(job["fraction"], bool) or not isinstance(job["fraction"], (int, float))
                or not math.isfinite(job["fraction"]) or not 0 < job["fraction"] <= 1
                or job["option_source_ids"] != old.mapping(job["rotation"])
                or job["allowed_actions"] != ("ABCD" if job["arm"] == "plain" else "ABCDE")
                or not re.fullmatch(r"[0-9a-f]{64}", job["source_prompt_sha256"])
                or base.sha(job["prompt"].encode()) != job["prompt_sha256"]):
            raise ValueError("job source, actions, or scoring semantics differ")
        payload = old._payload(job["prompt"])
        canonical_options = {job["option_source_ids"][o["id"]]: o["text"] for o in payload["options"]}
        menu_key, prefix_key = (job["qid"], job["condition"]), (job["qid"], job["prefix_id"])
        if canonical_menus.setdefault(menu_key, canonical_options) != canonical_options or prefixes.setdefault(prefix_key, payload["question_prefix"]) != payload["question_prefix"]:
            raise ValueError("matched menu or prefix changes")
        source_key = (job["qid"], job["condition"], job["prefix_id"])
        identity = tuple(job[k] for k in ("source_job_id", "source_prompt_sha256", "group_id", "menu_id", "fraction"))
        if sources.setdefault(source_key, identity) != identity:
            raise ValueError("original source metadata changes")
        plain_payload = {"question_prefix": payload["question_prefix"], "options": [{"id": k, "text": canonical_options[k]} for k in "ABCD"]}
        rebuilt = _job(job, plain_payload, job["arm"], job["rotation"]); rebuilt["score_index"] = index
        if rebuilt != job:
            raise ValueError("public record differs from canonical rerender")
    for qid in chosen:
        previous_text = ""
        for prefix in PREFIX_IDS:
            current = prefixes[qid, prefix]
            if not current.startswith(previous_text):
                raise ValueError("non-cumulative reveal sequence")
            previous_text = current
        if base.sha(base.canonical(normalized_tokens(previous_text))) != manifest["selected_normalized_text_sha256"][qid]:
            raise ValueError("full-question normalized checksum differs")
    return rows
