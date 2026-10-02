"""Freeze a new exploratory QBReader pilot without retrieving model outputs.

The API sampling is unseeded upstream. Seed 1 controls only deterministic
selection, splits, menus, and ordering within the retained HTTP snapshot. This
is not a recovery of Jane's original 200-question or QBReader-v3 experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qb_data.jane_paired import build_jobs, normalize_answer, validate_dataset

CATEGORIES = ("History", "Literature", "Science")
SEED = 1
PER_CATEGORY_FETCH = 600
PER_CATEGORY_RESERVOIR = 60
DEV_COUNT = 12
MAIN_SPLITS = ("calibration",) * 50 + ("selection",) * 50 + ("test",) * 100
TARGET_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
PUBLIC_FIELDS = {"job_id", "qid", "group_id", "split", "format", "condition",
                 "menu_id", "prefix_id", "fraction", "prompt", "prompt_sha256"}
CHOICE_FIELDS = {"job_id", "qid", "group_id", "split", "format", "condition",
                 "menu_id", "options", "prompt", "prompt_sha256"}


def canonical(value: object) -> str:
    """Serialize finite JSON canonically for deterministic identities."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def sha(value: bytes | str) -> str:
    """Return SHA256 over exact UTF-8 bytes when passed text."""
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write one create-once finite JSON artifact."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class _VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.unsafe = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.unsafe = True
        if tag == "br":
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def extract_canonical(raw: str) -> tuple[str | None, str]:
    """Accept only a short, underlined leading primary answer, without aliases.

    Parameters
    ----------
    raw : str
        Complete formatted source answerline, retained separately unchanged.

    Returns
    -------
    tuple
        Canonical leading text or None, and the explicit eligibility reason.
        This syntactic filter is not independent human answerline validation.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "missing full formatted answerline"
    if re.search(r"\b(before|after|until|prompt)\b", raw, re.I):
        return None, "temporal or prompt instruction excluded prospectively"
    head = raw.split("[", 1)[0].strip()
    if not re.search(r"<u(?:\s[^>]*)?>", head, re.I):
        return None, "primary answer has no underlined evidence"
    parser = _VisibleText()
    parser.feed(head)
    text = " ".join("".join(parser.parts).split()).strip()
    if parser.unsafe or not text or len(text.split()) > 12 or len(text) > 120:
        return None, "unsafe, empty, or long canonical primary answer"
    if re.search(r"[\[\]();/=:\n\r]|\bor\b|\band\b", text, re.I):
        return None, "compound or ambiguous primary-answer syntax excluded"
    if any(ord(char) < 32 for char in text):
        return None, "control character in canonical answer"
    return text, "strict leading-primary extraction; only this exact canonical auto-accepted"


def sentence_prefixes(question: str) -> list[dict]:
    """Return up to five increasing exact prefixes at sentence-boundary proxies.

    Parameters
    ----------
    question : str
        Original sanitized question body, unmodified.

    Returns
    -------
    list of dict
        Closest sentence ends to .2/.4/.6/.8/1, deduplicated, with actual
        whitespace-token counts and character ends. Regex sentence boundaries
        can split abbreviations and are not reviewed clue annotations.
    """
    tokens = list(re.finditer(r"\S+", question))
    if len(tokens) < 40:
        raise ValueError("Question requires at least 40 whitespace tokens")
    candidates = []
    for match in re.finditer(r"[.!?][\"'”’)]*(?=\s+|$)", question):
        end = match.end()
        count = sum(token.end() <= end for token in tokens)
        if 0 < count < len(tokens) and (end == len(question) or question[end].isspace()):
            candidates.append((count, end))
    candidates.append((len(tokens), len(question)))
    candidates = sorted(set(candidates))
    if len(candidates) < 2:
        raise ValueError("Question requires at least two sentence-boundary proxies")
    chosen = {}
    for target in TARGET_FRACTIONS:
        count, end = min(candidates, key=lambda pair: (abs(pair[0] / len(tokens) - target), pair[0]))
        chosen.setdefault((count, end), []).append(target)
    chosen[(len(tokens), len(question))] = [1.0]
    result = []
    for index, ((count, end), targets) in enumerate(sorted(chosen.items()), 1):
        result.append({"prefix_id": f"p{index}", "text": question[:end],
                       "fraction": count / len(tokens), "token_count": count,
                       "token_start_index": 0, "token_end_index_exclusive": count,
                       "character_end_exclusive": end, "nearest_requested_fractions": targets,
                       "boundary_status": "unreviewed regex sentence-end proxy"})
    return result


def _rank(domain: str, source_id: str) -> str:
    return sha(f"jane-new-gpu-pilot|seed={SEED}|{domain}|{source_id}")


def prospective_plan(excluded_ids: list[str], excluded_answers: list[str]) -> dict:
    """Return the complete data-selection plan frozen before API retrieval."""
    return {
        "schema_version": "jane-new-gpu-data-plan-v1", "seed": SEED,
        "evidence_scope": "engineering_smoke", "prompt_template": "concise_json_v3",
        "origin": "new exploratory pilot; not original Jane pilot reproduction",
        "categories": list(CATEGORIES), "number_per_category": PER_CATEGORY_FETCH,
        "api_parameters": {"standardOnly": "true", "minYear": 2010, "maxYear": 2025},
        "request_policy": "One request per category; no resampling conditioned on model performance.",
        "api_random_seed": None,
        "seed_scope": "local SHA256 ranking only; upstream random API cannot be seed-controlled",
        "exclude_prior_smoke_source_ids": sorted(excluded_ids),
        "exclude_prior_smoke_normalized_aliases": sorted(set(excluded_answers)),
        "eligibility": "nonempty full answerline and sanitized question; strict short underlined leading primary answer, excludes temporal/prompt/compound syntax; >=40 tokens and >=2 regex sentence-end proxies; unique ID, normalized question, and canonical-answer group",
        "reservoir": {"per_category": PER_CATEGORY_RESERVOIR,
                      "policy": "first seed-ranked eligible unique answers in each category; disjoint from all dev/main answers and questions"},
        "selection": "seed-ranked remaining eligible candidates, first 12 dev then next 200 main; no model outputs used",
        "dev": {"count": DEV_COUNT, "split_assignment": "4 calibration, 4 selection, 4 test for existing schema compatibility; all remain development-only"},
        "main": {"count": 200, "split_counts": {"calibration": 50, "selection": 50, "test": 100},
                 "split_assignment": "ranked blocks 0:50, 50:100, 100:200"},
        "prefix_targets": list(TARGET_FRACTIONS),
        "prefix_policy": "nearest actual regex sentence ends; earlier tie; deduplicate; include unchanged full question; retain actual fractions/token/character indices; unreviewed engineering clue proxies",
        "menus": ["independent_pool", "same_category_pool"],
        "menu_policy": "three seed-ranked canonical answers from independent reservoir; 4 options; gold positions cycle A/B/C/D within dev and main separately; same gold position across both menus; fixed across prefixes; no hardness or prefix compatibility validation",
        "oe_grading": "only exact canonical comparison under existing NFKC/case/whitespace normalization; all other nonempty valid responses need review against full answerline; no auto-parsed alternative aliases",
        "human_review": False,
    }


def fetch_source(out_dir: Path) -> None:
    """Retrieve source bytes once after writing the prospective data plan."""
    out_dir.mkdir(parents=True, exist_ok=False)
    source_dir = out_dir / "source"
    source_dir.mkdir()
    prior = json.loads((ROOT / "data/jane_smoke/answer_review.json").read_text())
    excluded_ids = [row["source_id"] for row in prior["questions"]]
    excluded_answers = [normalize_answer(alias) for row in prior["questions"] for alias in row["accepted"]]
    plan = prospective_plan(excluded_ids, excluded_answers)
    write_json(out_dir / "data_plan.json", plan)
    retrievals = []
    for category in CATEGORIES:
        url = ("https://www.qbreader.org/api/random-tossup?number=" + str(PER_CATEGORY_FETCH)
               + "&categories=" + category + "&standardOnly=true&minYear=2010&maxYear=2025")
        request = urllib.request.Request(url, headers={"User-Agent": "JaneExploratoryPilot/1.0"})
        started = datetime.now(timezone.utc).isoformat()
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
            record = {"category": category, "url": url, "started_utc": started,
                      "completed_utc": datetime.now(timezone.utc).isoformat(),
                      "http_status": response.status, "sha256": sha(body), "byte_count": len(body),
                      "headers": {key: value for key, value in response.headers.items()
                                  if key.lower() in {"date", "content-type", "etag", "last-modified"}},
                      "file": f"source/{category.lower()}_raw.json"}
        json.loads(body)
        (out_dir / record["file"]).write_bytes(body)
        retrievals.append(record)
        print(json.dumps({"retrieved_category": category, "bytes": len(body), "sha256": sha(body)}), flush=True)
    write_json(source_dir / "retrievals.json", {"schema_version": "jane-source-retrieval-v1",
               "data_plan_sha256": sha((out_dir / "data_plan.json").read_bytes()),
               "retrievals": retrievals})


def select_records(raw_responses: list[dict], plan: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Select disjoint reservoir/development/main records from a frozen snapshot."""
    eligible, inventory = [], []
    seen_ids, seen_answers, seen_questions = set(), set(), set()
    sources = [row for response in raw_responses for row in response["tossups"]]
    sources.sort(key=lambda row: (_rank("eligibility", str(row.get("_id", ""))), canonical(row)))
    for source in sources:
        source_id = str(source.get("_id", ""))
        question = source.get("question_sanitized", "")
        raw_answer = source.get("answer", "")
        answer, reason = extract_canonical(raw_answer)
        key = normalize_answer(answer) if answer else None
        qkey = normalize_answer(question) if isinstance(question, str) else None
        if source_id in plan["exclude_prior_smoke_source_ids"]:
            reason, answer = "prior engineering smoke source ID excluded", None
        elif key in plan["exclude_prior_smoke_normalized_aliases"]:
            reason, answer = "prior engineering smoke answer alias excluded", None
        elif source.get("category") not in CATEGORIES:
            reason, answer = "unexpected category", None
        elif not source_id or source_id in seen_ids:
            reason, answer = "missing or repeated source ID", None
        elif key is not None and key in seen_answers:
            reason, answer = "duplicate normalized canonical-answer group", None
        elif not isinstance(question, str) or not question.strip() or qkey in seen_questions:
            reason, answer = "missing or duplicate normalized question text", None
        elif answer:
            try:
                sentence_prefixes(question)
            except ValueError as exc:
                reason, answer = str(exc), None
        inventory.append({"source_id": source_id, "eligible": answer is not None, "reason": reason,
                          "category": source.get("category")})
        seen_ids.add(source_id)
        if answer is not None:
            seen_answers.add(key)
            seen_questions.add(qkey)
            eligible.append({"source": source, "canonical_answer": answer,
                             "extraction_reason": reason})
    eligible.sort(key=lambda record: _rank("selection", record["source"]["_id"]))
    reservoir = []
    for category in CATEGORIES:
        pool = [record for record in eligible if record["source"]["category"] == category]
        if len(pool) < PER_CATEGORY_RESERVOIR:
            raise ValueError(f"Frozen snapshot has only {len(pool)} eligible {category} answers; requires 60 reservoir")
        reservoir.extend(pool[:PER_CATEGORY_RESERVOIR])
    reservoir_ids = {record["source"]["_id"] for record in reservoir}
    remaining = [record for record in eligible if record["source"]["_id"] not in reservoir_ids]
    if len(remaining) < DEV_COUNT + 200:
        raise ValueError(f"Frozen snapshot has {len(remaining)} non-reservoir eligible answers; requires 212")
    return reservoir, remaining[:DEV_COUNT], remaining[DEV_COUNT:DEV_COUNT + 200], inventory


def build_dataset(records: list[dict], reservoir: list[dict], splits: tuple[str, ...],
                  provenance: dict, plan: dict, phase: str) -> dict:
    """Build an evaluator-only dataset and fixed reservoir menus."""
    questions = []
    for index, (record, split) in enumerate(zip(records, splits, strict=True)):
        source, answer = record["source"], record["canonical_answer"]
        menus = []
        for condition in ("independent_pool", "same_category_pool"):
            candidates = [candidate for candidate in reservoir
                          if condition == "independent_pool" or
                          candidate["source"]["category"] == source["category"]]
            candidates.sort(key=lambda candidate: _rank(
                f"menu|{condition}|{source['_id']}", candidate["source"]["_id"]))
            distractors = candidates[:3]
            options = [candidate["canonical_answer"] for candidate in distractors]
            options.insert(index % 4, answer)
            menus.append({"condition": condition, "menu_id": "fixed_1",
                          "options": [{"id": letter, "text": text} for letter, text in zip("ABCD", options)],
                          "gold_option_id": "ABCD"[index % 4],
                          "provenance": {"method": "seed-1 SHA256-ranked independent canonical-answer reservoir",
                                         "candidate_source_ids": [candidate["source"]["_id"] for candidate in distractors],
                                         "candidate_pool_disjoint_from_dev_main": True,
                                         "same_category_constraint": condition == "same_category_pool",
                                         "fixed_across_prefixes": True, "balanced_gold_position": "A/B/C/D cycle within phase",
                                         "hardness_validated": False, "pyramid_aligned": False,
                                         "review_status": "no human answer equivalence or clue-compatibility review"}})
        questions.append({"qid": "qbreader:" + source["_id"],
                          "group_id": "answer:" + sha(normalize_answer(answer))[:16],
                          "split": split, "question": source["question_sanitized"],
                          "prefixes": sentence_prefixes(source["question_sanitized"]),
                          "answer": {"raw": source["answer"], "accepted": [answer], "rejected": [], "prompt": []},
                          "menus": menus,
                          "source": {"id": source["_id"], "url": "https://www.qbreader.org/api/tossup?_id=" + source["_id"],
                                     "category": source["category"], "subcategory": source.get("subcategory"),
                                     "set": source.get("set"), "packet": source.get("packet"),
                                     "question_field": "question_sanitized", "answer_field": "answer",
                                     "question_utf8_sha256": sha(source["question_sanitized"]),
                                     "answerline_utf8_sha256": sha(source["answer"]),
                                     "canonical_answer": answer, "answer_extraction": record["extraction_reason"],
                                     "answer_review": "automated strict canonical extraction; no independent human validation"}})
    dataset = {"schema_version": "jane-paired-v1", "evidence_scope": "engineering_smoke",
               "prompt_template": "concise_json_v3",
               "source": {"origin": "QBReader current public random-tossup API; new exploratory pilot",
                          "provenance": provenance, "full_answerlines_available": True, "phase": phase,
                          "selection_policy": plan, "review_status": "no human review",
                          "scope_note": "Convenience/simple-answerline filtered three-category sample; unreviewed sentence-end clue proxies; not original Jane pilot or representative scientific evidence."},
               "questions": questions}
    validate_dataset(dataset)
    return dataset


def build_choice_controls(dataset: dict) -> dict:
    """Build only independent menu-control jobs using the GPU backend template."""
    from scripts.jane_gpu_backend import build_choice_control_prompt
    jobs = []
    for question in sorted(dataset["questions"], key=lambda question: question["qid"]):
        for menu in sorted(question["menus"], key=lambda menu: (menu["condition"], menu["menu_id"])):
            options = [{"id": option["id"], "text": option["text"]} for option in menu["options"]]
            prompt = build_choice_control_prompt(options)
            identity = {"qid": question["qid"], "group_id": question["group_id"],
                        "split": question["split"], "format": "mc", "condition": menu["condition"],
                        "menu_id": menu["menu_id"], "options": options, "prompt": prompt}
            jobs.append({"job_id": sha(canonical(identity)), **identity, "prompt_sha256": sha(prompt)})
    return {"schema_version": "jane-choice-controls-v1", "evidence_scope": "engineering_smoke", "jobs": jobs}


def assert_public(package: dict, *, choices: bool = False) -> None:
    """Fail if public records contain any evaluator field or hash mismatch."""
    expected_schema = "jane-choice-controls-v1" if choices else "jane-public-jobs-v1"
    if set(package) != {"schema_version", "evidence_scope", "jobs"} or package["schema_version"] != expected_schema:
        raise ValueError("Unexpected public envelope")
    fields = CHOICE_FIELDS if choices else PUBLIC_FIELDS
    seen = set()
    for job in package["jobs"]:
        if set(job) != fields or job["job_id"] in seen:
            raise ValueError("Evaluator metadata or duplicate ID in public job")
        seen.add(job["job_id"])
        if job["prompt_sha256"] != sha(job["prompt"]):
            raise ValueError("Public prompt hash mismatch")
        identity = {key: value for key, value in job.items() if key not in {"job_id", "prompt_sha256"}}
        if job["job_id"] != sha(canonical(identity)):
            raise ValueError("Public job identity hash mismatch")
        if choices:
            from scripts.jane_gpu_backend import build_choice_control_prompt
            if job["prompt"] != build_choice_control_prompt(job["options"]):
                raise ValueError("Choice prompt differs from exact public menu template")
        else:
            payload = json.loads(job["prompt"].split("\n", 1)[0])
            if set(payload) != ({"question_prefix", "options"} if job["format"] == "mc" else {"question_prefix"}):
                raise ValueError("Unexpected evaluator content in prompt payload")


def prepare(out_dir: Path) -> None:
    """Create public/evaluator frozen artifacts from exact retained responses."""
    plan = json.loads((out_dir / "data_plan.json").read_text())
    retrievals = json.loads((out_dir / "source/retrievals.json").read_text())
    if retrievals["data_plan_sha256"] != sha((out_dir / "data_plan.json").read_bytes()):
        raise ValueError("Data plan changed after retrieval")
    responses = []
    for record in retrievals["retrievals"]:
        body = (out_dir / record["file"]).read_bytes()
        if sha(body) != record["sha256"]:
            raise ValueError("Raw API source bytes changed")
        responses.append(json.loads(body))
    reservoir, dev, main, inventory = select_records(responses, plan)
    provenance = {"data_plan_sha256": retrievals["data_plan_sha256"], "retrievals": retrievals["retrievals"],
                  "preparer_code_sha256": sha(Path(__file__).read_bytes())}
    evaluator, public = out_dir / "evaluator", out_dir / "public"
    evaluator.mkdir(exist_ok=False)
    public.mkdir(exist_ok=False)
    datasets = {}
    summaries = {}
    for phase, records, splits in (("dev", dev, ("calibration",) * 4 + ("selection",) * 4 + ("test",) * 4),
                                    ("main", main, MAIN_SPLITS)):
        dataset = build_dataset(records, reservoir, splits, provenance, plan, phase)
        datasets[phase] = dataset
        write_json(evaluator / f"{phase}_dataset.json", dataset)
        trajectory = {"schema_version": "jane-public-jobs-v1", "evidence_scope": "engineering_smoke", "jobs": build_jobs(dataset)}
        controls = build_choice_controls(dataset)
        assert_public(trajectory)
        assert_public(controls, choices=True)
        write_json(public / f"{phase}_jobs.json", trajectory)
        write_json(public / f"{phase}_choices_only.json", controls)
        choice_gold = {job["job_id"]: next(menu["gold_option_id"] for question in dataset["questions"]
                       if question["qid"] == job["qid"] for menu in question["menus"]
                       if (menu["condition"], menu["menu_id"]) == (job["condition"], job["menu_id"]))
                       for job in controls["jobs"]}
        write_json(evaluator / f"{phase}_choices_only_gold.json", choice_gold)
        summaries[phase] = {"question_count": len(dataset["questions"]), "trajectory_job_count": len(trajectory["jobs"]),
                            "choices_only_job_count": len(controls["jobs"]),
                            "split_counts": dict(Counter(question["split"] for question in dataset["questions"])),
                            "category_counts": dict(Counter(question["source"]["category"] for question in dataset["questions"])),
                            "prefix_count_distribution": dict(Counter(len(question["prefixes"]) for question in dataset["questions"])),
                            "gold_position_counts_per_condition": dict(Counter(question["menus"][0]["gold_option_id"] for question in dataset["questions"]))}
    write_json(evaluator / "reservoir.json", reservoir)
    write_json(evaluator / "selection_inventory.json", {"records": inventory, "raw_record_count": sum(len(response["tossups"]) for response in responses),
               "eligible_count": sum(record["eligible"] for record in inventory), "selected_ids": {phase: [question["source"]["id"] for question in dataset["questions"]] for phase, dataset in datasets.items()}})
    files = {path.name: {"sha256": sha(path.read_bytes()), "byte_count": path.stat().st_size,
                        "job_count": len(json.loads(path.read_text())["jobs"])} for path in sorted(public.glob("*.json"))}
    write_json(public / "manifest.json", {"schema_version": "jane-public-input-manifest-v1", "files": files,
               "scope": "only these four public files may be sent to inference; evaluator and raw sources are private to local grading"})
    write_json(out_dir / "freeze_summary.json", {"schema_version": "jane-gpu-data-freeze-v1", "seed": SEED,
               "evidence_scope": "engineering_smoke", "human_review": False, "reservoir_count": len(reservoir),
               "phases": summaries, "public_manifest_sha256": sha((public / "manifest.json").read_bytes()),
               "limitations": ["new current-API snapshot, not original pilot", "upstream random retrieval unseeded",
                   "simple-answerline convenience filter and only three categories", "no human answerline/distractor review",
                   "regex sentence ends are unreviewed clue proxies", "noncanonical OE responses require review"]})
    print(json.dumps(summaries, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "modal_pilot_data")
    args = parser.parse_args()
    if args.action == "fetch":
        fetch_source(args.out_dir)
    else:
        prepare(args.out_dir)


if __name__ == "__main__":
    main()
