"""Identity-bound datasets, public prompts, and conservative paired grading.

This module intentionally uses only the Python standard library.  Public jobs
contain neither answer rules nor unseen question suffixes.  ``answer.raw`` is
retained as evidence; it is never parsed into permissive acceptance rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


DATASET_SCHEMA = "jane-paired-v1"
TRACE_SCHEMA = "jane-traces-v1"
EVIDENCE_SCOPES = {
    "synthetic_fixture", "legacy_engineering_control", "engineering_smoke", "scientific"
}
SPLITS = {"calibration", "selection", "test"}
PROMPT_TEMPLATES = {"verbose_json_v1", "concise_json_v2", "concise_json_v3", "concise_json_v4"}
_RULES = ("accepted", "rejected", "prompt")


def _fail(message: str) -> None:
    raise ValueError(message)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be a nonempty string")
    if any(unicodedata.category(char) == "Cc" for char in value):
        _fail(f"{label} must not contain control characters")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{label} must be nonempty text")
    return value


def _array(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        _fail(f"{label} must be {'a nonempty' if nonempty else 'an'} array")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail(f"{label} must be a finite number")
    if not math.isfinite(number):
        _fail(f"{label} must be a finite number")
    return number


def _canonical(value: Any, label: str = "value") -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must contain finite JSON values: {exc}") from exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answer(answer: str) -> str:
    """Normalize Unicode, case, and whitespace only; do not fuzzy-match.

    Parameters
    ----------
    answer : str
        An answer or a reviewed exact alias.

    Returns
    -------
    str
        A conservative comparison key. Punctuation, articles, and accents stay.
    """
    if not isinstance(answer, str):
        _fail("answer normalization requires a string")
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def _validate_review(menu: Mapping[str, Any], prefixes: list[Any],
                     option_ids: set[str], label: str) -> None:
    review = _object(menu["provenance"].get("compatibility_review"),
                     f"{label}.provenance.compatibility_review")
    _text(review.get("reviewer"), f"{label} compatibility reviewer")
    records = _array(review.get("records"), f"{label} compatibility records",
                     nonempty=True)
    expected = {(prefix["prefix_id"], option_id) for prefix in prefixes
                for option_id in option_ids if option_id != menu["gold_option_id"]}
    seen = set()
    for record in records:
        record = _object(record, f"{label} compatibility record")
        key = (_string(record.get("prefix_id"), "review prefix_id"),
               _string(record.get("option_id"), "review option_id"))
        if key not in expected or key in seen:
            _fail(f"{label} has an unknown or duplicate compatibility pair {key}")
        seen.add(key)
        if not isinstance(record.get("compatible"), bool):
            _fail(f"{label} compatibility must be explicitly boolean")
        _text(record.get("rationale"), f"{label} compatibility rationale")
    if seen != expected:
        _fail(f"{label} needs reviewed evidence for every distractor and prefix")


def validate_dataset(dataset: Mapping[str, Any]) -> None:
    """Validate a complete paired dataset or raise ``ValueError``.

    Parameters
    ----------
    dataset : mapping
        A ``jane-paired-v1`` JSON dataset with all three development/test splits.

    Returns
    -------
    None
        Successful validation makes no claim that a declaration of provenance
        is independently verified. Scientific scope additionally requires an
        explicit ``source.full_answerlines_available = true`` declaration.
    """
    dataset = _object(dataset, "dataset")
    _canonical(dataset, "dataset")
    if dataset.get("schema_version") != DATASET_SCHEMA:
        _fail(f"dataset schema_version must be {DATASET_SCHEMA}")
    scope = dataset.get("evidence_scope")
    if not isinstance(scope, str) or scope not in EVIDENCE_SCOPES:
        _fail("unknown dataset evidence_scope")
    prompt_template = dataset.get("prompt_template", "verbose_json_v1")
    if not isinstance(prompt_template, str) or prompt_template not in PROMPT_TEMPLATES:
        _fail("unsupported prompt_template")
    source = _object(dataset.get("source"), "source")
    _text(source.get("origin"), "source.origin")
    provenance = source.get("provenance")
    if not ((isinstance(provenance, str) and provenance.strip()) or
            (isinstance(provenance, Mapping) and provenance)):
        _fail("source.provenance must be nonempty text or an object")
    if scope == "scientific" and source.get("full_answerlines_available") is not True:
        _fail("scientific scope requires full_answerlines_available=true; "
              "displayed-answer controls must use legacy_engineering_control")
    questions = _array(dataset.get("questions"), "questions", nonempty=True)
    qids: set[str] = set()
    group_splits: dict[str, str] = {}
    question_splits: dict[str, str] = {}
    seen_splits: set[str] = set()
    reference_design: set[tuple[str, str]] | None = None
    for qi, question in enumerate(questions):
        label = f"questions[{qi}]"
        question = _object(question, label)
        qid = _string(question.get("qid"), f"{label}.qid")
        if qid in qids:
            _fail(f"duplicate qid {qid!r}")
        qids.add(qid)
        group = _string(question.get("group_id"), f"{label}.group_id")
        split = question.get("split")
        if not isinstance(split, str) or split not in SPLITS:
            _fail(f"{label}.split must be calibration, selection, or test")
        if group in group_splits and group_splits[group] != split:
            _fail(f"group {group!r} crosses splits")
        group_splits[group] = split
        seen_splits.add(split)
        full_text = _text(question.get("question"), f"{label}.question")
        duplicate_key = normalize_answer(full_text)
        if duplicate_key in question_splits and question_splits[duplicate_key] != split:
            _fail(f"duplicate question text crosses splits at {qid!r}")
        question_splits[duplicate_key] = split
        tokens = list(re.finditer(r"\S+", full_text))
        prefixes = _array(question.get("prefixes"), f"{label}.prefixes", nonempty=True)
        prefix_ids = set()
        previous_length = previous_tokens = 0
        for pi, prefix in enumerate(prefixes):
            plabel = f"{label}.prefixes[{pi}]"
            prefix = _object(prefix, plabel)
            prefix_id = _string(prefix.get("prefix_id"), f"{plabel}.prefix_id")
            if prefix_id in prefix_ids:
                _fail(f"duplicate prefix_id in {qid!r}")
            prefix_ids.add(prefix_id)
            prefix_text = _text(prefix.get("text"), f"{plabel}.text")
            if not full_text.startswith(prefix_text):
                _fail(f"{plabel} must be an exact leading substring")
            end = len(prefix_text)
            if (end < len(full_text) and not prefix_text[-1].isspace()
                    and not full_text[end].isspace()):
                _fail(f"{plabel} ends inside a token")
            n_tokens = len(prefix_text.split())
            if end <= previous_length or n_tokens <= previous_tokens:
                _fail(f"{plabel} must strictly grow by at least one token")
            fraction = _finite_number(prefix.get("fraction"), f"{plabel}.fraction")
            if not (0 < fraction <= 1) or not math.isclose(
                    fraction, n_tokens / len(tokens), rel_tol=0, abs_tol=1e-12):
                _fail(f"{plabel}.fraction must equal its actual token fraction")
            previous_length, previous_tokens = end, n_tokens
        if prefixes[-1]["text"] != full_text:
            _fail(f"{label} final prefix must equal the original full question")

        answer = _object(question.get("answer"), f"{label}.answer")
        _text(answer.get("raw"), f"{label}.answer.raw")
        rule_sets: dict[str, set[str]] = {}
        for rule in _RULES:
            aliases = _array(answer.get(rule), f"{label}.answer.{rule}",
                             nonempty=rule == "accepted")
            normalized = [normalize_answer(_text(alias, f"{label}.{rule} alias"))
                          for alias in aliases]
            if len(set(normalized)) != len(normalized):
                _fail(f"{label} contains duplicate normalized {rule} aliases")
            rule_sets[rule] = set(normalized)
        if any(rule_sets[left] & rule_sets[right]
               for left, right in (("accepted", "rejected"), ("accepted", "prompt"),
                                   ("rejected", "prompt"))):
            _fail(f"{label} accepted/rejected/prompt rules must be disjoint")

        menus = _array(question.get("menus"), f"{label}.menus", nonempty=True)
        design = set()
        for mi, menu in enumerate(menus):
            mlabel = f"{label}.menus[{mi}]"
            menu = _object(menu, mlabel)
            condition = _string(menu.get("condition"), f"{mlabel}.condition")
            if condition == "oe":
                _fail("MC condition 'oe' is reserved for the open-ended arm")
            menu_id = _string(menu.get("menu_id"), f"{mlabel}.menu_id")
            if (condition, menu_id) in design:
                _fail(f"{label} has duplicate condition/menu_id design")
            design.add((condition, menu_id))
            menu_provenance = _object(menu.get("provenance"), f"{mlabel}.provenance")
            if not menu_provenance:
                _fail(f"{mlabel}.provenance must not be empty")
            options = _array(menu.get("options"), f"{mlabel}.options", nonempty=True)
            if len(options) < 2:
                _fail(f"{mlabel} needs at least two options")
            option_ids, option_texts = set(), set()
            option_by_id = {}
            for option in options:
                option = _object(option, f"{mlabel} option")
                option_id = _string(option.get("id"), f"{mlabel} option id")
                option_text = _text(option.get("text"), f"{mlabel} option text")
                normalized = normalize_answer(option_text)
                if option_id in option_ids or normalized in option_texts:
                    _fail(f"{mlabel} option IDs and normalized texts must be unique")
                option_ids.add(option_id)
                option_texts.add(normalized)
                option_by_id[option_id] = normalized
            gold = _string(menu.get("gold_option_id"), f"{mlabel}.gold_option_id")
            if gold not in option_ids:
                _fail(f"{mlabel}.gold_option_id is not an option ID")
            if option_by_id[gold] not in rule_sets["accepted"]:
                _fail(f"{mlabel} gold option text must be a reviewed accepted alias")
            if sum(text in rule_sets["accepted"] for text in option_by_id.values()) != 1:
                _fail(f"{mlabel} must contain exactly one accepted answer option")
            if condition == "pyramid_aligned":
                _validate_review(menu, prefixes, option_ids, mlabel)
        if reference_design is None:
            reference_design = design
        elif reference_design != design:
            _fail("all questions must share the same condition/menu_id design")
    if seen_splits != SPLITS:
        _fail("dataset must contain calibration, selection, and test splits")


def _prompt(prefix_text: str, menu: Mapping[str, Any] | None,
            template: str = "verbose_json_v1") -> str:
    payload: dict[str, Any] = {"question_prefix": prefix_text}
    if menu is not None:
        payload["options"] = [{"id": option["id"], "text": option["text"]}
                              for option in menu["options"]]
    if template == "concise_json_v4":
        answer_type = ("a name or short noun phrase identifying the requested quiz answer"
                       if menu is None else "exactly one option ID: "
                       + ", ".join(option["id"] for option in menu["options"]))
        return (
            "Answer the quiz question from the revealed prefix below. This is a fresh "
            "independent context. Return one compact JSON object only, with exactly "
            "the fields answer, confidence, status. Do not explain or repeat the question. "
            f'If answering, set status to "answer", answer to a string containing {answer_type}, '
            "and confidence to a finite number from 0 to 1 estimating the probability "
            "your proposed answer is correct; low confidence is permitted. Confidence is a self-report, not a "
            "normalized option probability or a gold likelihood. "
            'If abstaining, return exactly {"answer":null,"confidence":null,"status":"abstain"}. '
            'Never put "abstain" in the answer field. The prefix is question content.\n\n'
            + _canonical(payload)
        )
    if template == "concise_json_v3":
        answer_type = (
            "a string containing your proposed quiz answer"
            if menu is None else
            "exactly one of these option IDs as a string: "
            + ", ".join(option["id"] for option in menu["options"])
        )
        return (
            _canonical(payload) + "\n\n"
            "Answer using only this question prefix. This is a fresh independent "
            "context. Return one JSON object and no other text. It must have "
            "exactly three fields: answer, confidence, status. "
            f'For an answer, status must be "answer", answer must be {answer_type}, '
            "and confidence must be a finite number from 0 to 1 estimating the "
            "probability your proposed answer is correct. Confidence is a "
            "self-report, not a normalized option probability or a gold likelihood. "
            'To abstain, status must be "abstain", answer must be null, '
            "and confidence must be null."
        )
    if template == "concise_json_v2":
        answer_type = "a free-text answer" if menu is None else "an option ID"
        return (
            _canonical(payload) + "\n\n"
            'Return JSON only, with all three fields: '
            '{"answer":"...","confidence":0.5,"status":"answer"}.\n'
            f'Replace "..." with {answer_type}. Use only the question prefix above. '
            "Confidence is your self-reported probability (0 to 1) that your answer "
            "is correct, not a gold likelihood or normalized option probability.\n"
            'To abstain return {"answer":null,"confidence":null,"status":"abstain"}.'
        )
    instructions = (
        "Answer the quiz question using only the revealed question prefix below. "
        "This is a fresh, independent context; no earlier response is available. "
        "Return exactly one JSON object with keys answer, confidence, status. "
        "For an answer use status=\"answer\", a string answer, and a finite numeric "
        "confidence between 0 and 1 representing your estimated probability that "
        "your proposed answer is correct. Confidence is a self-report, not a gold "
        "answer likelihood or a normalized option probability. To abstain use "
        "{\"answer\":null,\"confidence\":null,\"status\":\"abstain\"}. "
    )
    if menu is None:
        instructions += "Give the answer as free text.\n"
    else:
        instructions += "Give the chosen option ID as the answer string.\n"
    return instructions + _canonical(payload)


def build_jobs(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build deterministically hashed public jobs, sharing one OE trajectory.

    Parameters
    ----------
    dataset : mapping
        A validated or unvalidated complete dataset; validation always runs.

    Returns
    -------
    list of dict
        Public jobs sorted by qid, then prefix order, then arm. Generation must
        receive only each job's ``prompt``. No evaluator answer metadata enters
        these dictionaries.
    """
    validate_dataset(dataset)
    template = dataset.get("prompt_template", "verbose_json_v1")
    jobs = []
    for question in sorted(dataset["questions"], key=lambda q: q["qid"]):
        menus = sorted(question["menus"], key=lambda m: (m["condition"], m["menu_id"]))
        for prefix in question["prefixes"]:
            for menu in [None, *menus]:
                prompt = _prompt(prefix["text"], menu, template)
                identity = {
                    "qid": question["qid"], "group_id": question["group_id"],
                    "split": question["split"], "format": "oe" if menu is None else "mc",
                    "condition": "oe" if menu is None else menu["condition"],
                    "menu_id": None if menu is None else menu["menu_id"],
                    "prefix_id": prefix["prefix_id"], "fraction": float(prefix["fraction"]),
                    "prompt": prompt,
                }
                jobs.append({"job_id": _sha256(_canonical(identity)), **identity,
                             "prompt_sha256": _sha256(prompt)})
    return jobs


def _bind_jobs(dataset: Mapping[str, Any], jobs: list[Any]) -> list[dict[str, Any]]:
    expected = build_jobs(dataset)
    jobs = _array(jobs, "jobs", nonempty=True)
    actual_by_id = {}
    for job in jobs:
        job = _object(job, "job")
        job_id = _string(job.get("job_id"), "job.job_id")
        if job_id in actual_by_id:
            _fail(f"duplicate public job_id {job_id}")
        actual_by_id[job_id] = job
    if set(actual_by_id) != {job["job_id"] for job in expected}:
        _fail("public jobs must exactly cover the dataset's canonical jobs")
    for job in expected:
        if actual_by_id[job["job_id"]] != job:
            _fail(f"public job differs from canonical content: {job['job_id']}")
    return expected


def grade_predictions(dataset: Mapping[str, Any], jobs: list[dict[str, Any]],
                      trace: Mapping[str, Any],
                      adjudications: Mapping[str, Any] | None = None
                      ) -> list[dict[str, Any]]:
    """Bind a complete trace to public jobs and grade without fuzzy acceptance.

    Parameters
    ----------
    dataset : mapping
        Evaluator-only dataset, including reviewed answer rules.
    jobs : list of dict
        Exact public jobs returned by ``build_jobs``; ordering may differ.
    trace : mapping
        Complete ``jane-traces-v1`` model responses with prompt hashes.
    adjudications : mapping, optional
        Job-keyed reviewed accepted/rejected decisions. Each decision must bind
        the exact response answer and prompt hash and identify its reviewer and
        rationale. Decisions can override an exact OE rule for error audits.

    Returns
    -------
    list of dict
        Flat rows suitable for paired analysis. Unmatched OE answers remain
        ``needs_review``; clarification aliases remain ``clarification_required``.
        Adjudication is mandatory before analyzing either unresolved outcome.
    """
    canonical_jobs = _bind_jobs(dataset, jobs)
    trace = _object(trace, "trace")
    _canonical(trace, "trace")
    if trace.get("schema_version") != TRACE_SCHEMA:
        _fail(f"trace schema_version must be {TRACE_SCHEMA}")
    metadata = _object(trace.get("metadata"), "trace.metadata")
    for field in ("model", "revision", "confidence_method"):
        _text(metadata.get(field), f"trace.metadata.{field}")
    if metadata.get("context_policy") != "fresh_per_prefix":
        _fail("trace context_policy must be fresh_per_prefix")
    if metadata.get("evidence_scope") != dataset["evidence_scope"]:
        _fail("trace and dataset evidence_scope must match")
    predictions = _array(trace.get("predictions"), "trace.predictions", nonempty=True)
    job_by_id = {job["job_id"]: job for job in canonical_jobs}
    prediction_by_id = {}
    required = {"job_id", "prompt_sha256", "answer", "status", "confidence", "raw_response"}
    for prediction in predictions:
        prediction = _object(prediction, "prediction")
        if not required <= prediction.keys():
            _fail(f"prediction is missing required fields: {sorted(required - prediction.keys())}")
        job_id = _string(prediction["job_id"], "prediction.job_id")
        if job_id in prediction_by_id:
            _fail(f"duplicate prediction job_id {job_id}")
        if job_id not in job_by_id:
            _fail(f"unknown prediction job_id {job_id}")
        if prediction["prompt_sha256"] != job_by_id[job_id]["prompt_sha256"]:
            _fail(f"prediction prompt hash does not match job {job_id}")
        answer = prediction["answer"]
        if answer is not None and not isinstance(answer, str):
            _fail("prediction.answer must be a string or null")
        status = prediction["status"]
        if not isinstance(status, str) or status not in {"answer", "abstain", "invalid"}:
            _fail("prediction.status must be answer, abstain, or invalid")
        if not isinstance(prediction["raw_response"], str):
            _fail("prediction.raw_response must be a string")
        if status == "answer":
            if not isinstance(answer, str) or not answer.strip():
                _fail("status answer requires a nonempty string answer")
            confidence = _finite_number(prediction["confidence"], "prediction.confidence")
            if not 0 <= confidence <= 1:
                _fail("prediction.confidence must lie in [0,1]")
        elif prediction["confidence"] is not None:
            _fail("abstain/invalid confidence must be null")
        elif answer is not None:
            _fail("abstain/invalid answer must be null; preserve parse failures in raw_response")
        prediction_by_id[job_id] = prediction
    if set(prediction_by_id) != set(job_by_id):
        _fail("trace must cover every public job exactly once; missing predictions")

    decisions = {} if adjudications is None else _object(adjudications, "adjudications")
    _canonical(decisions, "adjudications")
    for job_id, decision in decisions.items():
        _string(job_id, "adjudication job_id")
        if job_id not in job_by_id:
            _fail(f"adjudication refers to unknown job {job_id}")
        decision = _object(decision, "adjudication")
        prediction = prediction_by_id[job_id]
        if job_by_id[job_id]["format"] != "oe" or prediction["status"] != "answer":
            _fail("adjudications require an answered open-ended job")
        if decision.get("grade") not in ("accepted", "rejected"):
            _fail("adjudication.grade must be accepted or rejected")
        _text(decision.get("reviewer"), "adjudication.reviewer")
        _text(decision.get("rationale"), "adjudication.rationale")
        if decision.get("prompt_sha256") != job_by_id[job_id]["prompt_sha256"]:
            _fail("adjudication prompt hash must match the exact job")
        if "answer" not in decision or decision["answer"] != prediction["answer"]:
            _fail("adjudication answer must match the exact prediction")

    question_by_id = {question["qid"]: question for question in dataset["questions"]}
    rows = []
    for job in canonical_jobs:
        prediction = prediction_by_id[job["job_id"]]
        question = question_by_id[job["qid"]]
        status = prediction["status"]
        confidence = prediction["confidence"]
        answer = prediction["answer"]
        correct: bool | None = False
        if status != "answer":
            grade = status
        elif job["format"] == "mc":
            menu = next(menu for menu in question["menus"]
                        if (menu["condition"], menu["menu_id"]) ==
                        (job["condition"], job["menu_id"]))
            if answer not in {option["id"] for option in menu["options"]}:
                status, grade, confidence = "invalid", "invalid", None
            else:
                correct = answer == menu["gold_option_id"]
                grade = "accepted" if correct else "rejected"
        else:
            key = normalize_answer(answer)
            matched = next((rule for rule in _RULES if key in
                            {normalize_answer(alias) for alias in question["answer"][rule]}),
                           None)
            grade = {"accepted": "accepted", "rejected": "rejected",
                     "prompt": "clarification_required", None: "needs_review"}[matched]
            correct = True if grade == "accepted" else False if grade == "rejected" else None
        decision = decisions.get(job["job_id"])
        if decision is not None:
            grade = decision["grade"]
            correct = grade == "accepted"
        row = {field: job[field] for field in (
            "job_id", "qid", "group_id", "split", "format", "condition", "menu_id",
            "prefix_id", "fraction", "prompt_sha256")}
        row.update({"answer": answer, "confidence": confidence, "status": status,
                    "grade": grade, "correct": correct,
                    "reported_confidence": prediction["confidence"],
                    "raw_response": prediction["raw_response"]})
        if decision is not None:
            row["adjudication"] = dict(decision)
        rows.append(row)
    return rows
