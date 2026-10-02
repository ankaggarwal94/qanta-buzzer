"""Freeze a larger paired question dataset from retained QBReader source bytes.

No network request, model generation, training, or inferred human judgment occurs.
The supported source is JSONL containing normalized QBReader tossup objects. Raw
HTML answerlines are retained. Explicit simple aliases are syntax extractions,
not a replacement for human adjudication. Run --help for the create-once CLI.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import sys
import unicodedata
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from qb_data.jane_paired import build_jobs, normalize_answer, validate_dataset
from scripts.prepare_jane_gpu_pilot import build_choice_controls

SPLIT_COUNTS = {"calibration": 1000, "selection": 1000, "test": 3000}
PUBLIC_FIELDS = {"job_id", "qid", "group_id", "split", "format", "condition",
                 "menu_id", "prefix_id", "fraction", "prompt", "prompt_sha256"}
CONDITIONS = ("independent_pool", "same_category_pool")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rank(seed: int, domain: str, identity: str) -> str:
    return sha(f"acl-expansion-v1|seed={seed}|{domain}|{identity}")


class VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.unsafe = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"script", "style"}:
            self.unsafe = True
        if tag in {"br", "p", "div"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def visible_text(value: str) -> str:
    parser = VisibleText()
    parser.feed(value)
    if parser.unsafe:
        raise ValueError("script/style HTML is not eligible")
    return " ".join("".join(parser.parts).split())


def entity_key(value: str) -> str:
    """Use a conservative collision key for exclusions, never for grading."""
    text = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def extract_answer(raw: str) -> tuple[str | None, list[str], str]:
    """Extract a simple leading answer and explicitly unconditional aliases.

    The entire answerline remains available for later adjudication. Compound
    leading answers and condition-dependent aliases are not silently simplified.
    Underlining is required as source evidence, but surname fragments are not
    converted into accepted aliases. Only whole simple '[accept X]' or '[or X]'
    clauses are automatic aliases; other instructions remain unresolved.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, [], "missing answerline"
    raw = re.sub(r"^\s*ANSWER:\s*", "", raw, flags=re.I)
    head = raw.split("[", 1)[0].strip()
    if not re.search(r"<u(?:\s[^>]*)?>", head, re.I):
        return None, [], "no underlined leading answer"
    try:
        primary = visible_text(head)
    except ValueError as exc:
        return None, [], str(exc)
    def simple(text: str) -> bool:
        return bool(text and len(text) <= 120 and len(text.split()) <= 12
                    and not re.search(r"[\[\]();/=:\n\r]|\b(or|and|before|after|until|prompt|accept|reject)\b", text, re.I)
                    and not any(unicodedata.category(c) == "Cc" for c in text))
    if not simple(primary):
        return None, [], "complex leading answer requires review"
    aliases = [primary]
    for clause in re.findall(r"\[([^\[\]]+)\]", raw):
        try:
            text = visible_text(clause)
        except ValueError:
            continue
        match = re.fullmatch(r"(?:accept|or)\s+(.+)", text, re.I)
        if (match and simple(match.group(1))
                and not re.search(r"\b(either|any|anything|equivalent|equivalents|similar|alternate|alternative|alternatives|word|words|form|forms|description|descriptions|response|responses|answer|answers|when|during|for|unless|except|also)\b", match.group(1), re.I)):
            alias = match.group(1).strip()
            if normalize_answer(alias) not in {normalize_answer(a) for a in aliases}:
                aliases.append(alias)
    return primary, aliases, "unreviewed strict leading answer plus explicit unconditional simple aliases"


def primary_required_form(raw: str) -> str:
    """Return underlined leading text only as a conservative identity key."""
    head = raw.split("[", 1)[0]
    pieces = re.findall(r"<u(?:\s[^>]*)?>(.*?)</u>", head, re.I | re.S)
    try:
        return visible_text(" ".join(pieces))
    except ValueError:
        return ""


def primary_identity_forms(raw: str) -> list[str]:
    """Add separate underlined identities only for an explicit either rule.

    This supports source answerlines such as a person's name and title marked
    '[accept either underlined portion]'. The fragments are conservative
    freshness/grouping keys, never automatic OE grading aliases. Ordinary
    multiword answers with several underlined spans remain a single identity.
    """
    combined = primary_required_form(raw)
    forms = [combined] if combined else []
    clauses = re.findall(r"\[([^\[\]]+)\]", raw)
    either_instruction = False
    for clause in clauses:
        try:
            instruction = visible_text(clause).split(";", 1)[0].strip()
        except ValueError:
            continue
        if re.fullmatch(r"accept either(?: underlined portions?)?", instruction, re.I):
            either_instruction = True
            break
    if either_instruction:
        head = raw.split("[", 1)[0]
        for piece in re.findall(r"<u(?:\s[^>]*)?>(.*?)</u>", head, re.I | re.S):
            try:
                form = visible_text(piece)
            except ValueError:
                continue
            if form and form not in forms:
                forms.append(form)
    return forms


def word_prefixes(question: str) -> list[dict]:
    """Return ten exact leading substrings at floor(N*k/10) word endpoints."""
    tokens = list(re.finditer(r"\S+", question))
    if len(tokens) < 40:
        raise ValueError("question has fewer than 40 whitespace words")
    prefixes = []
    for k in range(1, 11):
        count = len(tokens) * k // 10
        end = len(question) if k == 10 else tokens[count - 1].end()
        prefixes.append({"prefix_id": f"p{k}", "text": question[:end],
                         "fraction": count / len(tokens), "requested_fraction": k / 10,
                         "token_count": count, "token_start_index": 0,
                         "token_end_index_exclusive": count, "character_end_exclusive": end,
                         "boundary_status": "fixed whitespace-word endpoint; not a clue boundary"})
    return prefixes


def shingles(question: str) -> frozenset[int]:
    words = entity_key(question).split()
    return frozenset(int.from_bytes(hashlib.sha256(" ".join(words[i:i+5]).encode()).digest()[:8], "big")
                     for i in range(max(0, len(words) - 4)))


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.sizes = [1] * count

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.sizes[a] < self.sizes[b]:
                a, b = b, a
            self.parent[b] = a
            self.sizes[a] += self.sizes[b]


def near_duplicate_edges(values: list[frozenset[int]], threshold: float = .85) -> Iterable[tuple[int, int, float]]:
    """Yield all exact-threshold Jaccard pairs with prefix-index filtering.

    A shared global token ordering and ceil(t*size) prefix bound are used for
    candidate generation. Every candidate is checked with full shingle sets.
    SHA256 64-bit shingle hashes have negligible but nonzero collision risk.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    frequencies = Counter(token for value in values for token in value)
    index: dict[int, list[int]] = defaultdict(list)
    for i in sorted(range(len(values)), key=lambda i: (len(values[i]), i)):
        value = values[i]
        if not value:
            continue
        ordered = sorted(value, key=lambda token: (frequencies[token], token))
        prefix = ordered[:len(value) - math.ceil(threshold * len(value)) + 1]
        candidates = set()
        for token in prefix:
            candidates.update(j for j in index[token] if len(values[j]) >= threshold * len(value))
        for j in sorted(candidates):
            other = values[j]
            overlap = len(value & other)
            similarity = overlap / (len(value) + len(other) - overlap)
            if similarity + 1e-12 >= threshold:
                yield j, i, similarity
        for token in prefix:
            index[token].append(i)


def allocate(total: int, sizes: dict[tuple | str, int]) -> dict:
    """Allocate exact bounded proportional quotas by largest remainders."""
    denominator = sum(sizes.values())
    if total < 0 or total > denominator:
        raise ValueError(f"cannot allocate {total} from {denominator}")
    if denominator == 0:
        return {key: 0 for key in sizes}
    quotas = {key: total * size // denominator for key, size in sizes.items()}
    remaining = total - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (-(total * sizes[key] % denominator), str(key)))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def allocate_bounded(total: int, weights: dict, capacities: dict) -> dict:
    """Use source-weighted largest remainders, redistributing capacity deficits.

    Strata whose continuous allocation exceeds remaining capacity are fixed at
    capacity; their deficit is redistributed proportionally to source weights
    among the remaining strata. This repeats before final integer rounding.
    """
    keys = set(weights) | set(capacities)
    if total < 0 or total > sum(capacities.get(k, 0) for k in keys if weights.get(k, 0) > 0):
        raise ValueError("insufficient unique-component capacity for weighted sample")
    result = {key: 0 for key in keys}
    remaining = total
    active = {key for key in keys if weights.get(key, 0) > 0 and capacities.get(key, 0) > 0}
    while active and remaining:
        denominator = sum(weights[key] for key in active)
        capped = {key for key in active if remaining * weights[key] >= capacities[key] * denominator}
        if not capped:
            allocated = {key: remaining * weights[key] // denominator for key in active}
            residue = remaining - sum(allocated.values())
            order = sorted(active, key=lambda key: (-(remaining * weights[key] % denominator), str(key)))
            for key in order[:residue]:
                allocated[key] += 1
            result.update(allocated)
            remaining = 0
            break
        for key in capped:
            result[key] = capacities[key]
            remaining -= capacities[key]
        active -= capped
    if remaining or sum(result.values()) != total or any(result[k] > capacities.get(k, 0) for k in keys):
        raise ValueError("bounded allocation failed exact count or capacity check")
    return result


@dataclass(frozen=True)
class Config:
    seed: int = 1
    min_year: int = 2010
    max_year: int = 2025
    categories: tuple[str, ...] = ()
    reservoir_per_category: int = 100
    main_count: int = 5000
    calibration_count: int = 1000
    selection_count: int = 1000
    holdout_year: int = 2026
    holdout_count: int = 500
    near_duplicate_threshold: float = .85


def _metadata(source: dict, key: str):
    if key in source:
        return source[key]
    set_info = source.get("set")
    return set_info.get(key) if isinstance(set_info, dict) else None


def load_candidates(source_path: Path, exclusions: dict, config: Config) -> tuple[list[dict], dict]:
    excluded_ids = {str(s).removeprefix("qbreader:") for s in exclusions.get("source_ids", [])}
    excluded_aliases = {entity_key(a) for a in exclusions.get("aliases", []) if isinstance(a, str)}
    excluded_questions = {entity_key(q) for q in exclusions.get("questions", []) if isinstance(q, str)}
    records, reasons, seen_ids = [], Counter(), set()
    source_weights = Counter()
    with source_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            source = json.loads(line)
            source_id = str(source.get("_id", source.get("id", "")))
            category = source.get("category")
            year, difficulty, standard = (_metadata(source, name) for name in ("year", "difficulty", "standard"))
            reason = None
            if not source_id or source_id in seen_ids:
                reason = "missing_or_duplicate_source_id"
            seen_ids.add(source_id)
            if not isinstance(category, str) or not category or category in {"Trash", "Pop Culture"}:
                reason = "excluded_or_missing_category"
            elif config.categories and category not in config.categories:
                reason = "category_outside_declared_scope"
            elif standard is not True:
                reason = "not_explicitly_standard"
            elif isinstance(year, bool) or not isinstance(year, int) or not (config.min_year <= year <= config.max_year or year == config.holdout_year):
                reason = "year_outside_declared_scope"
            elif isinstance(difficulty, bool) or not isinstance(difficulty, int):
                reason = "missing_integer_difficulty"
            if reason is None and config.min_year <= year <= config.max_year:
                source_weights[(category, difficulty)] += 1
            question = source.get("question_sanitized")
            if not isinstance(question, str) or not question.strip() or len(question.split()) < 40:
                reason = reason or "missing_or_short_question"
            elif any(unicodedata.category(c) == "Cc" and not c.isspace() for c in question):
                reason = reason or "question_control_character"
            answer, aliases, explanation = extract_answer(source.get("answer", ""))
            if not answer:
                reason = reason or "answerline_ineligible"
            keys = {entity_key(a) for a in aliases}
            keys.update(entity_key(form) for form in primary_identity_forms(source.get("answer", "")))
            freshness_blocked = source_id in excluded_ids or bool(keys & excluded_aliases) or (
                isinstance(question, str) and entity_key(question) in excluded_questions)
            if reason:
                reasons[reason] += 1
                continue
            normalized_source = {**source, "_id": source_id, "year": year, "difficulty": difficulty, "standard": standard}
            records.append({"source": normalized_source, "canonical_answer": answer, "aliases": aliases,
                            "extraction_reason": explanation, "alias_keys": sorted(keys), "source_line": line_number,
                            "freshness_blocked": freshness_blocked})
    return records, {"source_record_count": len(seen_ids), "eligible_before_grouping": len(records), "filter_counts": dict(reasons),
                     "source_population_weights": {f"{a}|{b}": n for (a,b),n in sorted(source_weights.items())},
                     "source_weight_definition": "all standard academic source rows in main year range before answerline, freshness, grouping, or reservoir filters"}


def group_candidates(records: list[dict], exclusions: dict, config: Config, identity_overrides: dict | None = None) -> tuple[list[dict], dict]:
    """Deduplicate alias and near-question components, including prior text."""
    records.sort(key=lambda row: rank(config.seed, "component_representative", row["source"]["_id"]))
    prior_texts = sorted({q for q in exclusions.get("questions", []) if isinstance(q, str) and len(q.split()) >= 5})
    sets = [shingles(row["source"]["question_sanitized"]) for row in records] + [shingles(q) for q in prior_texts]
    groups = UnionFind(len(sets))
    curated_groups = []
    if identity_overrides is not None:
        if (identity_overrides.get("schema_version") != "acl-identity-overrides-v1"
                or not isinstance(identity_overrides.get("reviewer"), str)
                or not identity_overrides["reviewer"].strip()
                or not isinstance(identity_overrides.get("equivalences"), list)):
            raise ValueError("invalid curated identity override envelope")
        positions = {row["source"]["_id"]: i for i,row in enumerate(records)}
        for entry in identity_overrides["equivalences"]:
            ids = entry.get("source_ids")
            evidence = entry.get("evidence")
            if (not isinstance(ids, list) or len(set(ids)) != len(ids) or len(ids) < 2
                    or not all(isinstance(source_id, str) and source_id in positions for source_id in ids)
                    or not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip()
                    or not isinstance(evidence, list)):
                raise ValueError("curated identity group has unknown IDs, duplicates, or missing rationale/evidence")
            checked = set()
            for item in evidence:
                source_id = item.get("source_id")
                if source_id not in ids or source_id in checked:
                    raise ValueError("curated identity evidence has unknown or duplicate source ID")
                raw = records[positions[source_id]]["source"]["answer"]
                if (item.get("answerline") != raw or not isinstance(item.get("span"), str)
                        or not item["span"] or item["span"] not in raw):
                    raise ValueError("curated identity evidence does not match exact source answerline/span")
                checked.add(source_id)
            if checked != set(ids):
                raise ValueError("curated identity group lacks evidence for every source ID")
            for source_id in ids[1:]:
                groups.union(positions[ids[0]], positions[source_id])
            curated_groups.append(entry)
    alias_owner = {}
    alias_edges = 0
    for i, record in enumerate(records):
        for key in record["alias_keys"]:
            if key in alias_owner:
                groups.union(i, alias_owner[key])
                alias_edges += 1
            else:
                alias_owner[key] = i
    near_edges = []
    for a, b, similarity in near_duplicate_edges(sets, config.near_duplicate_threshold):
        groups.union(a, b)
        near_edges.append({"a": records[a]["source"]["_id"] if a < len(records) else f"prior_text:{a-len(records)}",
                           "b": records[b]["source"]["_id"] if b < len(records) else f"prior_text:{b-len(records)}",
                           "jaccard": similarity})
    prior_groups = {groups.find(i) for i in range(len(records), len(sets))}
    prior_groups.update(groups.find(i) for i, record in enumerate(records) if record.get("freshness_blocked"))
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        components[groups.find(i)].append(i)
    representatives, excluded_near_prior = [], 0
    for root, indices in components.items():
        if root in prior_groups:
            excluded_near_prior += len(indices)
            continue
        representative = records[indices[0]]
        source_ids = sorted(records[i]["source"]["_id"] for i in indices)
        representative["group_id"] = "component:" + sha(canonical(source_ids))
        representative["component_source_ids"] = source_ids
        representative["component_alias_keys"] = sorted({key for i in indices for key in records[i]["alias_keys"]})
        representatives.append(representative)
    return representatives, {"curated_identity_override_groups": curated_groups, "curated_identity_reviewer": identity_overrides.get("reviewer") if identity_overrides else None,
                            "alias_edges": alias_edges, "near_duplicate_threshold": config.near_duplicate_threshold,
                            "near_duplicate_edges": near_edges, "component_count_before_prior_removal": len(components),
                            "prior_near_duplicate_excluded_records": excluded_near_prior,
                            "eligible_unique_components": len(representatives)}


def stratum(record: dict) -> tuple[str, int]:
    return record["source"]["category"], record["source"]["difficulty"]


def sample_stratified(records: list[dict], count: int, seed: int, domain: str, quotas: dict | None = None) -> tuple[list[dict], list[dict]]:
    pools: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        pools[stratum(record)].append(record)
    if quotas is None:
        quotas = allocate(count, {key: len(rows) for key, rows in pools.items()})
    if sum(quotas.values()) != count or any(n > len(pools.get(key, [])) for key,n in quotas.items()):
        raise ValueError("sampling quota exceeds capacity or requested count")
    selected, remaining = [], []
    for key, rows in sorted(pools.items()):
        rows.sort(key=lambda row: rank(seed, domain, row["source"]["_id"]))
        selected.extend(rows[:quotas[key]])
        remaining.extend(rows[quotas[key]:])
    return selected, remaining


def select_records(records: list[dict], config: Config, source_weights: dict | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    historical = [row for row in records if config.min_year <= row["source"]["year"] <= config.max_year]
    holdout_pool = [row for row in records if row["source"]["year"] == config.holdout_year and not config.min_year <= row["source"]["year"] <= config.max_year]
    reservoir, remaining = [], []
    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in historical:
        by_category[record["source"]["category"]].append(record)
    dropped_categories = []
    for category, rows in sorted(by_category.items()):
        if len(rows) < config.reservoir_per_category + 1:
            dropped_categories.append({"category": category, "unique_components": len(rows), "reason": "insufficient separate reservoir plus main"})
            continue
        rows.sort(key=lambda row: rank(config.seed, "reservoir", row["source"]["_id"]))
        reservoir.extend(rows[:config.reservoir_per_category])
        remaining.extend(rows[config.reservoir_per_category:])
    capacities = dict(Counter(stratum(row) for row in remaining))
    weights = source_weights if source_weights is not None else capacities
    quotas = allocate_bounded(config.main_count, weights, capacities)
    unbounded = allocate(config.main_count, weights)
    allocation_audit = [{"category": key[0], "difficulty": key[1],
                         "source_rows": weights.get(key, 0), "available_unique_components": capacities.get(key, 0),
                         "target_exact": config.main_count * weights.get(key, 0) / sum(weights.values()),
                         "target_integer_unconstrained": unbounded.get(key, 0), "allocated_main": quotas.get(key, 0),
                         "redistributed_difference": quotas.get(key, 0) - unbounded.get(key, 0)}
                        for key in sorted(set(weights) | set(capacities))]
    main, unused = sample_stratified(remaining, config.main_count, config.seed, "main_selection", quotas)
    calibration, residual = sample_stratified(main, config.calibration_count, config.seed, "calibration")
    selection, test = sample_stratified(residual, config.selection_count, config.seed, "selection")
    for split, rows in (("calibration", calibration), ("selection", selection), ("test", test)):
        for row in rows:
            row["split"] = split
    main = calibration + selection + test
    holdout_count = min(config.holdout_count, len(holdout_pool))
    holdout, _ = sample_stratified(holdout_pool, holdout_count, config.seed, "holdout_reserve")
    return main, reservoir, holdout, {"main_eligible_after_reservoir": len(remaining), "unused_historical_components": len(unused),
             "dropped_categories": dropped_categories, "holdout_eligible_components": len(holdout_pool),
             "holdout_requested": config.holdout_count, "holdout_actual": len(holdout),
             "source_weight_total": sum(weights.values()), "stratum_allocation": allocation_audit,
             "allocation_policy": "raw standard academic source category/difficulty weights; iterative unique-component capacity capping and proportional redistribution; largest-remainder rounding",
             "estimand": "eligible unique-component sample targeted to raw question category/difficulty mix, with capacity-driven deviations; not a random sample of all source questions"}


def build_dataset(records: list[dict], reservoir: list[dict], config: Config, provenance: dict) -> dict:
    questions = []
    for index, record in enumerate(records):
        source, answer = record["source"], record["canonical_answer"]
        menus = []
        for condition in CONDITIONS:
            candidates = [r for r in reservoir if condition == "independent_pool" or r["source"]["category"] == source["category"]]
            candidates.sort(key=lambda r: rank(config.seed, f"menu|{condition}|{source['_id']}", r["source"]["_id"]))
            distractors = candidates[:3]
            if len(distractors) != 3:
                raise ValueError("fewer than three distinct distractor components")
            if any(set(record["component_alias_keys"]) & set(r["component_alias_keys"]) for r in distractors):
                raise ValueError("distractor component overlaps correct answer")
            options = [r["canonical_answer"] for r in distractors]
            options.insert(index % 4, answer)
            menus.append({"condition": condition, "menu_id": "fixed_1", "gold_option_id": "ABCD"[index % 4],
                "options": [{"id": letter, "text": text} for letter, text in zip("ABCD", options)],
                "provenance": {"method": "seeded SHA256-ranked disjoint answer-component reservoir",
                    "candidate_source_ids": [r["source"]["_id"] for r in distractors],
                    "same_category_constraint": condition == "same_category_pool", "fixed_across_prefixes": True,
                    "syntactic_alias_disjoint": True, "semantic_equivalence_review": False,
                    "answer_type_matched": False, "clue_compatibility_review": False}})
        questions.append({"qid": "qbreader:" + source["_id"], "group_id": record["group_id"],
            "split": record["split"], "question": source["question_sanitized"],
            "prefixes": word_prefixes(source["question_sanitized"]),
            "answer": {"raw": source["answer"], "accepted": record["aliases"], "rejected": [], "prompt": []},
            "menus": menus, "source": {**source, "id": source["_id"],
                "canonical_answer": answer, "answer_extraction": record["extraction_reason"],
                "component_source_ids": record["component_source_ids"],
                "component_alias_keys": record["component_alias_keys"],
                "question_utf8_sha256": sha(source["question_sanitized"]),
                "answerline_utf8_sha256": sha(source["answer"])}})
    dataset = {"schema_version": "jane-paired-v1", "evidence_scope": "scientific", "prompt_template": "concise_json_v4",
        "source": {"origin": "frozen QBReader database snapshot; ACL expansion", "provenance": provenance,
            "full_answerlines_available": True, "human_review": False,
            "scope_note": "Prepared inference inputs, not completed experiments. Scientific schema does not certify answerline or distractor validity."},
        "questions": questions}
    validate_dataset(dataset)
    return dataset


def assert_public_jobs(jobs: list[dict]) -> None:
    seen = set()
    for job in jobs:
        if set(job) != PUBLIC_FIELDS or job["job_id"] in seen:
            raise ValueError("unexpected public fields or duplicate job")
        seen.add(job["job_id"])
        identity = {key: value for key, value in job.items() if key not in {"job_id", "prompt_sha256"}}
        if sha(canonical(identity)) != job["job_id"] or sha(job["prompt"]) != job["prompt_sha256"]:
            raise ValueError("public job hash mismatch")
        payload = json.loads(job["prompt"].rsplit("\n\n", 1)[1])
        if set(payload) != ({"question_prefix"} if job["format"] == "oe" else {"question_prefix", "options"}):
            raise ValueError("unexpected prompt payload fields")


def counts(records: list[dict]) -> dict:
    return {"question_count": len(records), "categories": dict(Counter(r["source"]["category"] for r in records)),
            "difficulties": dict(Counter(str(r["source"]["difficulty"]) for r in records)),
            "years": dict(Counter(str(r["source"]["year"]) for r in records)),
            "category_difficulty": {f"{a}|{b}": n for (a,b),n in sorted(Counter(stratum(r) for r in records).items())}}


def prepare(source_path: Path, exclusions_path: Path, out_dir: Path, config: Config, identity_overrides_path: Path | None = None) -> dict:
    """Create one immutable input freeze; fail if the output directory exists."""
    if config.calibration_count <= 0 or config.selection_count <= 0 or config.main_count <= config.calibration_count + config.selection_count:
        raise ValueError("all three splits must be nonempty")
    if config.reservoir_per_category < 3:
        raise ValueError("reservoir requires at least three answers per category")
    exclusions = json.loads(exclusions_path.read_text())
    identity_overrides = json.loads(identity_overrides_path.read_text()) if identity_overrides_path else None
    out_dir.mkdir(parents=True, exist_ok=False)
    evaluator, public = out_dir / "evaluator", out_dir / "public"
    evaluator.mkdir(); public.mkdir()
    provenance = {"normalized_source_sha256": file_sha(source_path), "exclusions_sha256": file_sha(exclusions_path),
                  "preparer_code_sha256": file_sha(Path(__file__)), "seed": config.seed,
                  "identity_overrides_sha256": file_sha(identity_overrides_path) if identity_overrides_path else None}
    plan = {"schema_version": "acl-expansion-data-plan-v1", "config": config.__dict__, **provenance,
            "split_policy": "raw-source-weighted category/difficulty quotas with unique-component capacity redistribution, then proportional 1000/1000/3000 partitions",
            "prefix_policy": "ten exact word-endpoint prefixes floor(N*k/10), k1..10; full text at k10",
            "near_duplicate_policy": "exact Jaccard of punctuation/case-normalized word5 shingles at declared threshold, transitive components",
            "answer_policy": "strict primary plus explicitly unconditional simple aliases; preserve raw HTML; all unmatched responses require review",
            "entity_identity_limitation": "syntactic alias components plus explicitly source-evidenced curated overrides, not certified semantic entities",
            "identity_override_policy": "optional source-ID equivalences validated against exact source answerlines and spans before component selection; reviewer and input hash retained",
            "holdout_policy": "extra source candidates from separately declared year, no outcomes or inference jobs",
            "answer_type_matching": "not inferred automatically; explicit human-review queue, no type-matched condition yet"}
    write_json(out_dir / "data_plan.json", plan)
    records, loading = load_candidates(source_path, exclusions, config)
    print(canonical({"stage": "loaded", **loading}), flush=True)
    representatives, grouping = group_candidates(records, exclusions, config, identity_overrides)
    print(canonical({"stage": "grouped", "eligible_unique_components": len(representatives)}), flush=True)
    weights = {(key.rsplit("|", 1)[0], int(key.rsplit("|", 1)[1])): value
               for key,value in loading["source_population_weights"].items()}
    main, reservoir, holdout, selection = select_records(representatives, config, weights)
    dataset = build_dataset(main, reservoir, config, provenance)
    write_json(evaluator / "main_dataset.json", dataset)
    write_json(evaluator / "reservoir.json", reservoir)
    write_json(evaluator / "holdout_candidates.json", {"status": "reserved source records only; not evaluated", "records": holdout})
    write_json(evaluator / "selection_audit.json", {"loading": loading, "grouping": grouping, "selection": selection,
        "selected_source_ids": [r["source"]["_id"] for r in main], "source_component_count": len(representatives)})
    write_json(evaluator / "review_queue.json", {"status": "human review pending; no fabricated labels",
        "required_checks": ["full answerline and accepted alias validity", "prompt/reject/conditional instructions", "semantic answer equivalence across partitions", "distractor answer type and unique correct option", "clue compatibility"],
        "questions": [{"qid": q["qid"], "split": q["split"], "answerline": q["answer"]["raw"],
                       "syntactically_accepted": q["answer"]["accepted"], "answer_type": None,
                       "answer_type_reviewed": False, "human_status": "pending"} for q in dataset["questions"]]})
    jobs = build_jobs(dataset)
    assert_public_jobs(jobs)
    write_json(public / "main_jobs.json", {"schema_version": "jane-public-jobs-v1", "evidence_scope": "scientific", "jobs": jobs})
    with (public / "main_inference_minimal.jsonl").open("x", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(canonical({key: job[key] for key in ("job_id", "prompt", "prompt_sha256")}) + "\n")
    controls = build_choice_controls(dataset)
    controls["evidence_scope"] = "scientific"
    write_json(public / "main_choices_only.json", controls)
    gold_lookup = {(q["qid"],m["condition"],m["menu_id"]): m["gold_option_id"] for q in dataset["questions"] for m in q["menus"]}
    write_json(evaluator / "main_choices_only_gold.json", {j["job_id"]:gold_lookup[(j["qid"],j["condition"],j["menu_id"])] for j in controls["jobs"]})
    summary = {"schema_version": "acl-expansion-freeze-v1", "seed": config.seed, "inference_run": False,
        "human_review_completed": False, "main": counts(main), "reservoir": counts(reservoir), "holdout": counts(holdout),
        "splits": {split: counts([r for r in main if r["split"] == split]) for split in ("calibration", "selection", "test")},
        "trajectory_jobs_per_model": len(jobs), "choices_only_jobs_per_model": len(controls["jobs"]),
        "prefixes_per_question": 10, "mc_conditions": list(CONDITIONS),
        "type_matched_condition_status": "pending human answer-type annotations; absent from current jobs",
        "sealed_test_status": "test outcomes nonexistent; identities fixed before inference; procedural holdout only, no encryption or access control",
        "limitations": ["eligibility filtering changes population", "syntactic identity does not prove semantic entity disjointness", "word fraction is not clue boundary", "no human grading/adjudication", "no model inference or experimental results"]}
    write_json(out_dir / "freeze_summary.json", summary)
    manifest = {str(path.relative_to(out_dir)): {"sha256": file_sha(path), "byte_count": path.stat().st_size}
                for path in sorted(out_dir.rglob("*")) if path.is_file()}
    write_json(out_dir / "manifest.json", {"schema_version": "acl-expansion-manifest-v1", "files": manifest,
        "source_provenance": provenance, "public_inputs": [key for key in manifest if key.startswith("public/")]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--identity-overrides", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-year", type=int, default=2010)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--categories", default="")
    parser.add_argument("--reservoir-per-category", type=int, default=100)
    parser.add_argument("--holdout-year", type=int, default=2026)
    parser.add_argument("--holdout-count", type=int, default=500)
    args = parser.parse_args()
    config = Config(seed=args.seed, min_year=args.min_year, max_year=args.max_year,
                    categories=tuple(x.strip() for x in args.categories.split(",") if x.strip()),
                    reservoir_per_category=args.reservoir_per_category,
                    holdout_year=args.holdout_year, holdout_count=args.holdout_count)
    print(canonical(prepare(args.source_jsonl, args.exclusions, args.out_dir, config, args.identity_overrides)))


if __name__ == "__main__":
    main()
