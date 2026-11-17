"""Train-reservoir, reviewed-equivalence menu oracles for opt-in ACL experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
from scipy.sparse import coo_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


STRATEGIES = frozenset({
    "baseline", "uniform_global", "uniform_category", "tfidf_easy", "tfidf_hard",
    "sbert_easy", "sbert_hard", "mixed", "prefix_tfidf", "prefix_sbert",
})


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _label_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("answer or alias must be nonempty text")
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def select_indices(n: int, count: int, digits: Sequence[int]) -> tuple[int, ...]:
    """Decode a shrinking-pool choice vector into distinct original indices.

    Parameters
    ----------
    n, count : int
        Original pool size and number of draws.
    digits : sequence of int
        At draw i, a zero-based index into the remaining sorted pool.

    Returns
    -------
    tuple of int
        Original indices in draw order.
    """
    if type(n) is not int or type(count) is not int or not 0 <= count <= n:
        raise ValueError("invalid draw count or pool size")
    if len(digits) != count:
        raise ValueError("digit count must match draw count")
    remaining = list(range(n))
    selected = []
    for digit in digits:
        if type(digit) is not int or not 0 <= digit < len(remaining):
            raise ValueError("digit outside shrinking pool")
        selected.append(remaining.pop(digit))
    return tuple(selected)


@dataclass(frozen=True)
class ReservoirRow:
    answer: str
    profile: str
    category: str
    group_id: str
    split: str = "train"
    source_id: str = ""


@dataclass(frozen=True)
class Menu:
    options: tuple[str, ...]
    class_ids: tuple[str, ...]
    source_ids: tuple[tuple[str, ...], ...]
    gold_index: int
    strategy: str
    equivalence_version: str
    equivalence_digest: str
    log_probability: float

    @property
    def digest(self) -> str:
        """Content identity of the realized ordered menu and audit metadata."""
        return _digest(asdict(self))


class ProfileScorer(Protocol):
    def score(self, clue_prefix: str, option_profiles: list[str]) -> Sequence[float]: ...


class DrawDecoder(Protocol):
    def draw(self, n: int, digits: Sequence[int]) -> Sequence[int]: ...
    def insert(self, xs: Sequence[int], gold: int, pos: int) -> Sequence[int]: ...


class AnswerClasses:
    """Closed registry of reviewed global answer equivalences."""

    def __init__(self, aliases: Mapping):
        if not isinstance(aliases, Mapping):
            raise ValueError("missing reviewed equivalence registry")
        for key in ("version", "reviewer"):
            if not isinstance(aliases.get(key), str) or not aliases[key].strip():
                raise ValueError(f"missing equivalence review {key}")
        classes = aliases.get("classes")
        if not isinstance(classes, Mapping) or not classes:
            raise ValueError("missing reviewed answer classes")
        self._lookup: dict[str, str] = {}
        for canonical, variants in classes.items():
            if not isinstance(variants, list):
                raise ValueError("reviewed aliases must be lists")
            for label in [canonical, *variants]:
                key = _label_key(label)
                if self._lookup.setdefault(key, canonical) != canonical:
                    raise ValueError(f"ambiguous reviewed alias {label!r}")
        self.equivalence_version = aliases["version"]
        self.equivalence_digest = _digest(aliases)

    def class_id(self, label: str) -> str:
        """Return the reviewed canonical answer for a covered label."""
        key = _label_key(label)
        if key not in self._lookup:
            raise ValueError(f"unreviewed answer {label!r}")
        return self._lookup[key]

    def rule_classes(self, labels, *, context_only=()) -> set[str]:
        """Resolve known classes, allowing explicitly local spellings to lack global IDs."""
        if not isinstance(context_only, (list, tuple)):
            raise ValueError("source-dependent terms must be listed for this question")
        local = {_label_key(label) for label in context_only}
        classes = set()
        for label in labels:
            key = _label_key(label)
            if key in self._lookup:
                classes.add(self._lookup[key])
            elif key not in local:
                raise ValueError(f"unreviewed source rule {label!r}")
        return classes


class _GroupTfidf:
    """Raw class counts shared by exact leave-group-out TF-IDF indexes."""

    def __init__(self, catalog):
        self.labels = sorted(catalog)
        self.positions = {label: i for i, label in enumerate(self.labels)}
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float64)
        try:
            self.counts = self.vectorizer.fit_transform(
                " ".join(r.profile for _, r in catalog[label]) for label in self.labels)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            self.counts = None
            return
        self.counts.sort_indices()
        self.df = np.bincount(self.counts.indices, minlength=self.counts.shape[1])
        self.full_weights = TfidfTransformer().fit(self.counts)
        self.full_matrix = self.full_weights.transform(self.counts)

    def score(self, eligible, labels, affected, prefix):
        if self.counts is None:
            return np.zeros(len(labels))
        query = self.vectorizer.transform([prefix])
        if not affected:
            return (self.full_matrix @ self.full_weights.transform(query).T).toarray().ravel()

        # Remove excluded classes and replace only the partially excluded profiles.
        # Their remaining tokens are a subset of the old vocabulary. Terms absent
        # from the eligible corpus get zero query weight, just as with a fresh fit.
        df = self.df.copy()
        replacements, positions = [], []
        eligible_pos = {label: i for i, label in enumerate(labels)}
        for label in sorted(affected):
            original = self.counts[self.positions[label]]
            df[original.indices] -= 1
            if label in eligible:
                remaining = self.vectorizer.transform(
                    [" ".join(r.profile for _, r in eligible[label])])
                df[remaining.indices] += 1
                replacements.append(remaining - original)
                positions.append(eligible_pos[label])

        weights = TfidfTransformer()
        weights.idf_ = np.where(df > 0, np.log((1 + len(labels)) / (1 + df)) + 1., 0.)
        query_weights = weights.transform(query)
        if query_weights.count_nonzero() == 0:
            return np.zeros(len(labels))
        counts = self.counts[[self.positions[label] for label in labels]]
        if replacements:
            delta = vstack(replacements).tocoo()
            counts = counts + coo_matrix((delta.data, (np.asarray(positions)[delta.row], delta.col)),
                                         shape=counts.shape)
        # Only documents containing a query term can have nonzero cosine similarity.
        candidates = np.unique(counts[:, query.indices].nonzero()[0])
        scores = np.zeros(len(labels))
        if len(candidates):
            scores[candidates] = (weights.transform(counts[candidates]) @ query_weights.T).toarray().ravel()
        return scores


class MenuOracle(AnswerClasses):
    """Generate menus from source-attributed training rows and a closed review registry."""

    def __init__(self, rows: Iterable[ReservoirRow], *, aliases: Mapping,
                 encoder: ProfileScorer | None = None, decoder: DrawDecoder | None = None):
        self.encoder = encoder
        self.decoder = decoder
        super().__init__(aliases)
        ordered = []
        source_ids = set()
        for row in rows:
            if row.split not in {"train", "reference"}:
                raise ValueError("reservoir requires train or disjoint reference rows")
            if not all(isinstance(s, str) and s.strip()
                       for s in (row.group_id, row.category, row.profile)):
                raise ValueError("empty or invalid reservoir group, category or profile")
            source = row.source_id or row.group_id
            if not isinstance(source, str) or not source.strip() or source in source_ids:
                raise ValueError("invalid or repeated reservoir source ID")
            source_ids.add(source)
            ordered.append((self.class_id(row.answer), source, row))
        self._full_catalog = defaultdict(list)
        self._group_classes = defaultdict(set)
        for key, source, row in sorted(ordered, key=lambda record: record[:2]):
            self._full_catalog[key].append((source, row))
            self._group_classes[row.group_id].add(key)
        self._full_tfidf = None

    def _catalog(self, group_id):
        """Remove target-group contributions before profile fitting or retrieval."""
        affected = self._group_classes.get(group_id)
        if not affected:
            return self._full_catalog
        catalog = self._full_catalog.copy()
        for key in affected:
            catalog[key] = [(s, r) for s, r in catalog[key] if r.group_id != group_id]
            if not catalog[key]:
                del catalog[key]
        return catalog

    def _rank(self, catalog, prefix, strategy, group_id):
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("retrieval requires an observed nonempty prefix")
        ids = sorted(catalog)

        if "sbert" in strategy:
            if self.encoder is None:
                raise ValueError("SBERT menu requires a configured SBERT profile scorer")
            profiles = [" ".join(row.profile for _, row in catalog[i]) for i in ids]
            scores = np.asarray(self.encoder.score(prefix, profiles), dtype=float)
        else:
            if self._full_tfidf is None:
                self._full_tfidf = _GroupTfidf(self._full_catalog)
            scores = self._full_tfidf.score(catalog, ids, self._group_classes.get(group_id), prefix)
        if scores.shape != (len(ids),) or not np.isfinite(scores).all():
            raise ValueError("invalid retrieval scores")
        return [key for _, key in sorted(zip(-scores, ids))]

    @staticmethod
    def _mixed(ranked, catalog, count):
        """Alternate high/low rank, preferring previously unrepresented categories."""
        selected = []
        covered = set()
        remaining = ranked.copy()
        for i in range(count):
            direction = remaining[::-1] if i % 2 else remaining
            def categories(key):
                return {row.category for _, row in catalog[key]}
            key = next((c for c in direction if categories(c) - covered), direction[0])
            selected.append(key)
            covered.update(categories(key))
            remaining.remove(key)
        return selected

    def sample(self, *, gold: str, group_id: str, category: str, prefix: str = "",
               strategy: str = "uniform_global", k: int = 4, seed: int = 1,
               explicit: Sequence[str] | None = None,
               explicit_source_ids: Sequence[Sequence[str]] | None = None,
               draw_id: str | None = None) -> Menu:
        """Draw once at the supplied public observation or replay a baseline menu."""
        if type(k) is not int or k < 2 or type(seed) is not int:
            raise ValueError("k must be an integer >= 2 and seed must be an integer")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("missing target group")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown menu strategy {strategy!r}")
        gold_id = self.class_id(gold)
        catalog = self._catalog(group_id)
        count = k - 1
        if strategy == "baseline":
            if isinstance(explicit, (str, bytes)) or explicit is None or len(explicit) != k:
                raise ValueError("baseline requires an explicit ordered menu of length k")
            options = tuple(explicit)
            classes = [self.class_id(option) for option in options]
            if len(set(classes)) != k or gold_id not in classes:
                raise ValueError("baseline requires distinct classes and exactly one gold")
            if any(key not in catalog for key in classes if key != gold_id):
                raise ValueError("baseline distractor outside eligible training reservoir")
            logp = 0.
        else:
            if explicit is not None or explicit_source_ids is not None:
                raise ValueError("explicit menus and provenance require baseline strategy")
            pool = sorted(key for key, entries in catalog.items()
                          if key != gold_id and (strategy != "uniform_category" or
                             any(row.category == category for _, row in entries)))
            if len(pool) < count:
                raise ValueError(f"insufficient eligible distractors: need {count}, have {len(pool)}")
            if not strategy.startswith("uniform_"):
                # Score the eligible corpus before removing gold. The scorer receives
                # neither a gold label nor gold position, even for unseen gold classes.
                pool = [key for key in self._rank(catalog, prefix, strategy, group_id) if key != gold_id]
                width = max(count, (len(pool) + 1) // 2)
                if strategy == "mixed":
                    pool = self._mixed(pool, catalog, count)
                elif strategy.startswith("prefix_"):
                    pool = pool[:count]
                elif strategy.endswith("_easy"):
                    pool = pool[-width:]
                else:
                    pool = pool[:width]
            rng = random.Random(_digest(["acl-menu-v1", seed, group_id, draw_id, strategy]))
            digits = [rng.randrange(len(pool) - i) for i in range(count)]
            gold_position = rng.randrange(k)
            if self.decoder is None:
                indices = select_indices(len(pool), count, digits)
                classes = [pool[i] for i in indices]
                classes.insert(gold_position, gold_id)
            else:
                indices = tuple(self.decoder.draw(len(pool), digits))
                if (indices != select_indices(len(pool), count, digits) or
                        any(type(i) is not int or not 0 <= i < len(pool) for i in indices)):
                    raise ValueError("invalid audited draw decoder result")
                sentinel = len(pool)
                ordered = tuple(self.decoder.insert(indices, sentinel, gold_position))
                if (any(type(i) is not int for i in ordered) or len(ordered) != k or ordered[gold_position] != sentinel or
                        ordered[:gold_position] + ordered[gold_position+1:] != indices):
                    raise ValueError("invalid audited gold insertion")
                classes = [gold_id if i == sentinel else pool[i] for i in ordered]
            options = tuple(classes)
            logp = -math.log(k) - sum(math.log(len(pool) - i) for i in range(count))
        sources = [tuple(source for source, _ in catalog.get(key, ())) for key in classes]
        if explicit_source_ids is not None:
            if len(explicit_source_ids) != k:
                raise ValueError("baseline source ID lists must align with options")
            for i, ids in enumerate(explicit_source_ids):
                if (not isinstance(ids, Sequence) or isinstance(ids, str) or
                        (not ids and classes[i] != gold_id) or
                        not all(isinstance(source, str) for source in ids) or
                        len(set(ids)) != len(ids)):
                    raise ValueError("invalid baseline source ID list")
                if not set(ids).issubset(sources[i]):
                    raise ValueError("baseline source outside eligible training reservoir")
                sources[i] = tuple(ids)
        return Menu(options, tuple(classes), tuple(sources), classes.index(gold_id),
                    strategy, self.equivalence_version, self.equivalence_digest, logp)
