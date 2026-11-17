"""Prospective token masking for the paired pilot's strict JSON interface.

The grammar is an experimental intervention, not a response repair.  It fixes
field order and limits confidence to six decimal places. OE answer length is
bounded only by the generation token cap. Literal Unicode and escaped quote/backslash/slash are accepted;
control-character and Unicode escape spellings are excluded.  No labels enter
this module.  The pinned RegexParser avoids JsonSchemaParser's unsupported
numeric minimum/maximum constraints.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import re
from typing import Any


LMFE_VERSION = '0.11.3'
INTEREGULAR_VERSION = '0.3.3'
SCHEMA_VERSION = 'jane-constrained-json-v1'
MAX_OE_CHARACTERS = None
CONFIDENCE_DECIMAL_PLACES = 6


def output_regex(fmt: str) -> str:
    """Return an exact, anchored-by-the-parser JSON completion language."""
    if fmt not in {'mc', 'oe'}:
        raise ValueError('constraint format must be mc or oe')
    confidence = r'(0(\.[0-9]{1,6})?|1(\.0{1,6})?)'
    if fmt == 'mc':
        answer = r'"[ABCD]"'
    else:
        # All Unicode whitespace recognized by Python str.isspace is excluded
        # initially, so every accepted string passes the unchanged strict parser.
        non_ascii_whitespace = '\u00a0\u1680' + ''.join(chr(code) for code in range(0x2000, 0x200b)) + '\u2028\u2029\u202f\u205f\u3000'
        first = r'([^"\\\x00-\x20\x7f-\x9f' + non_ascii_whitespace + r']|\\["\\/])'
        subsequent = r'([^"\\\x00-\x1f]|\\["\\/])'
        answer = '"' + first + subsequent + '*"'
    answered = r'\{"answer":' + answer + r',"confidence":' + confidence + r',"status":"answer"\}'
    abstained = r'\{"answer":null,"confidence":null,"status":"abstain"\}'
    return '(' + answered + '|' + abstained + ')'


def verify_dependencies() -> None:
    """Refuse unreviewed constraint implementations before any GPU inference."""
    for package, expected in [('lm-format-enforcer', LMFE_VERSION),
                              ('interegular', INTEREGULAR_VERSION)]:
        actual = version(package)
        if actual != expected:
            raise ValueError(f'{package} must be pinned to {expected}; found {actual}')


def make_parser(fmt: str):
    """Build the pinned finite-state character parser without importing torch."""
    from lmformatenforcer import RegexParser
    return RegexParser(output_regex(fmt))


def cached_alphabet_parser(inner):
    """Adapt the exact character language to cached membership sets.

    The pinned TokenEnforcer intersects each vocabulary-tree node with the
    allowed-character iterable. Its RegexParser supplies a long Unicode string;
    repeatedly scanning that string costs tens of seconds at each free-text
    state. A frozenset has exactly the same members and supports efficient
    intersections. Grammar transitions, final states, and token caches are
    unchanged. This is a custom CharacterLevelParser, not an upstream patch.
    """
    from lmformatenforcer import CharacterLevelParser

    class CachedAlphabetParser(CharacterLevelParser):
        def __init__(self, parser, cache=None):
            self.inner = parser
            self.alphabet_cache = {} if cache is None else cache

        @property
        def config(self):
            return self.inner.config

        @config.setter
        def config(self, value):
            if self.alphabet_cache:
                raise ValueError("configure tokenizer alphabet before computing masks")
            self.inner.config = value

        def get_allowed_characters(self):
            key = self.inner.cache_key()
            if key not in self.alphabet_cache:
                self.alphabet_cache[key] = frozenset(self.inner.get_allowed_characters())
            return self.alphabet_cache[key]

        def add_character(self, char):
            return type(self)(self.inner.add_character(char), self.alphabet_cache)

        def can_end(self):
            return self.inner.can_end()

        def cache_key(self):
            return self.inner.cache_key()

    return CachedAlphabetParser(inner)


def constraint_provenance() -> dict[str, Any]:
    """Describe exact imposed output restrictions, independently of prompts."""
    return {
        'schema_version': SCHEMA_VERSION,
        'method': 'prospective_regex_token_masking_transformers_prefix_allowed_tokens_fn',
        'package': 'lm-format-enforcer', 'package_version': LMFE_VERSION,
        'regex_package': 'interegular', 'regex_package_version': INTEREGULAR_VERSION,
        'character_parser_adapter': 'cached_frozenset_alphabet_exact_membership',
        'field_order': ['answer', 'confidence', 'status'],
        'confidence_decimal_places': CONFIDENCE_DECIMAL_PLACES,
        'max_oe_characters': MAX_OE_CHARACTERS,
        'oe_character_policy': 'nonwhitespace first character; literal Unicode; escaped quote/backslash/slash; no control or Unicode escapes',
        'whitespace_policy': 'compact JSON; no whitespace outside answer strings',
        'eos_policy': 'tokenizer EOS only, permitted after complete grammar; required to complete trace',
        'posthoc_repair': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'grammars': {fmt: {'regex': output_regex(fmt),
                          'sha256': hashlib.sha256(output_regex(fmt).encode()).hexdigest()}
                     for fmt in ('mc', 'oe')},
    }


def validate_constrained_completion(raw: str, fmt: str, *, finished_eos: bool) -> None:
    """Fail closed on mask failures or truncation; never edit the raw response."""
    if not finished_eos:
        raise ValueError('constrained completion reached token cap without EOS')
    if re.fullmatch(output_regex(fmt), raw) is None:
        raise ValueError('raw completion violates the frozen output grammar')


class OutputConstraints:
    """Reuse vocabulary/FSM caches while resetting per-batch prefix histories.

    This calls the official Transformers integration.  Only one generate call
    may be active for an instance. The cache reset uses the reviewed 0.11.3
    TokenEnforcer interface, hence the mandatory version check. Completed rows
    still pass the grammar and EOS validator outside the library.
    """

    def __init__(self, tokenizer, eos_ids: set[int]):
        verify_dependencies()
        if (isinstance(tokenizer.eos_token_id, bool)
                or not isinstance(tokenizer.eos_token_id, int)
                or tokenizer.eos_token_id not in eos_ids):
            raise ValueError('constraint tokenizer EOS must be a model EOS token')
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
        self._functions = {
            fmt: build_transformers_prefix_allowed_tokens_fn(
                tokenizer_data, cached_alphabet_parser(make_parser(fmt)))
            for fmt in ('mc', 'oe')
        }

    def for_batch(self, formats: list[str]):
        """Dispatch each row to its format-specific token mask and reject extras."""
        if not formats or any(fmt not in self._functions for fmt in formats):
            raise ValueError('batch constraints require supported formats')
        for function in self._functions.values():
            function.token_enforcer.prefix_states.clear()
        row_functions = [self._functions[fmt] for fmt in formats]

        def allowed_tokens(batch_id, token_ids):
            if not 0 <= batch_id < len(row_functions):
                raise ValueError('constraint batch row out of bounds')
            return row_functions[batch_id](batch_id, token_ids)

        return allowed_tokens
