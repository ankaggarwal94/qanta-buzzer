"""Generation grammar regressions; no weights, GPU, or evaluator labels."""
from __future__ import annotations

import json
import re

import pytest

from scripts import jane_output_constraints as constrained


def completion(answer, confidence=0.9, status='answer'):
    return json.dumps({'answer': answer, 'confidence': confidence, 'status': status},
                      ensure_ascii=False, separators=(',', ':'))


@pytest.mark.parametrize('fmt,raw', [
    ('mc', completion('A')), ('mc', completion('D', 1)),
    ('mc', completion('B', 0)), ('mc', completion('C', 0.123456)),
    ('oe', completion('René Descartes')), ('oe', completion('The "Mona Lisa"')),
    ('oe', completion('C:\\Temp')), ('oe', completion('α and β')),
    ('oe', completion('A' * 96)), ('oe', completion('A' * 97)),
    ('mc', completion(None, None, 'abstain')),
    ('oe', completion(None, None, 'abstain')),
])
def test_valid_json_completions_are_accepted(fmt, raw):
    assert re.fullmatch(constrained.output_regex(fmt), raw)
    parser = constrained.make_parser(fmt)
    # TokenEnforcer supplies the actual vocabulary alphabet in production.
    from lmformatenforcer import CharacterLevelParserConfig
    parser.config = CharacterLevelParserConfig(alphabet=parser.config.alphabet + raw)
    for char in raw:
        assert char in parser.get_allowed_characters()
        parser = parser.add_character(char)
    assert parser.can_end()
    assert parser.get_allowed_characters() == ''


@pytest.mark.parametrize('fmt,raw', [
    ('mc', completion('Z')), ('mc', completion('Paris')),
    ('mc', completion('a')), ('mc', completion('AB')),
    ('mc', completion('A', 1.01)), ('mc', completion('A', -0.1)),
    ('mc', completion('A', True)), ('mc', completion('A', 0.1234567)),
    ('mc', completion('A', None)), ('mc', completion(None, .5)),
    ('mc', completion('A', .5, 'abstain')),
    ('mc', completion(None, .5, 'abstain')),
    ('oe', completion('')), ('oe', completion(' ')), ('oe', completion('\u00a0')),
    ('oe', completion(' A')),
    ('oe', '{"answer":"A","cofidence":0.9,"status":"answer"}'),
    ('oe', completion('A') + ' explanation'),
    ('oe', completion('A')[:-1]),
    ('oe', completion('A').replace('"confidence":0.9', '"confidence":NaN')),
])
def test_illegal_completions_cannot_be_generated(fmt, raw):
    assert re.fullmatch(constrained.output_regex(fmt), raw) is None
    parser = constrained.make_parser(fmt)
    for char in raw:
        if char not in parser.get_allowed_characters():
            return
        parser = parser.add_character(char)
    assert not parser.can_end()


def test_eos_impossible_until_whole_json_is_complete():
    raw = completion('B', .7)
    parser = constrained.make_parser('mc')
    for char in raw:
        assert not parser.can_end()
        parser = parser.add_character(char)
    assert parser.can_end()


def test_provenance_binds_both_grammars_and_dependency():
    provenance = constrained.constraint_provenance()
    assert provenance['package'] == 'lm-format-enforcer'
    assert provenance['package_version'] == '0.11.3'
    assert provenance['schema_version'] == 'jane-constrained-json-v1'
    assert provenance['max_oe_characters'] is None
    assert provenance['confidence_decimal_places'] == 6
    assert provenance['posthoc_repair'] is False
    for fmt in ['mc', 'oe']:
        assert provenance['grammars'][fmt]['regex'] == constrained.output_regex(fmt)
        assert len(provenance['grammars'][fmt]['sha256']) == 64


def test_unsupported_format_fails_closed():
    with pytest.raises(ValueError, match='format'):
        constrained.output_regex('unknown')


def test_dependency_version_mismatch_fails_before_integration_import(monkeypatch):
    monkeypatch.setattr(constrained, 'version', lambda package: 'unreviewed')
    with pytest.raises(ValueError, match='must be pinned'):
        constrained.verify_dependencies()


def test_tokenizer_and_model_eos_must_agree(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(constrained, 'verify_dependencies', lambda: None)
    for eos in [None, True, [151645], 151644]:
        with pytest.raises(ValueError, match='EOS'):
            constrained.OutputConstraints(SimpleNamespace(eos_token_id=eos), {151645, 151643})


def test_completion_checker_requires_eos_and_never_modifies_text():
    raw = completion('A')
    constrained.validate_constrained_completion(raw, 'mc', finished_eos=True)
    with pytest.raises(ValueError, match='without EOS'):
        constrained.validate_constrained_completion(raw, 'mc', finished_eos=False)
    with pytest.raises(ValueError, match='grammar'):
        constrained.validate_constrained_completion(raw + '\n', 'mc', finished_eos=True)


@pytest.mark.parametrize('fmt,raw', [
    ('mc', completion('A')),
    ('oe', completion('René "Descartes"')),
    ('oe', completion(None, None, 'abstain')),
    ('oe', completion('A' * 96)), ('oe', completion('A' * 97)),
])
def test_cached_alphabet_adapter_preserves_every_transition(fmt, raw):
    from lmformatenforcer import CharacterLevelParserConfig
    original = constrained.make_parser(fmt)
    adapted = constrained.cached_alphabet_parser(constrained.make_parser(fmt))
    cfg = CharacterLevelParserConfig(alphabet=original.config.alphabet + raw)
    original.config = cfg
    adapted.config = cfg
    for char in raw:
        assert adapted.get_allowed_characters() == frozenset(original.get_allowed_characters())
        assert adapted.can_end() == original.can_end()
        assert adapted.cache_key() == original.cache_key()
        # An illegal transition has the same library state too; generation must
        # never select it because it is absent from the allowed character set.
        forbidden = next(c for c in '\x00~ZX' if c not in original.get_allowed_characters())
        bad_original = original.add_character(forbidden)
        bad_adapted = adapted.add_character(forbidden)
        assert bad_adapted.get_allowed_characters() == frozenset(bad_original.get_allowed_characters())
        assert bad_adapted.can_end() == bad_original.can_end()
        original = original.add_character(char)
        adapted = adapted.add_character(char)
    assert adapted.can_end() and not adapted.get_allowed_characters()


def test_cached_alphabet_rejects_changes_after_mask_computation():
    from lmformatenforcer import CharacterLevelParserConfig
    parser = constrained.cached_alphabet_parser(constrained.make_parser('oe'))
    for char in '{"answer":"':
        parser = parser.add_character(char)
    assert 'α' not in parser.get_allowed_characters()
    with pytest.raises(ValueError, match='before computing masks'):
        parser.config = CharacterLevelParserConfig(alphabet=parser.config.alphabet + 'α')
