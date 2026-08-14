"""Dependency-contract regressions for PR #30 adapter loading."""
from __future__ import annotations

import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SENTENCE_TRANSFORMERS_FLOOR = "sentence-transformers>=2.3.0"
TORCH_FLOOR = "torch>=2.6.0"


def test_sentence_transformers_floor_supports_trust_remote_code() -> None:
    """Keep both install manifests on the first compatible constructor API."""
    # SentenceTransformer.__init__ first accepted trust_remote_code in v2.3.0;
    # adapter_build passes that keyword explicitly to keep remote code disabled.
    pyproject = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_requirements = [
        requirement
        for requirement in pyproject["project"]["dependencies"]
        if requirement.startswith("sentence-transformers")
    ]
    assert project_requirements == [SENTENCE_TRANSFORMERS_FLOOR]

    minimum_requirements = [
        line.strip()
        for line in (REPO / "requirements-min.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    sentence_transformers_requirements = [
        requirement
        for requirement in minimum_requirements
        if requirement.startswith("sentence-transformers")
    ]
    assert sentence_transformers_requirements == [
        f"{SENTENCE_TRANSFORMERS_FLOOR},<6"
    ]


def test_torch_floor_covers_cve_2025_32434() -> None:
    """Keep both install manifests at or above the torch.load RCE fix."""
    # torch.load-based checkpoint loading; CVE-2025-32434 (torch.load RCE) is
    # fixed in torch 2.6.0, so relaxing either floor silently re-admits the
    # vulnerable range. The Modal image's matching pin is regression-tested
    # separately (test_modal_runner_facade.py::test_v5_image_pins_torch_at_cve_floor).
    pyproject = tomllib.loads(
        (REPO / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_requirements = [
        requirement
        for requirement in pyproject["project"]["dependencies"]
        if requirement.startswith("torch")
    ]
    assert project_requirements == [TORCH_FLOOR]

    minimum_requirements = [
        line.strip()
        for line in (REPO / "requirements-min.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    torch_requirements = [
        requirement
        for requirement in minimum_requirements
        if requirement.startswith("torch")
    ]
    assert torch_requirements == ["torch>=2.6,<3"]
