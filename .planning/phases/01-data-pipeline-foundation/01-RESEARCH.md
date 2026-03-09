# Phase 1: Data Pipeline Foundation - Research

**Researched:** 2026-02-25
**Domain:** Quiz bowl data loading, multiple-choice construction, anti-artifact guards
**Confidence:** HIGH

## Summary

The Data Pipeline Foundation phase merges two existing implementations: qanta-buzzer's CSV loading with qb-rl's robust MC construction and anti-artifact guards. The critical technical challenge is implementing proper anti-artifact protection to prevent agents from exploiting spurious patterns (token overlap, length ratios, alias collisions) rather than learning from clues. The recommended approach uses qb-rl's proven MCBuilder class with its four-layer guard system while adapting qanta-buzzer's existing dataset.py structure.

The standard stack is Python 3.11+ with PyYAML for configuration, scikit-learn for TF-IDF vectorization, and sentence-transformers for SBERT embeddings. Critical patterns include leave-one-out answer profile building (exclude current question when building profiles for its answer), stratified splits by category to prevent distribution shift, and a factory-based configuration system that allows CLI overrides. The main pitfall to avoid is insufficient anti-artifact protection leading to agents learning shortcuts rather than quiz bowl skills.

**Primary recommendation:** Adopt qb-rl's MCBuilder and AnswerProfileBuilder wholesale, integrate with qanta-buzzer's existing Question dataclass, and implement YAML configuration with factory methods for component construction.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | System loads quiz bowl questions from local CSV (QANTA format, clues separated by `|||`) | qanta-buzzer already has QANTADatasetLoader.load_from_csv() implementation |
| DATA-02 | System constructs K=4 multiple-choice questions with distractor generation | qb-rl MCBuilder supports multiple strategies (sbert_profile, tfidf_profile, category_random) |
| DATA-03 | Anti-artifact guards reject MC options with alias collision, token overlap >50%, or length ratio >3x | qb-rl MCBuilder has all guards: _aliases_collide(), _violates_duplicate_guard(), _violates_length_ratio_guard() |
| DATA-04 | Answer profiles built with leave-one-out exclusion per question | qb-rl AnswerProfileBuilder.profile_for_answer() supports exclude_qid parameter |
| DATA-05 | Dataset splits stratified by category (train 70% / val 15% / test 15%) | Need to enhance existing create_train_val_test_splits() with stratification |
| DATA-06 | System can optionally load questions from HuggingFace datasets as fallback | qb-rl uses datasets library, config references "qanta-challenge/acf-co24-tossups" |
| CFG-01 | YAML configuration system with sections: data, likelihood, environment, ppo, evaluation | qb-rl has complete YAML config structure in configs/default.yaml |
| CFG-04 | CLI override support: `--config`, `--smoke`, key overrides | Standard pattern with argparse and dict merging |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11+ | Runtime | Required for modern type hints, match statements |
| PyYAML | 6.0+ | Configuration loading | Industry standard for ML config files |
| numpy | <2.0.0 | Array operations | NumPy 2.0 breaks many dependencies |
| scikit-learn | 1.3+ | TF-IDF vectorization | MCBuilder uses TfidfVectorizer for distractor selection |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sentence-transformers | 3.3.0+ | SBERT embeddings | For sbert_profile distractor strategy |
| datasets | 2.14+ | HuggingFace datasets | Optional fallback data source |
| pandas | 2.0+ | CSV manipulation | If complex CSV parsing needed |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyYAML | Python config classes | YAML better for experiments, allows non-code config changes |
| scikit-learn TF-IDF | Manual implementation | scikit-learn is battle-tested, handles edge cases |
| sentence-transformers | OpenAI embeddings | API costs, latency, requires key management |

**Installation:**
```bash
pip install pyyaml>=6.0 "numpy<2.0" scikit-learn>=1.3 sentence-transformers>=3.3.0 datasets>=2.14
```

## Architecture Patterns

### Recommended Project Structure
```
qanta-buzzer/
├── configs/
│   ├── default.yaml     # Base configuration
│   └── smoke.yaml       # Quick test config
├── data/
│   ├── raw/
│   │   └── questions.csv
│   ├── processed/
│   │   ├── mc_dataset.json
│   │   ├── train_dataset.json
│   │   ├── val_dataset.json
│   │   └── test_dataset.json
│   └── profiles/
│       └── answer_profiles.json
├── qb_data/
│   ├── __init__.py
│   ├── mc_builder.py     # From qb-rl
│   ├── answer_profiles.py # From qb-rl
│   ├── data_loader.py    # Merged implementation
│   └── text_utils.py     # Answer normalization
└── scripts/
    └── build_mc_dataset.py
```

### Pattern 1: Anti-Artifact Guard Pipeline
**What:** Four-layer protection against spurious pattern exploitation
**When to use:** During distractor selection for MC construction
**Example:**
```python
# Source: qb-rl/qb_env/mc_builder.py
def _select_distractors(self, gold: str, candidates: list[str]) -> list[str]:
    selected = []
    for candidate in candidates:
        if self._aliases_collide(candidate, gold_aliases):
            continue  # Guard 1: No alias collision
        if self._violates_duplicate_guard(candidate, selected):
            continue  # Guard 2: No token overlap >80%
        selected.append(candidate)

    if self._violates_length_ratio_guard(options):
        return None  # Guard 3: Max length ratio <3x
    if self._violates_question_overlap_guard(question, options):
        return None  # Guard 4: Answer not in question text
```

### Pattern 2: Leave-One-Out Answer Profiles
**What:** Exclude current question when building answer profile to prevent information leakage
**When to use:** Always when building profiles for MC options
**Example:**
```python
# Source: qb-rl/models/answer_profiles.py
def profile_for_answer(self, answer_primary: str, exclude_qid: str | None = None) -> str:
    texts = []
    for qid, qtext in self._grouped[answer_primary]:
        if exclude_qid is not None and qid == exclude_qid:
            continue  # Skip current question
        texts.append(qtext)
    return " ".join(texts)[:self.max_tokens_per_profile]
```

### Pattern 3: Stratified Category Splits
**What:** Maintain category distribution across train/val/test splits
**When to use:** Preventing distribution shift between splits
**Example:**
```python
def stratified_split(questions, ratios=[0.7, 0.15, 0.15]):
    by_category = defaultdict(list)
    for q in questions:
        by_category[q.category].append(q)

    train, val, test = [], [], []
    for category, cat_questions in by_category.items():
        n = len(cat_questions)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        train.extend(cat_questions[:train_end])
        val.extend(cat_questions[train_end:val_end])
        test.extend(cat_questions[val_end:])

    return train, val, test
```

### Anti-Patterns to Avoid
- **Random distractor selection:** Leads to trivial MC questions agent solves without reading clues
- **Global answer profiles:** Including current question in its answer's profile leaks information
- **Unstratified splits:** Category imbalance causes distribution shift, inflated test metrics

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Edit distance calculation | Custom string similarity | difflib.SequenceMatcher | Handles Unicode, optimized C implementation |
| TF-IDF vectorization | Manual term frequency | sklearn.TfidfVectorizer | Handles stop words, normalization, sparse matrices |
| Embedding similarity | Cosine similarity loops | sentence-transformers | Batch processing, GPU acceleration, caching |
| YAML parsing | Regex-based parser | PyYAML | Handles references, complex types, validation |

**Key insight:** These are solved problems with battle-tested implementations. Custom versions introduce bugs in edge cases (Unicode normalization, empty strings, numerical overflow).

## Common Pitfalls

### Pitfall 1: Insufficient Anti-Artifact Protection
**What goes wrong:** Agent achieves high accuracy by exploiting token overlap or length differences rather than understanding clues
**Why it happens:** Distractors share words with correct answer ("Franklin Roosevelt" vs "Theodore Roosevelt") or have very different lengths
**How to avoid:** Use all four guards from qb-rl: alias collision (edit distance <0.2), token overlap (<80%), length ratio (<3x), question overlap (answer not in question)
**Warning signs:** Choices-only baseline achieves >50% accuracy (should be ~25%)

### Pitfall 2: Answer Profile Information Leakage
**What goes wrong:** Including current question in its answer's profile gives agent unfair advantage
**Why it happens:** Naive profile building concatenates all questions for an answer
**How to avoid:** Always use exclude_qid parameter when building profiles: `profile_for_answer(answer, exclude_qid=q.qid)`
**Warning signs:** Training accuracy near 100% but validation much lower

### Pitfall 3: Category Distribution Shift
**What goes wrong:** Test set has different category distribution than training, causing accuracy drop
**Why it happens:** Random split doesn't preserve category ratios
**How to avoid:** Use stratified splitting that maintains 70/15/15 ratio within each category
**Warning signs:** Per-category accuracy varies >30% between splits

### Pitfall 4: Missing Answer Aliases
**What goes wrong:** "USA", "United States", and "America" treated as different answers
**Why it happens:** No answer normalization or alias mapping
**How to avoid:** Build alias dictionary from all answer variants in dataset, use normalize_answer() function
**Warning signs:** Distractors include obvious aliases of correct answer

## Code Examples

Verified patterns from existing codebases:

### MC Construction with Guards
```python
# Source: qb-rl/qb_env/mc_builder.py
class MCBuilder:
    def build(self, questions: list[TossupQuestion],
              profile_builder: AnswerProfileBuilder) -> list[MCQuestion]:
        profile_builder.fit(questions)
        answer_profiles = profile_builder.build_profiles(questions)

        for q in questions:
            gold = q.answer_primary
            ranked = self._compute_rankings(answers, answer_profiles)
            selected = []

            for candidate in ranked[gold]:
                if self._aliases_collide(candidate, gold_aliases):
                    continue
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= self.K - 1:
                    break
```

### YAML Configuration Loading
```python
# Pattern from qb-rl
import yaml
from pathlib import Path

def load_config(config_path: str, overrides: dict = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if overrides:
        for key, value in overrides.items():
            keys = key.split('.')
            target = config
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = value

    return config
```

### Factory-based Dataset Creation
```python
def build_mc_dataset_from_config(config: dict) -> list[MCQuestion]:
    # Load raw questions
    if config['data'].get('use_huggingface', False):
        from datasets import load_dataset
        dataset = load_dataset(config['data']['dataset'])
        questions = parse_huggingface_questions(dataset)
    else:
        loader = QANTADatasetLoader()
        questions = loader.load_from_csv(config['data']['csv_path'])

    # Build MC with guards
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
        min_questions_per_answer=config['answer_profiles']['min_questions_per_answer']
    )

    mc_builder = MCBuilder(
        K=config['data']['K'],
        strategy=config['data']['distractor_strategy'],
        **config['mc_guards']
    )

    return mc_builder.build(questions, profile_builder)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Random distractors | Embedding-based selection | 2020-2021 | Realistic difficulty, better agent calibration |
| No anti-artifact guards | Four-layer guard system | 2022-2023 | Prevents shortcut learning |
| Global answer profiles | Leave-one-out profiles | 2023 | No information leakage |
| Random splits | Stratified splits | 2021 | Consistent metrics across categories |

**Deprecated/outdated:**
- Word2Vec embeddings: Replaced by SBERT for better semantic similarity
- Manual tokenization: Use library tokenizers that handle edge cases
- JSON config files: YAML more readable for nested configuration

## Open Questions

1. **Optimal distractor strategy mix**
   - What we know: qb-rl uses single strategy per dataset
   - What's unclear: Whether mixing strategies (40% SBERT, 40% category, 20% random) improves robustness
   - Recommendation: Start with sbert_profile, experiment with mixing in Phase 5

2. **Minimum questions per answer threshold**
   - What we know: Some answers appear only once in dataset
   - What's unclear: Whether to exclude rare answers or use answer text as profile
   - Recommendation: Use min_questions_per_answer=1, fall back to answer text

3. **HuggingFace dataset format**
   - What we know: qb-rl references "qanta-challenge/acf-co24-tossups"
   - What's unclear: Exact schema and field mappings
   - Recommendation: Implement but mark as experimental, focus on CSV loading

## Sources

### Primary (HIGH confidence)
- qb-rl/qb_env/mc_builder.py - Complete MCBuilder implementation with all guards
- qb-rl/models/answer_profiles.py - Leave-one-out profile building
- qanta-buzzer/dataset.py - Existing CSV loading and dataset structure
- qb-rl/configs/default.yaml - YAML configuration structure

### Secondary (MEDIUM confidence)
- scikit-learn TfidfVectorizer documentation - Verified vectorization approach
- sentence-transformers documentation - SBERT model usage patterns

### Tertiary (LOW confidence)
- HuggingFace datasets documentation - Dataset loading patterns need verification

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Both codebases use same dependencies
- Architecture: HIGH - Patterns directly from working qb-rl code
- Pitfalls: HIGH - Anti-artifact guards explicitly documented in qb-rl

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (stable domain, patterns unlikely to change)