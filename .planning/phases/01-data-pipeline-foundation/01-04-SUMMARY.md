---
phase: 01-data-pipeline-foundation
plan: 04
subsystem: data-pipeline
tags: [stratified-splits, huggingface-loader, dataset-balancing]
dependency_graph:
  requires: [01-01]
  provides: [stratified-splits, huggingface-fallback]
  affects: [training-pipeline, evaluation]
tech_stack:
  added: [datasets-library-optional]
  patterns: [stratified-sampling, fallback-loading]
key_files:
  created: [qb_data/dataset_splits.py, qb_data/huggingface_loader.py]
  modified: [qb_data/text_utils.py]
decisions:
  - Use category-based stratification for train/val/test splits
  - Implement HuggingFace as optional fallback data source
  - Fix field name inconsistencies to match TossupQuestion dataclass
metrics:
  duration: 433s
  tasks_completed: 2
  files_changed: 3
  test_coverage: verified
key_decisions:
  - Maintain exact 70/15/15 ratios per category where possible
  - Handle small categories gracefully with minimum guarantees
  - Support multiple HuggingFace dataset formats with field mapping
completed_date: 2026-02-25T09:27:24Z
---

# Phase 01 Plan 04: Dataset Splitting and HuggingFace Loader Summary

**One-liner:** Stratified category-preserving splits with optional HuggingFace dataset fallback

## What Was Built

### Stratified Dataset Splitter (`qb_data/dataset_splits.py`)
- `create_stratified_splits()`: Maintains category distribution across 70/15/15 train/val/test splits
- Groups questions by category, splits each group independently
- Handles small categories with minimum guarantees (at least 1 in train if possible)
- Deterministic splits with seed parameter for reproducibility
- `save_splits()`: JSON serialization with metadata and category distributions

### HuggingFace Dataset Loader (`qb_data/huggingface_loader.py`)
- `load_from_huggingface()`: Loads quiz bowl datasets from HuggingFace Hub
- Supports known datasets: qanta-challenge/acf-co24-tossups, qanta-challenge/qanta25-playground
- `parse_huggingface_row()`: Flexible field mapping for different dataset formats
- Handles clue separation (|||, lists, or sentence splitting)
- `try_huggingface_fallback()`: Automatic fallback when CSV files missing

### Text Utilities Enhancement
- Added `tokenize_text()` function for consistent text tokenization

## Key Technical Decisions

1. **Category-Based Stratification**: Each category split independently to preserve distribution
2. **Deterministic Shuffling**: Category-specific seeds ensure reproducible splits
3. **Small Category Handling**: Categories with 1-2 questions handled specially
4. **Field Name Flexibility**: Support multiple naming conventions (Text/question, Answer/answer)
5. **Optional Dependencies**: HuggingFace datasets library is optional with graceful fallback

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing tokenize_text function**
- **Found during:** Task 2
- **Issue:** huggingface_loader imported tokenize_text which didn't exist in text_utils
- **Fix:** Added tokenize_text() function to text_utils.py
- **Files modified:** qb_data/text_utils.py
- **Commit:** 5421339

**2. [Rule 1 - Bug] Fixed TossupQuestion field name mismatches**
- **Found during:** Task 2
- **Issue:** Plan used different field names than actual dataclass (tokenized_question vs tokens, answer vs answer_primary)
- **Fix:** Updated both new files to match existing TossupQuestion structure
- **Files modified:** qb_data/dataset_splits.py, qb_data/huggingface_loader.py
- **Commit:** 5421339

## Verification Results

✅ All imports successful
✅ Stratified splits maintain ratios (tested with 10 questions: 7/1/2 split)
✅ Category distribution preserved across splits
✅ HuggingFace loader handles missing fields gracefully

## Dependencies Established

- Provides: Stratified splitting utility for consistent train/val/test splits
- Provides: Optional HuggingFace data loading for missing CSV files
- Depends on: TossupQuestion dataclass from 01-01
- Depends on: text_utils normalization from 01-01

## Next Phase Readiness

With stratified splitting complete, the data pipeline can now:
- Create reproducible train/val/test splits
- Load from HuggingFace if local data unavailable
- Maintain category balance for fair evaluation

Ready for Plan 01-05: Evaluation and Testing Infrastructure

## Self-Check: PASSED

Files verified:
- ✅ qb_data/dataset_splits.py exists
- ✅ qb_data/huggingface_loader.py exists
- ✅ qb_data/text_utils.py modified

Commits verified:
- ✅ fbb1e4a: stratified splitting utility
- ✅ 5421339: HuggingFace loader with fixes