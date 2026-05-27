# Split Provenance

## Fresh Split Gate Fields (v10 section 0.3)

```
FRESH_SPLIT_SEED=789685
FRESH_SPLIT_CREATED_AT=2026-05-26T07:49:26.645424+00:00
FRESH_SPLIT_COMMIT_SHA=a43be19733509c52e0820c44e322044b4304dc18
OLD_SPLIT_PRESERVED_AT=artifacts.pre_v10_freshsplit_20260526T074926Z
THRESHOLDS_FROZEN_AFTER_FRESH_SPLIT=true
THRESHOLD_MANIFEST_SHA256=6e9f0dc2fef71da9984324a7a06d0d9329656429895d8027c695a22aae4dfb85
THRESHOLD_FREEZE_TIMESTAMP=2026-05-26T07:53:23Z
TEST_SPLIT_INSPECTED_POST_FRESH_SPLIT=false
```

## Split Statistics

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 14264 | 69.9% |
| Val   | 3039 | 14.9% |
| Test  | 3104 | 15.2% |
| **Total** | **20407** | **100.0%** |

## Preservation Log

- Old artifacts preserved at: `artifacts.pre_v10_freshsplit_20260526T074926Z`
- Old processed data preserved at: `data/processed.pre_v10_freshsplit_20260526T074926Z`
- Fresh split output: `data/processed/`

## Ratios

- Train ratio: 0.7
- Val ratio: 0.15
- Test ratio: 0.15

## Integrity Notes

- Seed 789685 is NOT 42 (configs/default.yaml data.shuffle_seed) and NOT 13 (configs/default.yaml environment.seed)
- All random generators (random, numpy, torch) seeded before split
- Stratified by category to preserve distribution across splits
- No test-set content inspected or printed during this operation
