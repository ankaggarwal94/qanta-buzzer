---
phase: quick-1-repo-contract
plan: 01
subsystem: docs
tags: [agents-md, claude-md, agentic-yml, ci, smoke]

requires:
  - phase: none
    provides: n/a
provides:
  - "AGENTS.md canonical repo contract for all coding agents"
  - "Thin CLAUDE.md shim pointing to AGENTS.md"
  - ".agentic.yml machine-readable repo metadata"
  - "scripts/ci.sh pytest wrapper"
  - "scripts/manual-smoke.sh four-stage smoke pipeline wrapper"
affects: [all-agents, ci]

tech-stack:
  added: []
  patterns: ["AGENTS.md as single source of truth, CLAUDE.md as thin shim"]

key-files:
  created: [AGENTS.md, .agentic.yml, scripts/ci.sh, scripts/manual-smoke.sh]
  modified: [CLAUDE.md]

key-decisions:
  - "AGENTS.md is the single canonical contract; CLAUDE.md only adds Claude-specific notes"
  - "ci.sh passes through arguments to pytest for flexibility"
  - "manual-smoke.sh is human-oriented with stage labels, not CI (stages are heavyweight)"

patterns-established:
  - "Repo contract pattern: AGENTS.md canonical, tool-specific files are thin shims"

requirements-completed: [SCAFFOLD-01, SCAFFOLD-02, SCAFFOLD-03, SCAFFOLD-04]

duration: 2min
completed: 2026-03-12
---

# Quick Task 1: Repo Contract Scaffolding Summary

**AGENTS.md as canonical repo contract with 9 sections, CLAUDE.md reduced to 11-line shim, plus .agentic.yml and CI/smoke shell scripts**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-13T00:43:18Z
- **Completed:** 2026-03-13T00:45:04Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Created AGENTS.md with all 9 required sections (overview, setup, architecture, testing, smoke, T5, config, bridge, conventions)
- Reduced CLAUDE.md from 89 lines to 11 lines, referencing AGENTS.md for everything except Claude-specific notes
- Added .agentic.yml with truthful metadata derived from pyproject.toml and actual repo contents
- Created scripts/ci.sh (minimal pytest wrapper) and scripts/manual-smoke.sh (four-stage smoke pipeline)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AGENTS.md and reduce CLAUDE.md to thin shim** - `bfc21d2f` (feat)
2. **Task 2: Create .agentic.yml, scripts/ci.sh, and scripts/manual-smoke.sh** - `cafea1ad` (chore)

## Files Created/Modified
- `AGENTS.md` - Canonical repo contract with 9 sections for all coding agents
- `CLAUDE.md` - Thin shim (11 lines) pointing to AGENTS.md with Claude-specific notes
- `.agentic.yml` - Machine-readable repo metadata (packages, configs, testing, smoke steps)
- `scripts/ci.sh` - Minimal pytest wrapper, exits nonzero on failure
- `scripts/manual-smoke.sh` - Four-stage smoke pipeline with human-readable stage labels

## Decisions Made
- AGENTS.md is the single canonical contract; CLAUDE.md only adds Claude-specific notes (no duplication)
- ci.sh passes through arguments to pytest for flexibility (e.g., `--co`, `-x`, `-k`)
- manual-smoke.sh is for human verification, not CI -- stages are heavyweight ML runs

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 5 files created and committed
- Any future agent (Claude, Copilot, Cursor) can read AGENTS.md for full repo context
- CI pipelines can use scripts/ci.sh as entry point

## Self-Check: PASSED

All 6 files verified present. Both task commits (bfc21d2f, cafea1ad) verified in git log.

---
*Quick Task: 1-repo-contract-scaffolding-agents-md-thin*
*Completed: 2026-03-12*
