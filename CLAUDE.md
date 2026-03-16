# CLAUDE.md

See **AGENTS.md** for the full repo contract: setup, architecture, testing, smoke pipeline, and configuration.

## Claude-specific notes

- `.planning/` is durable project memory; respect STATE.md decisions.
- Prefer narrow verification over broad cargo-cult test runs.
- Do not add dependencies unless required.
- Seeds: use 1, 2, 3 for multi-seed runs.
- NumPy/PyTorch vectorized operations over loops in ML code.
