# CLAUDE.md

See **AGENTS.md** for the full repo contract: setup, architecture, testing, smoke pipeline, and configuration.

## Claude-specific notes

- `.planning/` is durable project memory; respect STATE.md decisions.
- Prefer narrow verification over broad cargo-cult test runs.
- Do not add dependencies unless required.
- Seeds: use 1, 2, 3 for multi-seed runs.
- NumPy/PyTorch vectorized operations over loops in ML code.

## Correctless

This project uses Correctless for structured development.
Read .correctless/AGENT_CONTEXT.md before starting any work.
Do NOT Read AGENT_CONTEXT.md from the project root — it may be stale or absent.
Available commands: /csetup, /cspec, /creview, /cmodel, /creview-spec, /ctdd, /cverify, /caudit, /cupdate-arch, /cdocs, /cpostmortem, /cdevadv, /credteam, /crefactor, /cpr-review, /ccontribute, /cmaintain, /cstatus, /csummary, /cmetrics, /cdebug, /chelp, /cwtf, /cquick, /crelease, /cexplain, /cauto, /carchitect, /cmodelupgrade

## Correctless Learnings
<!-- Auto-updated by Correctless workflow. Do not edit above this line. -->
