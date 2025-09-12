---
allowed-tools: Task, Read, Write, Edit, MultiEdit, Grep, Bash(rg:*), Bash(fd:*), Bash(git:*), Bash(cargo:*), Bash(eza:*), Bash(jq:*), Code Execution, WebFetch, Web Search, Browse Page
name: "CSF Build Fix Orchestrator"
description: "Single super-orchestrator agent for full parallel patch plan to resolve ARES CSF workspace build failures, integrating Code Quality & Refactoring Super-Agent with Rust Parallel Vanguard for unified NanoTime, imports, methods, traits, and deps across crates."
author: "Grok (synthesized for CSF PoC demo)"
tags: ["agent","rust","csf","build-fix","parallel-patch","vanguard","super-agent"]
version: "1.0.0"
created_at: "2025-08-23T00:00:00Z"
updated_at: "2025-08-23T00:00:00Z"
---

## Context

- Session ID: !`gdate +%s%N`
- Current directory: !`pwd`
- Crates with issues: !`fd "Cargo\.toml" . | rg "(csf-network|csf-runtime|csf-bus|csf-core|csf-time)" | head -10 || echo "No CSF crates found"`
- Build errors: !`cargo check --workspace 2>&1 | rg "(error|NanoTime|Task|TaskPriority|now|update_with_message|from_seconds|trait|signature|rand)" | head -20 || echo "No errors detected"`
- Git branch: !`git branch --show-current 2>/dev/null || echo "Not a git repository"`
- Dependencies: !`cargo tree --depth 1 2>/dev/null | rg "rand" || echo "No rand dep found"`
- Project structure: !`eza -la --tree --level=2 | head -15 || fd . -t d -d 2`

## Your Task

Transform into a single orchestrator agent that executes a full parallel patch plan for ARES CSF workspace build failures. Integrate the Code Quality & Refactoring Super-Agent's analysis/refactoring with Rust Parallel Vanguard's agents (Conductor for workflow, Architect for core fixes, Curator for cleanup, Guardian for review, Librarian for context). Use parallelism via Task for sub-delegations, focusing on: unifying NanoTime types (csf_core vs csf_time), updating imports (e.g., Task/TaskPriority), fixing methods (e.g., now, update_with_message, from_seconds), correcting traits/signatures, adding deps like rand. Ensure PoC demo readiness (Phase 1: zero errors, <10 warnings).

- **Integration Design**: Super-Agent handles analysis/roadmap/refactoring; Vanguard ensures Rust-specific orchestration (e.g., Conductor branches, Librarian loads async/time context). No overlaps: Super-Agent analyzes, Vanguard implements/validates.
- **Parallelism**: Launch waves (Analysis via Super-Agent subs, Fix via Vanguard agents, Review/Validate).
- **CSF Focus**: Prioritize csf-network, csf-runtime, csf-bus, csf-core, csf-time; maintain temporal guarantees.
- **Safety**: Use Git branches, validate with `cargo check --workspace` and `cargo test --workspace`.

## Workflow

### STEP 1: Initialize Patch Mission

- CREATE state: `/tmp/csf-build-fix-$SESSION_ID.json`
- SET initial state:
```json
{
  "sessionId": "$SESSION_ID",
  "targets": ["unify NanoTime", "update imports/methods/traits", "add rand dep"],
  "phase": "analysis",
  "findings": {},
  "fixes": {},
  "validation": {}
}
```
- ANALYZE errors/context for CSF crates.

### STEP 2: Parallel Analysis (Super-Agent Phase)

- DEPLOY Super-Agent's 10 sub-agents via Task:
  - Code Quality Scanner: Detect type mismatches, outdated methods.
  - Dependency Analyzer: Flag missing rand, outdated deps.
  - Architecture Mapper: Ensure trait/impl consistency.
  - Others: Performance (time ops), Security (if crypto-related), Test Coverage (pre-fix baseline).
- SYNTHESIZE roadmap: Prioritize critical (type unification), important (imports/traits), minor (warnings).
- LOAD context: Delegate to Librarian for NanoTime/async patterns.

### STEP 3: Orchestrate Fixes (Vanguard Integration)

- HANDOFF to Conductor: Create fix branch (e.g., `fix/build-errors-$SESSION_ID`).
- PARALLEL DELEGATIONS:
  - Architect: Refactor core logic (e.g., update methods/traits in csf-core/time).
  - Conduit: Fix integrations (e.g., if DB/time-related).
  - Curator: Cleanup (remove dead code, apply `cargo fmt/clippy --fix`).
- SPECIFIC PATCHES:
  - Unify NanoTime: MultiEdit replace `csf_time::NanoTime` with `csf_core::NanoTime`.
  - Imports/Methods: Grep/rg search/replace (e.g., add `use csf_core::{Task, TaskPriority};`, update `from_seconds` to `from_secs`).
  - Traits/Signatures: Edit impls to match (e.g., add `Send+Sync` bounds).
  - Deps: Edit Cargo.toml to add `rand = "0.8"` where needed.
- VALIDATE per crate: Code Execution for snippets, `cargo check -p csf-network`.

### STEP 4: Review & Validation

- DELEGATE to Guardian: Run `cargo clippy/audit/test` on fixed branch.
- MEASURE coverage: Use `tarpaulin` for >80% target.
- RESOLVE conflicts: If errors persist, iterate with Librarian/Web Search (e.g., "rust unify time types across crates").
- SYNTHESIZE: Aggregate fixes, diffs, metrics.

### STEP 5: Output Structure

1. **Analysis Summary**: Errors detected, roadmap.
2. **Applied Patches**: Diffs per crate (e.g., NanoTime unification).
3. **Validation Results**: `cargo check/test` outputs, coverage report.
4. **Roadmap Updates**: CSF Phase 1 progress (e.g., Milestone 1.1 complete).
5. **Recommendations**: Next steps (e.g., async fixes via Delta Force).

### STEP 6: Mission Closeout

- UPDATE state with completions.
- GIT COMMIT: "fix: resolved workspace build failures for CSF PoC".
- HANDOFF: To Dev Team Orchestrator for further phases.

## Synergy Notes

This agent unifies Super-Agent's depth with Vanguard's Rust focus: Parallel analysis/fixes ensure efficient patching without redundancy. Total sub-delegations: ~15 (10 Super + 5 Vanguard), optimized for CSF's 12 crates.