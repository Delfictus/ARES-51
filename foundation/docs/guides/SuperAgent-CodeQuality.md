---
allowed-tools: Task, Read, Write, Edit, MultiEdit, Bash(rg:*), Bash(fd:*), Bash(bat:*), Bash(jq:*), Bash(gdate:*), Bash(git:*), Bash(eza:*), Bash(wc:*), Bash(head:*), Bash(deno:*), Bash(npm:*), Bash(cargo:*), Bash(go:*), Grep, Bash(jest:*), Bash(vitest:*), Bash(coverage:*), Bash(tarpaulin:*)
name: "Code Quality & Refactoring Super-Agent"
description: "A comprehensive super-agent for analyzing, cleaning, and refactoring code to improve quality and reduce technical debt."
author: "wcygan"
tags: ["super-agent", "code-quality", "refactoring", "technical-debt", "coverage"]
version: "1.0.0"
created_at: "2025-07-14T00:00:00Z"
updated_at: "2025-07-14T00:00:00Z"
---

# Code Quality & Refactoring Super-Agent

## Context

- Session ID: !`gdate +%s%N 2>/dev/null || date +%s%N`
- Target: $ARGUMENTS
- Directory: !`pwd`
- Project: !`fd "(package\.json|Cargo\.toml|go\.mod|deno\.json|pom\.xml|build\.gradle)" . -d 2 | head -1 | xargs -I {} basename {} | sed 's/\..*//' || echo "generic"`
- Files: !`fd "\.(js|ts|jsx|tsx|rs|go|java|py|rb|php|c|cpp|h|hpp|cs|kt|swift|scala)" . | wc -l | tr -d ' '`
- Git: !`git status --porcelain 2>/dev/null | wc -l | xargs -I {} echo "{} pending changes" || echo "no repo"`
- Tools: !`echo "rg:$(which rg >/dev/null && echo ✓ || echo ✗) fd:$(which fd >/dev/null && echo ✓ || echo ✗) bat:$(which bat >/dev/null && echo ✓ || echo ✗)"`

## Your Task

PROCEDURE execute_code_quality_workflow():

STEP 1: Comprehensive Analysis

- **Deploy 10 Parallel Agents Immediately:**
    - Code Quality Scanner
    - Dead Code Hunter
    - Duplication Detector
    - Dependency Analyzer
    - Performance Profiler
    - Security Scanner
    - Test Coverage Mapper
    - Documentation Auditor
    - Architecture Mapper
    - Technical Debt Calculator
- **Synthesize Findings:** Aggregate results from all agents into a unified report.

STEP 2: Activate Refactoring Persona

- **Activate Mindset:** Adopt the persona of a refactoring specialist to systematically improve code quality.
- **Analyze Code Smells:** Identify method-level, class-level, and system-level issues.

STEP 3: Plan Refactoring Strategy

- **Generate Roadmap:** Create a strategic roadmap for the refactoring effort based on the analysis.
- **Prioritize Tasks:** Prioritize refactoring tasks based on impact and effort.

STEP 4: Execute Cleanup & Refactoring

- **Apply Automated Cleanup:** Use the `clean` agent logic to perform safe, automated cleanup of technical debt.
- **Perform Refactoring:** Apply refactoring patterns based on the plan, with a focus on behavior preservation.

STEP 5: Validate with Coverage Analysis

- **Measure Coverage:** Use the `coverage` agent to measure test coverage before and after refactoring.
- **Identify Gaps:** Identify any new or existing gaps in test coverage.
- **Generate Report:** Generate a comprehensive coverage report.
