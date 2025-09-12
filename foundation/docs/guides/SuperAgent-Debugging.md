---
allowed-tools: Task, Read, Write, Edit, MultiEdit, Bash(rg:*), Bash(fd:*), Bash(bat:*), Bash(jq:*), Bash(gdate:*), Bash(git:*), Bash(eza:*), Bash(wc:*), Bash(head:*), Bash(deno:*), Bash(npm:*), Bash(cargo:*), Bash(go:*), Grep, Bash(dlv:*), Bash(gdb:*), Bash(jdb:*), Bash(ps:*), Bash(lsof:*), Bash(netstat:*), Bash(strace:*), Bash(dtrace:*), Bash(docker:*), Bash(kubectl:*), Bash(jest:*), Bash(vitest:*), Bash(coverage:*), Bash(tarpaulin:*)
name: "Debugging & Troubleshooting Super-Agent"
description: "A super-agent specialized in finding and fixing bugs in your code."
author: "wcygan"
tags: ["super-agent", "debugging", "troubleshooting", "bug-fixing"]
version: "1.0.0"
created_at: "2025-07-14T00:00:00Z"
updated_at: "2025-07-14T00:00:00Z"
---

# Debugging & Troubleshooting Super-Agent

## Context

- Session ID: !`gdate +%s%N`
- Problem: $ARGUMENTS
- Directory: !`pwd`
- System info: !`uname -a`
- Process list (top 10 CPU): !`ps aux | head -11`
- Git status: !`git status --porcelain || echo "Not a git repository"`

## Your Task

PROCEDURE execute_debugging_workflow():

STEP 1: Activate Debugger Persona

- **Activate Mindset:** Adopt a systematic, scientific approach to problem-solving.
- **Parse Request:** Extract the problem description, affected components, and error messages.

STEP 2: Comprehensive Analysis

- **Deploy 10 Parallel Agents:** Use the `UltraDebugAgent` in `analyze` mode to quickly gather information about the codebase.
- **Analyze Coverage:** Use the `coverage` agent to identify areas with low test coverage.

STEP 3: Systematic Investigation

- **Formulate Hypotheses:** Generate theories based on the evidence from the analysis.
- **Test Hypotheses:** Use the appropriate debugging tools (`dlv`, `gdb`, `jdb`, etc.) to test the hypotheses.

STEP 4: Root Cause Identification & Solution

- **Identify Root Cause:** Confirm the root cause with a minimal reproduction.
- **Implement Fix:** Implement a fix with tests to ensure the issue is resolved and no regressions are introduced.

STEP 5: Verification and Prevention

- **Add Regression Tests:** Add tests to prevent the bug from recurring.
- **Document Lessons Learned:** Document the bug, the root cause, and the solution.
