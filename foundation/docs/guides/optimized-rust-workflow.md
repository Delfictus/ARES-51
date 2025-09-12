---
allowed-tools: Read, Write, Edit, MultiEdit, Task, WebFetch, Bash(cargo:*), Bash(go:*), Bash(npm:*), Bash(deno:*), Bash(fd:*), Bash(rg:*), Bash(jq:*), Bash(gdate:*), Bash(date:*), Bash(pwd:*)
name: "Rust Async Implementation Power-Workflow"
description: "A comprehensive workflow for Rust async development, from context loading to implementation planning and execution."
author: "wcygan"
tags: ["rust", "async", "workflow", "agent", "persona", "map"]
version: "1.0.0"
created_at: "2025-07-14T00:00:00Z"
updated_at: "2025-07-14T00:00:00Z"
---

## Context

- Session ID: !`gdate +%s%N 2>/dev/null || date +%s%N`
- Current directory: !`pwd`
- Project type: !`fd -t f "deno.json|package.json|pom.xml|Cargo.toml|go.mod|build.gradle" -d 2 | head -1 || echo "unknown"`
- Rust projects: !`fd "Cargo\.toml" . | head -5 || echo "No Rust projects found"`
- Async usage: !`rg "tokio|async|futures|await" . --type rust | wc -l | tr -d ' ' || echo "0"`
- Async dependencies: !`rg "tokio|async-std|futures" . -A 1 -B 1 | head -10 || echo "No async dependencies found"`
- Rust edition: !`rg "edition.*=.*\"202" . --type toml | head -3 || echo "No edition info found"`
- Async frameworks: !`rg "actix|warp|axum|tower|hyper" . --type rust --type toml | head -5 || echo "No web frameworks found"`
- Git status: !`git status --porcelain | head -3 || echo "Not a git repository"`
- Existing planning files: !`fd -t f -e md . | rg -i "plan|roadmap|strategy" | head -5 || echo "No existing planning files found"`
- Initiative: $ARGUMENTS

## Your Task

PROCEDURE execute_rust_async_power_workflow():

STEP 1: Initialize Workflow Session

- CREATE session state file: `/tmp/rust-power-workflow-$SESSION_ID.json`
- SET initial state:
  ```json
  {
    "sessionId": "$SESSION_ID",
    "phase": "initialization",
    "context_loaded": false,
    "persona_activated": false,
    "roadmap_generated": false
  }
  ```

STEP 2: Load Rust Async Context

- **Analyze Project**: Analyze Rust project structure, async patterns, and frameworks from the Context section.
- **Prioritize Documentation**: Based on the analysis, prioritize documentation for async runtimes, frameworks, and performance optimization.
- **Fetch Documentation**: Use the WebFetch tool to get comprehensive information from the following sources:
    - **Tokio Runtime**: `https://docs.rs/tokio/latest/tokio/`
    - **Futures Crate**: `https://docs.rs/futures/latest/futures/`
    - **The Async Book**: `https://rust-lang.github.io/async-book/`
    - **Tokio Tutorial**: `https://tokio.rs/tokio/tutorial`
    - **Async Channels and Synchronization**: `https://docs.rs/tokio/latest/tokio/sync/index.html`
- **Synthesize Context**: Organize the loaded context by async programming domains and provide project-specific guidance.
- **Update State**: Update the session state with the loaded context summary.

STEP 3: Activate Implementation Engineer Persona

- **Activate Mindset**: Activate the implementation engineer persona with a focus on the initiative from `$ARGUMENTS`.
- **Analyze Patterns**: Analyze existing codebase patterns for architecture, conventions, testing, and tooling.
- **Decompose Requirements**: Break down requirements into atomic, testable units with clear interfaces.
- **Update State**: Update the session state to reflect the activated persona.

STEP 4: Generate Strategic Roadmap

- **Deploy Sub-agents**: Launch 8 parallel sub-agents for comprehensive analysis and roadmap generation.
    1.  **Scope Analysis Agent**
    2.  **Goals Strategy Agent**
    3.  **Milestone Planning Agent**
    4.  **Risk Assessment Agent**
    5.  **Resource Planning Agent**
    6.  **Timeline Agent**
    7.  **Stakeholder Agent**
    8.  **Success Metrics Agent**
- **Generate Roadmap**: Compile the analysis into a structured roadmap and save it to a markdown file.
- **Update State**: Mark the roadmap as generated in the session state.

STEP 5: Execute Implementation

- **Interactive Development**: Based on the generated roadmap, loaded context, and activated persona, begin the implementation. This is an interactive step driven by the user.

## Roadmap Template Structure

```markdown
# Initiative: {Initiative Name}

## Executive Summary

{One-paragraph overview of initiative and expected outcomes}

## Strategic Goals

### Goal 1: {Goal Name}

- **Objective**: {Clear statement of what we're achieving}
- **Success Metrics**: {Quantifiable measures of success}
- **Timeline**: {Estimated completion timeframe}
- **Priority**: {High/Medium/Low}

### Goal 2: {Goal Name}

...

## Implementation Roadmap

### Phase 1: {Phase Name} ({Timeline})

**Milestone 1.1**: {Milestone Name}

- [ ] {Deliverable/Task}
- [ ] {Success Criteria}
- **Dependencies**: {List any blocking items}
- **Estimated Effort**: {Time/resource estimate}
- **Owner**: {Responsible party}

**Milestone 1.2**: {Milestone Name}
...

### Phase 2: {Phase Name} ({Timeline})

...

## Risk Analysis

| Risk Category | Description        | Impact       | Probability  | Mitigation Strategy   |
| ------------- | ------------------ | ------------ | ------------ | --------------------- |
| Technical     | {Risk description} | High/Med/Low | High/Med/Low | {Mitigation approach} |
| Resource      | {Risk description} | High/Med/Low | High/Med/Low | {Mitigation approach} |
| External      | {Risk description} | High/Med/Low | High/Med/Low | {Mitigation approach} |

## Resource Requirements

- **Human Resources**: {Skills and roles needed}
- **Technical Resources**: {Tools, infrastructure, licenses}
- **Budget**: {Estimated costs by category}
- **Timeline**: {Overall project duration}

## Success Criteria

- [ ] {Measurable outcome 1}
- [ ] {Measurable outcome 2}
- [ ] {Measurable outcome 3}

## Next Steps

1. {Immediate action item with owner}
2. {Next priority with timeline}
3. {Follow-up milestone target}
```

## State Management

- **Session State**: /tmp/rust-power-workflow-$SESSION_ID.json
- **Resumability**: If a previous session exists, it can be loaded to continue or refine the workflow.
- **Cleanup**: Temporary files are cleaned up upon session completion.
