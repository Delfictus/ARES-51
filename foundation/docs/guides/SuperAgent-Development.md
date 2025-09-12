---
allowed-tools: Task, Read, Write, Edit, MultiEdit, Bash(rg:*), Bash(fd:*), Bash(bat:*), Bash(jq:*), Bash(gdate:*), Bash(git:*), Bash(eza:*), Bash(wc:*), Bash(head:*), Bash(deno:*), Bash(npm:*), Bash(cargo:*), Bash(go:*), WebFetch, Bash(docker:*), Bash(kubectl:*), Bash(curl:*)
name: "Development & Implementation Super-Agent"
description: "A super-agent for planning and implementing new features, from roadmap generation to production-ready code."
author: "wcygan"
tags: ["super-agent", "development", "implementation", "planning", "rust", "async"]
version: "1.0.0"
created_at: "2025-07-14T00:00:00Z"
updated_at: "2025-07-14T00:00:00Z"
---

# Development & Implementation Super-Agent

## Context

- Session ID: !`gdate +%s%N 2>/dev/null || date +%s%N`
- Initiative: $ARGUMENTS
- Directory: !`pwd`
- Project Type: !`fd -t f "deno.json|package.json|pom.xml|Cargo.toml|go.mod|build.gradle" -d 2 | head -1 || echo "unknown"`
- Existing planning files: !`fd -t f -e md . | rg -i "plan|roadmap|strategy" | head -5 || echo "No existing planning files found"`

## Your Task

PROCEDURE execute_development_workflow():

STEP 1: Generate Strategic Roadmap

- **Deploy 8 Parallel Sub-agents:**
    - Scope Analysis Agent
    - Goals Strategy Agent
    - Milestone Planning Agent
    - Risk Assessment Agent
    - Resource Planning Agent
    - Timeline Agent
    - Stakeholder Agent
    - Success Metrics Agent
- **Generate Roadmap:** Compile the analysis into a structured roadmap.

STEP 2: Load Development Context

- **Analyze Project:** Analyze the project structure and technology stack.
- **Fetch Documentation:** Fetch relevant documentation for the languages and frameworks being used (e.g., Rust async context).

STEP 3: Activate Implementation Persona

- **Activate Mindset:** Adopt the persona of an implementation engineer to write production-ready code.
- **Decompose Requirements:** Break down the roadmap into atomic, testable units.

STEP 4: Implement Features

- **Write Code:** Implement the features according to the roadmap and best practices.
- **Integrate Services:** Use the `integrate` agent logic to handle integrations with other services, APIs, and databases.
- **Apply Quality Gates:** Ensure code coverage, static analysis, and security standards are met.
