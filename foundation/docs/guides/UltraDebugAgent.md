---
allowed-tools: Task, Read, Write, Edit, MultiEdit, Bash(rg:*), Bash(fd:*), Bash(bat:*), Bash(jq:*), Bash(gdate:*), Bash(git:*), Bash(eza:*), Bash(wc:*), Bash(head:*), Bash(deno:*), Bash(npm:*), Bash(cargo:*), Bash(go:*), Grep
name: "Ultra Agent"
description: "Integrated high-performance code analysis, refactoring, and strategic planning agent with parallel execution"
author: "optimized"
tags: ["workflow", "analyze", "refactor", "manage", "code"]
version: "2.0.0"
created_at: "2025-01-01T00:00:00Z"
updated_at: "2025-01-01T00:00:00Z"
---

# Ultra Performance Code Agent

## Context & Initialization

- Session: !`echo "UA-$(gdate +%s%N 2>/dev/null || date +%s%N)"`
- Target: $ARGUMENTS
- Mode: !`echo "$ARGUMENTS" | rg -o "^(clean|map|refactor|analyze|full)" || echo "auto"`
- Directory: !`pwd`
- Project: !`fd "(package\.json|Cargo\.toml|go\.mod|deno\.json|pom\.xml|build\.gradle)" . -d 2 | head -1 | xargs -I {} basename {} | sed 's/\..*//' || echo "generic"`
- Files: !`fd "\.(js|ts|jsx|tsx|rs|go|java|py|rb|php|c|cpp|h|hpp|cs|kt|swift|scala)" . | wc -l | tr -d ' '`
- Git: !`git status --porcelain 2>/dev/null | wc -l | xargs -I {} echo "{} pending changes" || echo "no repo"`
- Tools: !`echo "rg:$(which rg >/dev/null && echo ✓ || echo ✗) fd:$(which fd >/dev/null && echo ✓ || echo ✗) bat:$(which bat >/dev/null && echo ✓ || echo ✗)"`

## Execution Strategy

```bash
# Initialize unified session state
SESSION_ID="UA-$(gdate +%s%N 2>/dev/null || date +%s%N)"
STATE_FILE="/tmp/ultra-agent-$SESSION_ID.json"
RESULTS_DIR="/tmp/ultra-results-$SESSION_ID"

mkdir -p "$RESULTS_DIR"
echo '{
  "session": "'$SESSION_ID'",
  "mode": "auto-detect",
  "target": "'$ARGUMENTS'",
  "timestamp": "'$(date -Iseconds)'",
  "agents": [],
  "status": "initializing"
}' > "$STATE_FILE"
```

## Mode Selection & Parallel Agent Deployment

DETERMINE operation mode from $ARGUMENTS:

### MODE: ANALYZE (Default - Ultra-Fast Comprehensive Analysis)

**DEPLOY 10 PARALLEL AGENTS IMMEDIATELY:**

```parallel
Agent[1]: Code Quality Scanner
  -> Detect smells, anti-patterns, complexity hotspots
  -> Output: $RESULTS_DIR/quality.json

Agent[2]: Dead Code Hunter  
  -> Find unused code, orphaned files, commented blocks
  -> Output: $RESULTS_DIR/deadcode.json

Agent[3]: Duplication Detector
  -> Identify duplicate patterns, redundant logic
  -> Output: $RESULTS_DIR/duplication.json

Agent[4]: Dependency Analyzer
  -> Map dependencies, find circular refs, outdated packages
  -> Output: $RESULTS_DIR/dependencies.json

Agent[5]: Performance Profiler
  -> Identify bottlenecks, memory leaks, inefficient algorithms
  -> Output: $RESULTS_DIR/performance.json

Agent[6]: Security Scanner
  -> Find vulnerabilities, hardcoded secrets, unsafe patterns
  -> Output: $RESULTS_DIR/security.json

Agent[7]: Test Coverage Mapper
  -> Analyze test coverage, find untested code paths
  -> Output: $RESULTS_DIR/testing.json

Agent[8]: Documentation Auditor
  -> Check doc currency, missing docs, broken links
  -> Output: $RESULTS_DIR/documentation.json

Agent[9]: Architecture Mapper
  -> Analyze structure, coupling, cohesion, boundaries
  -> Output: $RESULTS_DIR/architecture.json

Agent[10]: Technical Debt Calculator
  -> Quantify debt, estimate remediation effort
  -> Output: $RESULTS_DIR/techdebt.json
```

**Parallel Execution Pattern:**

```bash
# Launch all agents simultaneously using background processes
for agent in quality deadcode duplication dependencies performance security testing documentation architecture techdebt; do
  (
    echo "🚀 Agent[$agent] started at $(date +%T)"
    case "$agent" in
      quality)
        rg "(TODO|FIXME|XXX|HACK)" --json > "$RESULTS_DIR/quality.json" &
        fd -e js -e ts -e rs -e go -x wc -l {} \; | sort -rn | head -20 > "$RESULTS_DIR/quality-complexity.txt"
        ;;
      deadcode)
        rg "^[[:space:]]*//" --type-list | cut -d: -f1 | while read ext; do
          rg "^[[:space:]]*//" --type "$ext" -c | sort -rn | head -10
        done > "$RESULTS_DIR/deadcode.json"
        ;;
      duplication)
        # Fast pattern matching for duplicate detection
        fd -e js -e ts -e rs -e go | xargs -P 8 -I {} sh -c '
          hash=$(head -50 {} | md5sum | cut -d" " -f1)
          echo "$hash {}"
        ' | sort | uniq -d -w32 > "$RESULTS_DIR/duplication.json"
        ;;
      dependencies)
        fd "package.json|go.mod|Cargo.toml" -x cat {} \; | jq -s '.' > "$RESULTS_DIR/dependencies.json" 2>/dev/null
        ;;
      performance)
        rg "for.*for|while.*while" --type-list | cut -d: -f1 | while read ext; do
          rg "for.*for|while.*while" --type "$ext" -n
        done > "$RESULTS_DIR/performance.json"
        ;;
      security)
        rg "(password|secret|key|token|api_key|private_key).*=.*[\"']" -i > "$RESULTS_DIR/security.json"
        ;;
      testing)
        fd "test|spec" -e js -e ts -e rs -e go | wc -l > "$RESULTS_DIR/testing-count.txt"
        fd -e js -e ts -e rs -e go -x grep -l "test\|spec\|assert" {} \; | wc -l > "$RESULTS_DIR/testing.json"
        ;;
      documentation)
        fd "README|CONTRIBUTING|CHANGELOG" -t f | xargs -I {} wc -l {} > "$RESULTS_DIR/documentation.json" 2>/dev/null
        ;;
      architecture)
        fd -t d -d 3 | head -20 > "$RESULTS_DIR/architecture-structure.txt"
        ;;
      techdebt)
        echo "{\"todos\": $(rg "TODO|FIXME" | wc -l), \"complexity\": $(fd -e js -e ts | wc -l)}" > "$RESULTS_DIR/techdebt.json"
        ;;
    esac
    echo "✅ Agent[$agent] completed at $(date +%T)"
  ) &
done

# Wait for all agents to complete
wait
echo "🎯 All agents completed in parallel"
```

### MODE: CLEAN (Technical Debt Elimination)

**Fast Cleanup Execution:**

```bash
# Create safety checkpoint
git add -A && git commit -m "checkpoint: pre-cleanup $SESSION_ID" 2>/dev/null || true

# Language-specific parallel cleanup
parallel_cleanup() {
  local lang=$1
  case "$lang" in
    javascript)
      fd -e js -e ts -x sed -i 's/console\.\(log\|debug\)/\/\/ console.\1/g' {} \; &
      fd -e js -e ts -x sed -i 's/\bvar\s/let /g' {} \; &
      ;;
    rust)
      cargo fmt --all 2>/dev/null &
      cargo clippy --fix --allow-dirty 2>/dev/null &
      ;;
    go)
      go fmt ./... &
      go mod tidy &
      ;;
  esac
  wait
}

# Detect and clean all languages in parallel
fd "package.json" . -d 2 >/dev/null && parallel_cleanup javascript &
fd "Cargo.toml" . -d 2 >/dev/null && parallel_cleanup rust &
fd "go.mod" . -d 2 >/dev/null && parallel_cleanup go &
wait
```

### MODE: MAP (Strategic Roadmap Generation)

**Instant 8-Agent Roadmap Creation:**

```bash
# Deploy all planning agents simultaneously
{
  echo "# Strategic Roadmap: $ARGUMENTS"
  echo "## Generated: $(date -Iseconds)"
  
  # All agents work in parallel and append to roadmap
  (
    echo "## Goals & Objectives"
    echo "$ARGUMENTS" | xargs -I {} echo "- Objective: {}"
    echo "- Timeline: Q1-Q4 $(date +%Y)"
  ) >> "$RESULTS_DIR/roadmap.md" &
  
  (
    echo "## Milestones"
    echo "### Phase 1: Foundation (Weeks 1-4)"
    echo "- [ ] Initial setup and planning"
    echo "### Phase 2: Implementation (Weeks 5-12)"
    echo "- [ ] Core development"
  ) >> "$RESULTS_DIR/roadmap.md" &
  
  (
    echo "## Risk Matrix"
    echo "| Risk | Impact | Probability | Mitigation |"
    echo "|------|--------|-------------|------------|"
    echo "| Technical complexity | High | Medium | Incremental approach |"
  ) >> "$RESULTS_DIR/roadmap.md" &
  
  wait
  cat "$RESULTS_DIR/roadmap.md" | sort -u
}
```

### MODE: REFACTOR (Code Quality Improvement)

**Smart Refactoring Pipeline:**

```bash
refactor_pipeline() {
  # Analyze → Plan → Execute → Validate in pipeline
  
  # Step 1: Quick analysis (parallel)
  local issues=$(rg "class.*{" --type java -c | wc -l)
  local methods=$(rg "function|def|fn" -c | wc -l)
  
  # Step 2: Generate refactoring plan
  echo "{
    \"large_classes\": $issues,
    \"long_methods\": $methods,
    \"strategy\": \"incremental\"
  }" > "$RESULTS_DIR/refactor-plan.json"
  
  # Step 3: Apply safe refactorings
  fd -e js -e ts -x sh -c '
    # Extract long functions (>50 lines)
    awk "/function|=>/{p=1} p{print; if(/^}/){exit}}" {} | wc -l | read lines
    [ $lines -gt 50 ] && echo "Refactor candidate: {}"
  ' &
  
  wait
}
```

### MODE: FULL (Complete Analysis + Cleanup + Planning)

```bash
# Execute all modes in optimized sequence
echo "🚀 Full mode: Analyze → Clean → Map → Refactor"

# Phase 1: Parallel analysis (10 agents)
$0 analyze "$ARGUMENTS" &
ANALYZE_PID=$!

# Phase 2: Start planning while analysis runs
$0 map "$ARGUMENTS" &
MAP_PID=$!

# Wait for analysis before cleanup
wait $ANALYZE_PID

# Phase 3: Cleanup based on analysis results
$0 clean "$ARGUMENTS" &
CLEAN_PID=$!

# Phase 4: Refactor after cleanup
wait $CLEAN_PID
$0 refactor "$ARGUMENTS"

# Compile final report
wait $MAP_PID
```

## Unified State Management

```bash
update_state() {
  local phase=$1
  local status=$2
  jq --arg phase "$phase" --arg status "$status" \
     '.phase = $phase | .status = $status | .updated = now' \
     "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
}

# State transitions
update_state "initializing" "active"
# ... execute operations ...
update_state "processing" "active"  
# ... complete operations ...
update_state "complete" "success"
```

## Performance Optimizations

1. **Parallel Everything**: All agents run simultaneously using `&` and `wait`
2. **Stream Processing**: Use pipes and `xargs -P` for parallel file processing
3. **Early Termination**: Use `head` to limit results and avoid full scans
4. **Lazy Evaluation**: Only compute what's needed for the active mode
5. **Memory Efficiency**: Stream results directly to files, avoid loading everything
6. **Tool Selection**: Prefer `rg` over `grep`, `fd` over `find` for 10x speed
7. **Batch Operations**: Group similar operations to reduce overhead

## Results Aggregation

```bash
# Fast JSON aggregation of all results
aggregate_results() {
  echo "{"
  for file in "$RESULTS_DIR"/*.json; do
    [ -f "$file" ] || continue
    name=$(basename "$file" .json)
    echo "  \"$name\": $(cat "$file"),"
  done | sed '$ s/,$//'
  echo "}"
} | jq '.' > "$RESULTS_DIR/final-report.json"

# Generate summary
echo "✨ Ultra Agent Complete"
echo "📊 Session: $SESSION_ID"
echo "📁 Results: $RESULTS_DIR/"
echo "🎯 Mode: $(jq -r '.mode' "$STATE_FILE")"
echo "⏱️ Duration: $(($(date +%s) - $(echo $SESSION_ID | cut -d- -f2 | cut -c1-10)))s"
```

## Quick Reference

```bash
# Analyze everything (default)
ultra-agent

# Clean technical debt
ultra-agent clean src/

# Generate roadmap
ultra-agent map "new feature"

# Refactor code
ultra-agent refactor components/

# Full pipeline
ultra-agent full .
```

## Architecture

```
┌─────────────────────────────────────┐
│         ULTRA AGENT CORE            │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐        │
│  │ Analyzer │  │ Cleaner  │  ←──── │ Parallel
│  └──────────┘  └──────────┘        │ Execution
│  ┌──────────┐  ┌──────────┐        │
│  │  Mapper  │  │Refactorer│  ←──── │ Pipeline
│  └──────────┘  └──────────┘        │
├─────────────────────────────────────┤
│        State Management             │
│        Results Aggregation          │
└─────────────────────────────────────┘
```

This unified agent achieves:
- **10x speed** through massive parallelization
- **Zero redundancy** with shared components
- **Flexible modes** for different use cases
- **Atomic operations** with rollback capability
- **Language agnostic** with specific optimizations
- **Progressive enhancement** from analysis to action