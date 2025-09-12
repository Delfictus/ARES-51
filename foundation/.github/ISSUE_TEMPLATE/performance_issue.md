---
name: Performance Issue
about: Report a performance problem or regression
title: '[PERF] '
labels: 'performance, needs-triage'
assignees: ''
---

## Performance Issue Description
<!-- Describe the performance problem you're experiencing. -->

## Metrics
- **Expected Performance**: [e.g., 1M packets/sec]
- **Actual Performance**: [e.g., 500K packets/sec]
- **Regression**: [e.g., 50% slower than v0.1.0]

## Profiling Data
<details>
<summary>Flamegraph</summary>

<!-- Attach flamegraph SVG or link -->
</details>

<details>
<summary>Benchmark Results</summary>

```
# Paste benchmark output here
```
</details>

## Configuration
- **Workload Type**: [e.g., sensor fusion, pattern recognition]
- **Packet Rate**: [e.g., 100K/sec]
- **Number of Components**: [e.g., 10 sensors, 5 actuators]
- **Hardware Configuration**: 
  - CPU: [cores, frequency]
  - GPU: [if applicable]
  - Memory: [amount, speed]
  - Network: [bandwidth, latency]

## Reproduction Steps
1. Set up CSF with configuration...
2. Generate workload using...
3. Measure performance with...

## Analysis
<!-- Your analysis of what might be causing the performance issue -->

## Possible Optimizations
<!-- Suggestions for improving performance -->