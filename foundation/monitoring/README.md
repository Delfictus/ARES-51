# ARES CSF Monitoring & Observability

This directory contains configuration for monitoring and observability of the ARES CSF system.

## Components

### Metrics (Prometheus)
- **Endpoint**: http://localhost:9093
- **Configuration**: `prometheus.yml`
- **Alerts**: `alerts/csf-alerts.yml`

Key metrics collected:
- Packet throughput and latency
- Task scheduling and completion rates
- C-LOGIC component health (DRPP coherence, ADP prediction error, etc.)
- Resource utilization (CPU, memory, GPU)
- Network peer connectivity

### Visualization (Grafana)
- **Endpoint**: http://localhost:3000
- **Default credentials**: admin/admin
- **Dashboards**: `grafana/dashboards/`

Available dashboards:
- CSF Overview: System-wide metrics
- Performance Analysis: Detailed latency breakdowns
- C-LOGIC Components: Individual component monitoring
- Resource Usage: CPU, memory, and GPU utilization

### Distributed Tracing (Jaeger)
- **UI**: http://localhost:16686
- **OTLP endpoint**: localhost:4317

Trace information includes:
- Packet flow through components
- Task execution timeline
- Cross-node communication
- Causality relationships

### Logging
CSF uses structured logging with the following levels:
- `ERROR`: Critical issues requiring immediate attention
- `WARN`: Potential problems or degraded performance
- `INFO`: Normal operational messages
- `DEBUG`: Detailed diagnostic information
- `TRACE`: Very detailed trace information

## Quick Start

### Docker Compose
```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f prometheus grafana jaeger
```

### Kubernetes
```bash
# Deploy Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Deploy CSF ServiceMonitor
kubectl apply -f deployments/kubernetes/monitoring/
```

## Key Metrics

### Performance Metrics
- `csf_packet_latency_seconds`: Histogram of packet processing latency
- `csf_tasks_scheduled_total`: Counter of scheduled tasks
- `csf_tasks_completed_total`: Counter of completed tasks
- `csf_deadline_misses_total`: Counter of missed deadlines

### C-LOGIC Metrics
- `csf_drpp_coherence`: DRPP field coherence (0.0-1.0)
- `csf_adp_prediction_error`: ADP prediction error rate
- `csf_egc_robustness`: EGC STL formula robustness
- `csf_ems_stress_level`: System stress level (0.0-1.0)

### Resource Metrics
- `csf_memory_usage_bytes`: Memory usage
- `csf_cpu_usage_percent`: CPU utilization
- `csf_gpu_usage_percent`: GPU utilization
- `csf_scheduler_utilization`: Scheduler core utilization

## Alert Configuration

Alerts are configured in `alerts/csf-alerts.yml` with the following severities:
- **Critical**: Immediate action required (e.g., deadline misses, security violations)
- **Warning**: Attention needed (e.g., high latency, resource usage)
- **Info**: Informational (e.g., configuration changes)

### Key Alerts
1. **HighPacketLatency**: 99th percentile latency > 10ms
2. **DeadlineMisses**: Tasks missing real-time deadlines
3. **HighTaskVetoRate**: EGC vetoing > 5% of tasks
4. **CryptoVerificationFailures**: SIL integrity check failures
5. **LowDrppCoherence**: Pattern recognition degraded

## Custom Queries

### Packet Processing Performance
```promql
# 99th percentile latency by component
histogram_quantile(0.99, 
  sum(rate(csf_packet_latency_seconds_bucket[5m])) 
  by (component, le)
)

# Packet throughput
sum(rate(csf_packets_published_total[1m]))
```

### Task Scheduling Efficiency
```promql
# Task completion rate
rate(csf_tasks_completed_total[5m])

# Deadline miss rate
rate(csf_deadline_misses_total[5m]) 
/ rate(csf_tasks_scheduled_total[5m])
```

### System Health
```promql
# Overall system health score
(
  csf_drpp_coherence * 0.3 +
  (1 - csf_ems_stress_level) * 0.3 +
  (1 - (rate(csf_deadline_misses_total[5m]) / rate(csf_tasks_scheduled_total[5m]))) * 0.4
)
```

## Troubleshooting

### High Memory Usage
1. Check `csf_memory_usage_bytes` metric
2. Review packet buffer sizes in configuration
3. Look for memory leaks in custom components

### Performance Degradation
1. Check packet latency histograms
2. Review scheduler utilization
3. Analyze trace data for bottlenecks

### Missing Metrics
1. Verify CSF is running with metrics enabled
2. Check Prometheus scrape configuration
3. Ensure network connectivity to metrics endpoint

## Integration with External Systems

### Datadog
```yaml
# datadog-agent.yaml
instances:
  - prometheus_url: http://localhost:9090/metrics
    namespace: "ares_csf"
    metrics:
      - csf_*
```

### New Relic
```yaml
# newrelic-infra.yml
integrations:
  - name: nri-prometheus
    config:
      standalone: true
      servers:
        - static_url: http://localhost:9090
```

### CloudWatch
Use the CloudWatch agent with Prometheus support:
```json
{
  "metrics": {
    "prometheus": {
      "prometheus_config_path": "/etc/prometheus/prometheus.yml",
      "emf_processor": {
        "metric_namespace": "ARES/CSF"
      }
    }
  }
}
```