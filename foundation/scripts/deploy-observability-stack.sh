#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OBSERVABILITY_DIR="$REPO_ROOT/deployments/observability"

NAMESPACE="${OBSERVABILITY_NAMESPACE:-observability}"
ELASTICSEARCH_PASSWORD="${ELASTICSEARCH_PASSWORD:-$(openssl rand -base64 32)}"
JAEGER_QUERY_PASSWORD="${JAEGER_QUERY_PASSWORD:-$(openssl rand -base64 32)}"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

check_prerequisites() {
    log "Checking prerequisites for observability stack deployment..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"
    command -v openssl >/dev/null 2>&1 || error "openssl is required but not installed"
    
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    log "Prerequisites check passed"
}

create_namespace() {
    log "Creating observability namespace..."
    
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl label namespace "$NAMESPACE" \
        name=observability \
        app.kubernetes.io/name=ares-observability \
        app.kubernetes.io/instance=ares-production \
        app.kubernetes.io/version=1.0.0 \
        --overwrite
    
    log "Namespace $NAMESPACE created/updated"
}

setup_elasticsearch_certificates() {
    log "Setting up Elasticsearch certificates..."
    
    # Generate CA certificate
    openssl req -x509 -newkey rsa:4096 -keyout /tmp/ca-key.pem -out /tmp/ca-cert.pem \
        -days 3650 -nodes -subj "/C=US/ST=CA/L=San Francisco/O=ARES Systems/CN=ARES Observability CA"
    
    # Generate Elasticsearch certificate
    openssl req -newkey rsa:4096 -keyout /tmp/elasticsearch-key.pem -out /tmp/elasticsearch-csr.pem \
        -nodes -subj "/C=US/ST=CA/L=San Francisco/O=ARES Systems/CN=elasticsearch.observability.svc.cluster.local"
    
    openssl x509 -req -in /tmp/elasticsearch-csr.pem -CA /tmp/ca-cert.pem -CAkey /tmp/ca-key.pem \
        -CAcreateserial -out /tmp/elasticsearch-cert.pem -days 365 \
        -extensions v3_req -extfile <(cat <<EOF
[v3_req]
subjectAltName = @alt_names
[alt_names]
DNS.1 = elasticsearch
DNS.2 = elasticsearch.observability
DNS.3 = elasticsearch.observability.svc
DNS.4 = elasticsearch.observability.svc.cluster.local
DNS.5 = localhost
IP.1 = 127.0.0.1
EOF
)
    
    # Create PKCS12 keystore
    openssl pkcs12 -export -out /tmp/elastic-certificates.p12 \
        -inkey /tmp/elasticsearch-key.pem -in /tmp/elasticsearch-cert.pem \
        -certfile /tmp/ca-cert.pem -passout pass:changeme
    
    # Create Kubernetes secrets
    kubectl create secret generic elasticsearch-certs \
        --namespace="$NAMESPACE" \
        --from-file=elastic-certificates.p12=/tmp/elastic-certificates.p12 \
        --from-file=ca-cert.pem=/tmp/ca-cert.pem \
        --dry-run=client -o yaml | kubectl apply -f -
    
    kubectl create secret generic elasticsearch-credentials \
        --namespace="$NAMESPACE" \
        --from-literal=password="$ELASTICSEARCH_PASSWORD" \
        --from-literal=username=elastic \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Clean up temporary files
    rm -f /tmp/ca-key.pem /tmp/ca-cert.pem /tmp/elasticsearch-key.pem \
          /tmp/elasticsearch-csr.pem /tmp/elasticsearch-cert.pem /tmp/elastic-certificates.p12
    
    log "Elasticsearch certificates configured"
}

deploy_elasticsearch() {
    log "Deploying Elasticsearch for trace storage..."
    
    kubectl apply -f "$OBSERVABILITY_DIR/elasticsearch-config.yaml"
    
    log "Waiting for Elasticsearch to be ready..."
    kubectl wait --for=condition=ready pod -l app=elasticsearch \
        --namespace="$NAMESPACE" --timeout=300s
    
    log "Elasticsearch deployed successfully"
}

setup_elasticsearch_indices() {
    log "Setting up Elasticsearch indices for ARES traces..."
    
    # Wait for Elasticsearch to be fully ready
    sleep 30
    
    # Port forward to access Elasticsearch
    kubectl port-forward svc/elasticsearch 9200:9200 --namespace="$NAMESPACE" &
    PORT_FORWARD_PID=$!
    
    sleep 10
    
    # Create index templates for ARES quantum traces
    curl -k -u "elastic:$ELASTICSEARCH_PASSWORD" -X PUT \
        "https://localhost:9200/_index_template/ares-quantum-traces" \
        -H "Content-Type: application/json" \
        -d '{
            "index_patterns": ["ares-quantum-*"],
            "template": {
                "settings": {
                    "number_of_shards": 5,
                    "number_of_replicas": 1,
                    "refresh_interval": "1s",
                    "analysis": {
                        "analyzer": {
                            "quantum_analyzer": {
                                "type": "custom",
                                "tokenizer": "keyword",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "traceID": { "type": "keyword" },
                        "spanID": { "type": "keyword" },
                        "parentSpanID": { "type": "keyword" },
                        "operationName": { "type": "keyword" },
                        "startTime": { "type": "date" },
                        "duration": { "type": "long" },
                        "quantum": {
                            "properties": {
                                "coherence_level": { "type": "double" },
                                "entanglement_id": { "type": "keyword" },
                                "temporal_coordinate": { "type": "long" },
                                "operation_type": { "type": "keyword" },
                                "error_correction_level": { "type": "integer" }
                            }
                        },
                        "tags": {
                            "type": "object",
                            "dynamic": true
                        },
                        "process": {
                            "properties": {
                                "serviceName": { "type": "keyword" },
                                "tags": { "type": "object", "dynamic": true }
                            }
                        }
                    }
                }
            },
            "priority": 100
        }'
    
    # Create index template for temporal events
    curl -k -u "elastic:$ELASTICSEARCH_PASSWORD" -X PUT \
        "https://localhost:9200/_index_template/ares-temporal-events" \
        -H "Content-Type: application/json" \
        -d '{
            "index_patterns": ["ares-temporal-*"],
            "template": {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "refresh_interval": "5s"
                },
                "mappings": {
                    "properties": {
                        "timestamp": { "type": "date" },
                        "temporal_coordinate": { "type": "long" },
                        "event_type": { "type": "keyword" },
                        "coherence_level": { "type": "double" },
                        "causal_relationship": {
                            "properties": {
                                "cause_event_id": { "type": "keyword" },
                                "effect_event_id": { "type": "keyword" },
                                "confidence": { "type": "double" },
                                "temporal_lag_femtoseconds": { "type": "long" }
                            }
                        },
                        "anomaly": {
                            "properties": {
                                "deviation_femtoseconds": { "type": "long" },
                                "impact_assessment": { "type": "keyword" },
                                "expected_coordinate": { "type": "long" },
                                "actual_coordinate": { "type": "long" }
                            }
                        }
                    }
                }
            },
            "priority": 100
        }'
    
    # Kill port forward
    kill $PORT_FORWARD_PID || true
    
    log "Elasticsearch indices configured"
}

deploy_jaeger() {
    log "Deploying Jaeger distributed tracing..."
    
    kubectl apply -f "$OBSERVABILITY_DIR/jaeger-config.yaml"
    
    log "Waiting for Jaeger components to be ready..."
    kubectl wait --for=condition=ready pod -l app=jaeger \
        --namespace="$NAMESPACE" --timeout=300s
    
    log "Jaeger deployed successfully"
}

deploy_grafana_quantum_dashboards() {
    log "Deploying Grafana dashboards for quantum observability..."
    
    # Add Grafana Helm repository
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Create Grafana configuration with quantum dashboards
    cat > /tmp/grafana-quantum-values.yaml << EOF
adminPassword: $JAEGER_QUERY_PASSWORD

persistence:
  enabled: true
  size: 10Gi
  storageClassName: fast-ssd

resources:
  requests:
    memory: 512Mi
    cpu: 250m
  limits:
    memory: 1Gi
    cpu: 500m

datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus.monitoring.svc.cluster.local:9090
      access: proxy
      isDefault: true
    - name: Jaeger
      type: jaeger
      url: http://jaeger-query.observability.svc.cluster.local:16686
      access: proxy
    - name: Elasticsearch
      type: elasticsearch
      url: https://elasticsearch.observability.svc.cluster.local:9200
      access: proxy
      basicAuth: true
      basicAuthUser: elastic
      secureJsonData:
        basicAuthPassword: $ELASTICSEARCH_PASSWORD
      jsonData:
        database: "ares-quantum-*"
        timeField: "startTime"
        esVersion: "8.0.0"

dashboardProviders:
  dashboardproviders.yaml:
    apiVersion: 1
    providers:
    - name: 'quantum-dashboards'
      orgId: 1
      folder: 'ARES Quantum'
      type: file
      disableDeletion: false
      updateIntervalSeconds: 10
      allowUiUpdates: true
      options:
        path: /var/lib/grafana/dashboards/quantum

dashboards:
  quantum-dashboards:
    quantum-operations:
      url: https://raw.githubusercontent.com/ares-systems/grafana-dashboards/main/quantum-operations.json
    temporal-analysis:
      url: https://raw.githubusercontent.com/ares-systems/grafana-dashboards/main/temporal-analysis.json
    coherence-monitoring:
      url: https://raw.githubusercontent.com/ares-systems/grafana-dashboards/main/coherence-monitoring.json
    entanglement-tracking:
      url: https://raw.githubusercontent.com/ares-systems/grafana-dashboards/main/entanglement-tracking.json

grafana.ini:
  server:
    root_url: https://observability.ares-internal.com/grafana
    serve_from_sub_path: true
  security:
    admin_user: admin
    admin_password: $JAEGER_QUERY_PASSWORD
    cookie_secure: true
    cookie_samesite: strict
    content_type_protection: true
    x_content_type_options: true
    x_xss_protection: true
  auth:
    disable_login_form: false
    oauth_auto_login: false
  analytics:
    reporting_enabled: false
    check_for_updates: false
  log:
    mode: console
    level: info

ingress:
  enabled: true
  ingressClassName: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rewrite-target: /\$2
  hosts:
  - host: observability.ares-internal.com
    paths:
    - path: /grafana(/|$)(.*)
      pathType: Prefix
  tls:
  - secretName: grafana-tls
    hosts:
    - observability.ares-internal.com

serviceMonitor:
  enabled: true
  namespace: monitoring
  labels:
    app: grafana
    release: prometheus
EOF

    # Deploy Grafana with quantum dashboards
    helm upgrade --install grafana grafana/grafana \
        --namespace="$NAMESPACE" \
        --values=/tmp/grafana-quantum-values.yaml \
        --wait --timeout=300s
    
    rm -f /tmp/grafana-quantum-values.yaml
    
    log "Grafana with quantum dashboards deployed"
}

create_quantum_dashboard_configs() {
    log "Creating quantum-specific dashboard configurations..."
    
    mkdir -p "$OBSERVABILITY_DIR/dashboards"
    
    # Quantum Operations Dashboard
    cat > "$OBSERVABILITY_DIR/dashboards/quantum-operations.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "ARES Quantum Operations",
    "description": "Real-time monitoring of quantum computing operations",
    "tags": ["ares", "quantum", "operations"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Quantum Coherence Level",
        "type": "stat",
        "targets": [
          {
            "expr": "quantum_coherence_ratio",
            "legendFormat": "Coherence",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.7},
                {"color": "green", "value": 0.85}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Active Entanglements",
        "type": "stat",
        "targets": [
          {
            "expr": "quantum_entanglement_count",
            "legendFormat": "Entanglements",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 100},
                {"color": "red", "value": 500}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Quantum Gate Operation Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(quantum_gate_operation_duration_bucket[5m]))",
            "legendFormat": "p50",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.95, rate(quantum_gate_operation_duration_bucket[5m]))",
            "legendFormat": "p95",
            "refId": "B"
          },
          {
            "expr": "histogram_quantile(0.99, rate(quantum_gate_operation_duration_bucket[5m]))",
            "legendFormat": "p99",
            "refId": "C"
          }
        ],
        "yAxes": [
          {
            "unit": "ns",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Temporal Accuracy",
        "type": "gauge",
        "targets": [
          {
            "expr": "quantum_temporal_accuracy",
            "legendFormat": "Accuracy",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0.99,
            "max": 1.0,
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0.99},
                {"color": "yellow", "value": 0.999},
                {"color": "green", "value": 0.9999}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "id": 5,
        "title": "Decoherence Events Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(quantum_decoherence_events_total[5m])",
            "legendFormat": "Decoherence Events/sec",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "unit": "ops",
            "min": 0
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    # Temporal Analysis Dashboard
    cat > "$OBSERVABILITY_DIR/dashboards/temporal-analysis.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "ARES Temporal Analysis",
    "description": "Advanced temporal correlation and causality analysis",
    "tags": ["ares", "temporal", "causality"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Temporal Anomalies",
        "type": "table",
        "targets": [
          {
            "expr": "temporal_anomaly_deviation_femtoseconds",
            "legendFormat": "{{operation_id}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "fs"
          }
        },
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Causal Chain Length Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "causal_chain_length",
            "legendFormat": "Chain Length",
            "refId": "A"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 3,
        "title": "Bootstrap Paradox Risk",
        "type": "gauge",
        "targets": [
          {
            "expr": "bootstrap_paradox_risk",
            "legendFormat": "Risk Level",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 1,
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 0.8}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
EOF

    log "Quantum dashboard configurations created"
}

setup_alertmanager_quantum_rules() {
    log "Setting up AlertManager rules for quantum operations..."
    
    cat > "$OBSERVABILITY_DIR/quantum-alert-rules.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantum-alert-rules
  namespace: $NAMESPACE
  labels:
    app: prometheus
    component: alerting-rules
data:
  quantum-rules.yml: |
    groups:
    - name: quantum.rules
      interval: 10s
      rules:
      - alert: QuantumCoherenceDropped
        expr: quantum_coherence_ratio < 0.85
        for: 30s
        labels:
          severity: warning
          service: ares-chronofabric
          component: quantum-core
        annotations:
          summary: "Quantum coherence has dropped below acceptable threshold"
          description: "Coherence level is {{ \$value | humanizePercentage }} for operation {{ \$labels.operation }}"
          runbook_url: "https://docs.ares-internal.com/runbooks/quantum-coherence"
      
      - alert: CriticalCoherenceLoss
        expr: quantum_coherence_ratio < 0.5
        for: 5s
        labels:
          severity: critical
          service: ares-chronofabric
          component: quantum-core
          escalation: immediate
        annotations:
          summary: "Critical quantum coherence loss detected"
          description: "CRITICAL: Coherence level dropped to {{ \$value | humanizePercentage }} - immediate intervention required"
          runbook_url: "https://docs.ares-internal.com/runbooks/critical-coherence-loss"
      
      - alert: TemporalAnomalyDetected
        expr: increase(temporal_anomaly_count[5m]) > 0
        for: 10s
        labels:
          severity: warning
          service: ares-chronofabric
          component: temporal-core
        annotations:
          summary: "Temporal anomaly detected in quantum operations"
          description: "{{ \$value }} temporal anomalies detected in the last 5 minutes"
          runbook_url: "https://docs.ares-internal.com/runbooks/temporal-anomalies"
      
      - alert: BootstrapParadoxRisk
        expr: bootstrap_paradox_risk > 0.8
        for: 1s
        labels:
          severity: critical
          service: ares-chronofabric
          component: temporal-core
          escalation: immediate
        annotations:
          summary: "High risk of bootstrap paradox detected"
          description: "Bootstrap paradox risk level: {{ \$value | humanizePercentage }} - causal loop detected"
          runbook_url: "https://docs.ares-internal.com/runbooks/bootstrap-paradox"
      
      - alert: QuantumGateLatencyHigh
        expr: histogram_quantile(0.95, rate(quantum_gate_operation_duration_bucket[5m])) > 10000
        for: 2m
        labels:
          severity: warning
          service: ares-chronofabric
          component: quantum-gates
        annotations:
          summary: "Quantum gate operation latency is high"
          description: "95th percentile latency is {{ \$value }}ns, exceeding 10μs threshold"
          runbook_url: "https://docs.ares-internal.com/runbooks/quantum-gate-latency"
      
      - alert: EntanglementCreationFailure
        expr: rate(quantum_entanglement_creation_failures_total[5m]) > 0.1
        for: 1m
        labels:
          severity: warning
          service: ares-chronofabric
          component: entanglement-manager
        annotations:
          summary: "High rate of entanglement creation failures"
          description: "Entanglement creation failure rate: {{ \$value | humanizePercentage }}/sec"
          runbook_url: "https://docs.ares-internal.com/runbooks/entanglement-failures"
      
      - alert: QuantumErrorCorrectionOverload
        expr: rate(quantum_error_correction_operations_total[1m]) > 1000
        for: 30s
        labels:
          severity: warning
          service: ares-chronofabric
          component: error-correction
        annotations:
          summary: "Quantum error correction system under heavy load"
          description: "Error correction rate: {{ \$value }} operations/sec - potential hardware instability"
          runbook_url: "https://docs.ares-internal.com/runbooks/error-correction-overload"
EOF

    kubectl apply -f "$OBSERVABILITY_DIR/quantum-alert-rules.yaml"
    
    log "Quantum alerting rules configured"
}

setup_log_aggregation() {
    log "Setting up enterprise log aggregation with Fluentd..."
    
    cat > "$OBSERVABILITY_DIR/fluentd-config.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: $NAMESPACE
data:
  fluent.conf: |
    # Input from Kubernetes containers
    <source>
      @type tail
      @id in_tail_container_logs
      path /var/log/containers/ares-*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      read_from_head true
      <parse>
        @type json
        time_format %Y-%m-%dT%H:%M:%S.%NZ
      </parse>
    </source>
    
    # Kubernetes metadata filter
    <filter kubernetes.**>
      @type kubernetes_metadata
      @id filter_kube_metadata
      kubernetes_url "#{ENV['KUBERNETES_SERVICE_HOST']}:#{ENV['KUBERNETES_SERVICE_PORT_HTTPS']}"
      verify_ssl "#{ENV['KUBERNETES_VERIFY_SSL'] || true}"
      ca_file "#{ENV['KUBERNETES_CA_FILE']}"
      skip_labels false
      skip_container_metadata false
      skip_master_url false
      skip_namespace_metadata false
    </filter>
    
    # Quantum log enhancement filter
    <filter kubernetes.**>
      @type parser
      @id quantum_log_parser
      key_name message
      reserve_data true
      inject_key_prefix quantum.
      <parse>
        @type regexp
        expression /coherence:\s*(?<coherence_level>\d+\.\d+)|temporal_coord:\s*(?<temporal_coordinate>\d+)|entanglement_id:\s*(?<entanglement_id>[a-f0-9-]+)/
      </parse>
    </filter>
    
    # Correlation ID injection
    <filter kubernetes.**>
      @type record_transformer
      @id correlation_id_injector
      <record>
        correlation_id \${record.dig("kubernetes", "labels", "correlation_id") || SecureRandom.uuid}
        service_name \${record.dig("kubernetes", "labels", "app") || "unknown"}
        quantum_enhanced \${record["quantum.coherence_level"] ? true : false}
      </record>
    </filter>
    
    # Output to Elasticsearch for long-term storage
    <match kubernetes.**>
      @type elasticsearch
      @id out_es
      @log_level info
      include_tag_key true
      host elasticsearch.observability.svc.cluster.local
      port 9200
      scheme https
      ssl_verify false
      user elastic
      password "#{ENV['ELASTICSEARCH_PASSWORD']}"
      
      index_name ares-logs
      type_name _doc
      
      <buffer>
        @type file
        path /var/log/fluentd-buffers/kubernetes.system.buffer
        flush_mode interval
        retry_type exponential_backoff
        flush_thread_count 2
        flush_interval 5s
        retry_forever true
        retry_max_interval 30
        chunk_limit_size 2M
        queue_limit_length 8
        overflow_action block
      </buffer>
      
      # Template for quantum-enhanced logs
      template_name ares_quantum_logs
      template_file /etc/fluent/templates/ares-quantum-template.json
    </match>
    
    # Output to Jaeger for trace correlation
    <match kubernetes.**>
      @type jaeger
      @id out_jaeger
      host jaeger-collector.observability.svc.cluster.local
      port 14268
      
      <buffer>
        @type memory
        flush_mode immediate
      </buffer>
    </match>

  ares-quantum-template.json: |
    {
      "index_patterns": ["ares-logs-*"],
      "template": {
        "settings": {
          "number_of_shards": 3,
          "number_of_replicas": 1,
          "refresh_interval": "5s",
          "analysis": {
            "analyzer": {
              "quantum_message_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "stop"]
              }
            }
          }
        },
        "mappings": {
          "properties": {
            "@timestamp": { "type": "date" },
            "message": { 
              "type": "text",
              "analyzer": "quantum_message_analyzer",
              "fields": {
                "keyword": { "type": "keyword" }
              }
            },
            "level": { "type": "keyword" },
            "service_name": { "type": "keyword" },
            "correlation_id": { "type": "keyword" },
            "quantum_enhanced": { "type": "boolean" },
            "quantum": {
              "properties": {
                "coherence_level": { "type": "double" },
                "temporal_coordinate": { "type": "long" },
                "entanglement_id": { "type": "keyword" },
                "operation_id": { "type": "keyword" }
              }
            },
            "kubernetes": {
              "properties": {
                "namespace": { "type": "keyword" },
                "pod_name": { "type": "keyword" },
                "container_name": { "type": "keyword" },
                "labels": { "type": "object", "dynamic": true }
              }
            }
          }
        }
      }
    }
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: $NAMESPACE
  labels:
    app: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1.16-debian-elasticsearch7-1
        env:
        - name: ELASTICSEARCH_PASSWORD
          valueFrom:
            secretRef:
              name: elasticsearch-credentials
              key: password
        - name: FLUENTD_SYSTEMD_CONF
          value: disable
        - name: FLUENTD_PROMETHEUS_CONF
          value: disable
        resources:
          requests:
            memory: 512Mi
            cpu: 250m
          limits:
            memory: 1Gi
            cpu: 500m
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
        - name: fluentd-config
          mountPath: /fluentd/etc/
        - name: fluentd-templates
          mountPath: /etc/fluent/templates/
      volumes:
      - name: varlog
        hostPath:
          path: /var/log
      - name: varlibdockercontainers
        hostPath:
          path: /var/lib/docker/containers
      - name: fluentd-config
        configMap:
          name: fluentd-config
      - name: fluentd-templates
        configMap:
          name: fluentd-config
          items:
          - key: ares-quantum-template.json
            path: ares-quantum-template.json
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: fluentd
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: fluentd
rules:
- apiGroups: [""]
  resources: ["pods", "namespaces"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: fluentd
roleRef:
  kind: ClusterRole
  name: fluentd
  apiGroup: rbac.authorization.k8s.io
subjects:
- kind: ServiceAccount
  name: fluentd
  namespace: $NAMESPACE
EOF

    kubectl apply -f "$OBSERVABILITY_DIR/fluentd-config.yaml"
    
    log "Enterprise log aggregation configured"
}

deploy_performance_profiler() {
    log "Deploying continuous performance profiler..."
    
    cat > "$OBSERVABILITY_DIR/pprof-config.yaml" << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: continuous-profiler
  namespace: $NAMESPACE
  labels:
    app: continuous-profiler
spec:
  replicas: 2
  selector:
    matchLabels:
      app: continuous-profiler
  template:
    metadata:
      labels:
        app: continuous-profiler
    spec:
      serviceAccountName: continuous-profiler
      containers:
      - name: profiler
        image: parca/parca:v0.20.0
        args:
        - "/parca"
        - "--config-path=/etc/parca/parca.yaml"
        - "--log-level=info"
        - "--http-address=0.0.0.0:7070"
        - "--grpc-address=0.0.0.0:7071"
        ports:
        - containerPort: 7070
          name: http
        - containerPort: 7071
          name: grpc
        env:
        - name: PARCA_KUBERNETES_CLIENT_INSECURE
          value: "true"
        resources:
          requests:
            memory: 1Gi
            cpu: 500m
          limits:
            memory: 2Gi
            cpu: 1
        volumeMounts:
        - name: parca-config
          mountPath: /etc/parca
        - name: parca-data
          mountPath: /var/lib/parca
      volumes:
      - name: parca-config
        configMap:
          name: parca-config
      - name: parca-data
        emptyDir:
          sizeLimit: 50Gi
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: parca-config
  namespace: $NAMESPACE
data:
  parca.yaml: |
    object_storage:
      bucket:
        type: "FILESYSTEM"
        config:
          directory: "/var/lib/parca"
    
    scrape_configs:
    - job_name: 'ares-chronofabric'
      scrape_interval: '10s'
      scrape_timeout: '10s'
      scheme: 'http'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - ares-production
          - ares-staging
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_parca_dev_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_parca_dev_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: \$1:6060
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
    
    - job_name: 'quantum-hardware-profiling'
      scrape_interval: '1s'
      scrape_timeout: '1s'
      scheme: 'http'
      static_configs:
      - targets:
        - 'quantum-hardware-monitor.ares-production.svc.cluster.local:8080'
        labels:
          job: 'quantum-hardware'
          component: 'hardware-monitor'
---
apiVersion: v1
kind: Service
metadata:
  name: continuous-profiler
  namespace: $NAMESPACE
  labels:
    app: continuous-profiler
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 7070
    targetPort: 7070
  - name: grpc
    port: 7071
    targetPort: 7071
  selector:
    app: continuous-profiler
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: continuous-profiler
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: continuous-profiler
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: continuous-profiler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: continuous-profiler
subjects:
- kind: ServiceAccount
  name: continuous-profiler
  namespace: $NAMESPACE
EOF

    kubectl apply -f "$OBSERVABILITY_DIR/pprof-config.yaml"
    
    log "Continuous performance profiler deployed"
}

validate_observability_stack() {
    log "Validating observability stack deployment..."
    
    # Check Jaeger
    kubectl wait --for=condition=ready pod -l app=jaeger --namespace="$NAMESPACE" --timeout=300s
    
    # Check Elasticsearch
    kubectl wait --for=condition=ready pod -l app=elasticsearch --namespace="$NAMESPACE" --timeout=300s
    
    # Check Grafana
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana --namespace="$NAMESPACE" --timeout=300s
    
    # Verify trace ingestion
    log "Testing trace ingestion..."
    kubectl port-forward svc/jaeger-collector 14268:14268 --namespace="$NAMESPACE" &
    PORT_FORWARD_PID=$!
    sleep 5
    
    # Send test trace
    curl -X POST http://localhost:14268/api/traces \
        -H "Content-Type: application/json" \
        -d '{
            "data": [{
                "traceID": "test-trace-12345",
                "spanID": "test-span-67890",
                "operationName": "test-quantum-operation",
                "startTime": '$(date +%s%N)',
                "duration": 1000000,
                "tags": [
                    {"key": "quantum.coherence_level", "value": "0.95"},
                    {"key": "quantum.operation_type", "value": "gate_operation"},
                    {"key": "service.name", "value": "ares-chronofabric"}
                ]
            }]
        }' || warn "Test trace ingestion failed"
    
    kill $PORT_FORWARD_PID || true
    
    log "Observability stack validation completed"
}

create_observability_documentation() {
    log "Creating observability stack documentation..."
    
    cat > "$REPO_ROOT/docs/OBSERVABILITY.md" << 'EOF'
# ARES ChronoFabric Enterprise Observability

## Overview
The ARES ChronoFabric observability stack provides comprehensive monitoring, tracing, and analysis capabilities specifically designed for quantum computing operations and temporal correlation analysis.

## Components

### Distributed Tracing (Jaeger)
- **Endpoint**: https://observability.ares-internal.com/jaeger
- **Quantum Enhancement**: Automatic coherence tracking and entanglement correlation
- **Retention**: 7 days for detailed traces, 30 days for aggregated metrics

### Log Aggregation (Elasticsearch + Fluentd)
- **Elasticsearch**: https://observability.ares-internal.com/elasticsearch
- **Quantum Context**: Automatic extraction of coherence levels and temporal coordinates
- **Correlation**: Cross-service log correlation with quantum operation tracking

### Metrics & Dashboards (Grafana)
- **Grafana**: https://observability.ares-internal.com/grafana
- **Quantum Dashboards**: Real-time coherence monitoring and temporal analysis
- **Business KPIs**: Revenue, customer satisfaction, and SLA tracking

### Performance Profiling (Parca)
- **Profiler**: Continuous CPU and memory profiling
- **Quantum Profiling**: Hardware-specific quantum operation analysis
- **Flame Graphs**: Visual performance analysis with quantum operation highlighting

## Quantum-Specific Features

### Coherence Monitoring
- Real-time coherence level tracking
- Decoherence event detection and alerting
- Environmental factor correlation

### Entanglement Tracking
- Active entanglement state monitoring
- Partner operation correlation
- Entanglement strength decay analysis

### Temporal Analysis
- Femtosecond-precision temporal coordinate tracking
- Causal relationship analysis
- Bootstrap paradox detection

### Performance Correlation
- Quantum operation performance mapping
- Coherence-performance correlation analysis
- Hardware efficiency metrics

## Alert Rules

### Critical Alerts
- **CriticalCoherenceLoss**: Coherence < 50% (immediate escalation)
- **BootstrapParadoxRisk**: Paradox risk > 80% (immediate escalation)

### Warning Alerts
- **QuantumCoherenceDropped**: Coherence < 85% (30s threshold)
- **TemporalAnomalyDetected**: Anomalies in 5m window
- **QuantumGateLatencyHigh**: 95p latency > 10μs

## Usage Examples

### Starting a Traced Operation
```rust
use crate::observability_enhanced::*;

let observability = EnterpriseObservabilityStack::new(TraceConfig::default()).await?;
let trace_id = observability.start_trace("quantum_gate_operation").await?;

// Your quantum operation here
let coherence = perform_quantum_operation().await?;

// Record quantum metrics
observability.record_metric(CustomMetric {
    name: "quantum.coherence_level".to_string(),
    value: coherence,
    timestamp: SystemTime::now(),
    dimensions: vec![],
    metric_type: MetricType::QuantumCoherence,
}).await?;
```

### Performance Profiling
```rust
let profile_id = observability.start_performance_profile("tensor_multiplication").await?;

// Your performance-critical operation
let result = perform_tensor_operation().await?;

let profile = observability.complete_performance_profile(&profile_id).await?;
println!("Operation took {}ns", profile.duration_ns);
```

### Log Correlation
```rust
let events = vec![
    LogEvent {
        id: Uuid::new_v4(),
        timestamp: SystemTime::now(),
        level: "INFO".to_string(),
        message: "Quantum operation started with coherence: 0.95".to_string(),
        service: "quantum-core".to_string(),
        trace_id: Some(trace_id.clone()),
        span_id: None,
        quantum_context: None,
        metadata: HashMap::new(),
    }
];

let correlation_id = observability.correlate_logs(events).await?;
```

## Maintenance

### Daily Tasks
- Review coherence alerts and trends
- Check temporal anomaly reports
- Verify trace ingestion rates
- Monitor storage utilization

### Weekly Tasks
- Analyze performance trends
- Review quantum operation efficiency
- Update alerting thresholds based on baselines
- Clean up old indices and traces

### Monthly Tasks
- Capacity planning for trace storage
- Performance optimization review
- Dashboard and alert rule updates
- Disaster recovery testing

## Troubleshooting

### Common Issues

1. **Missing Traces**
   - Check Jaeger collector logs: `kubectl logs -l app=jaeger,component=collector -n observability`
   - Verify sampling configuration in TraceConfig
   - Check network connectivity to Jaeger endpoint

2. **High Cardinality Metrics**
   - Review dimension cardinality warnings in metrics aggregator
   - Adjust aggregation rules to reduce cardinality
   - Consider sampling for high-volume metrics

3. **Log Correlation Failures**
   - Verify quantum context extraction patterns
   - Check correlation timeout settings
   - Review temporal window configurations

4. **Performance Profiling Issues**
   - Ensure pprof endpoints are exposed on target services
   - Verify Kubernetes RBAC permissions for profiler
   - Check continuous profiler logs for scraping errors

### Emergency Procedures

1. **Critical Coherence Loss**
   - Immediately check quantum hardware status
   - Review recent configuration changes
   - Implement emergency coherence recovery procedures
   - Escalate to quantum engineering team

2. **Bootstrap Paradox Detection**
   - Halt temporal operations immediately
   - Analyze causal chain leading to paradox
   - Implement temporal isolation procedures
   - Consult temporal physics team for resolution

## Contact Information
- **Operations Team**: ops@ares-systems.com
- **Quantum Engineering**: quantum-eng@ares-systems.com  
- **Security Team**: security@ares-systems.com
- **Emergency Escalation**: +1-555-ARES-OPS
EOF

    log "Observability documentation created"
}

main() {
    log "Starting ARES ChronoFabric enterprise observability stack deployment..."
    
    check_prerequisites
    create_namespace
    setup_elasticsearch_certificates
    deploy_elasticsearch
    setup_elasticsearch_indices
    deploy_jaeger
    deploy_grafana_quantum_dashboards
    create_quantum_dashboard_configs
    setup_alertmanager_quantum_rules
    setup_log_aggregation
    deploy_performance_profiler
    validate_observability_stack
    create_observability_documentation
    
    log "Enterprise observability stack deployment completed successfully!"
    log ""
    log "Access URLs:"
    log "- Jaeger UI: https://observability.ares-internal.com/jaeger"
    log "- Grafana: https://observability.ares-internal.com/grafana"
    log "- Elasticsearch: Internal cluster access only"
    log ""
    log "Credentials:"
    log "- Grafana admin password: $JAEGER_QUERY_PASSWORD"
    log "- Elasticsearch password: $ELASTICSEARCH_PASSWORD"
    log ""
    log "Next steps:"
    log "1. Configure quantum-specific alert routing"
    log "2. Set up automated dashboard provisioning"
    log "3. Enable cross-cluster trace federation"
    log "4. Configure long-term metric retention policies"
    log "5. Test emergency escalation procedures"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi